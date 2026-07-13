// The dispatcher's backend abstraction.
//
// The dispatcher core (dispatch.cpp) is backend-agnostic: it flattens a call,
// classifies the leaves, builds the cache key, and drives the LRU + compile
// protocol. Everything a backend has an opinion about lives behind the Engine
// interface defined here:
//
//   * how an aval is read off one of the backend's arrays,
//   * what a cache entry holds (the compile callback's artifacts),
//   * how a call executes against an entry, and how its outputs are wrapped.
//
// Two engines exist. PjrtEngine is the fast path for the "xla" backend: avals
// come off a PJRTBuffer's cached native metadata, inputs are assembled and
// executed natively, and the output buffers are wrapped into AnvlArrays and
// re-nested without leaving C++. ClosureEngine is the generic vehicle for any
// other backend (anvl's quickr, or one pjrt has never heard of): the entry
// holds a compiled R closure that is called on the flat leaves and returns the
// finished value, so the backend keeps full control over execution and
// wrapping. A future native backend would be a third Engine subclass.
//
// How an array leaf is read is the engine's business too (Engine::read_array):
// anvl's AnvlArray contract guarantees only a `$data` field, so the rest of a
// leaf's metadata is obtained the way each backend permits -- PjrtEngine off
// the PJRTBuffer, ClosureEngine through a backend-supplied extractor closure.
// The output-wrapping layout PjrtEngine writes is documented in ?dispatcher.

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dispatch_key.h"
#include "tree.h"

namespace rpjrt {

// What the dispatcher core needs from one array input, read by the engine that
// owns the leaf's backend (Engine::read_array). The core applies input policy
// -- plain-reject, backend-match, the device rules -- to these values; it never
// parses the array's own fields. Only `$data` is guaranteed by anvl's backend
// contract, so `av`/`device`/`backend` come from the buffer (PjrtEngine) or an
// extractor (ClosureEngine), never from a fixed field layout.
struct ArrayLeaf {
  aval av;                 // dtype + shape + ambiguous
  SEXP data = R_NilValue;  // $data: the execute-time input (buffer or R value)
  SEXP device = R_NilValue;  // the R device object: the key's device token
  std::string backend;       // the leaf's backend tag (policy + error messages)
};

// Classification of a bare (class-less) R value as an uploadable
// literal/array leaf. Mirrors anvl's is_valid_r_lit / is_valid_r_array and its
// default dtypes (double -> f32, integer -> i32, logical -> pred -- also
// pjrt_scalar()'s defaults).
struct RDataInfo {
  AnvlDtype dtype = AnvlDtype::kInvalid;
  std::vector<int64_t> shape;  // empty for a rank-0 literal
};

// Nullopt for anything that is not uploadable bare R data -- NA literals
// included: they have no dtype.
std::optional<RDataInfo> classify_rdata(SEXP leaf);

// class(x)[1L], as R would print it, for naming a rejected leaf's type.
std::string r_class_name(SEXP x);

// The subject of a rejection message: "input `x$w[[2]]`". Every leaf has a
// non-empty path, because the dispatcher roots the tree at a ListNode over the
// arguments, so even an unnamed one is reached as `[[1]]`. `leaf_index` is
// 0-based; tree_path() is 1-based.
std::string leaf_subject(const RTree& in_tree, std::size_t leaf_index);

// One execute-time input: a leaf the engine must supply to the compiled
// program, in the order the program expects. Static leaves never appear here --
// they are baked into the program as constants and live only in the cache key
// -- so an engine never sees them, never skips them, and never counts them: the
// inputs to supply are exactly this sequence, and how many there are is its
// size.
struct ExecInput {
  SEXP value = R_NilValue;   // an array leaf's `$data`, or the bare R leaf
  const aval* av = nullptr;  // the leaf's aval; an upload needs its shape
  bool upload = false;       // bare R data: upload it. Else: it is ready to use
};

// An engine's per-entry material -- what the compile callback produced, in the
// engine's own shape. Owned by the CacheEntry; SEXP members must be rooted via
// CacheEntry::preserve().
struct EntryData {
  // Virtual because CacheEntry deletes the derived entry through this base
  // pointer; a non-virtual destructor would skip the derived one (leaking, for
  // instance, PjrtEntry's vectors) and is undefined behaviour besides.
  virtual ~EntryData() = default;
};

// One cache entry: the engine's data plus every R object preserved for the
// entry's lifetime (released on eviction / dispatcher teardown).
//
// Deliberately no releasing destructor: LRUCache calls release_entry (its
// on_evict hook) and then destroys the entry, so a destructor would double-
// release. The flip side is that a stack-local entry leaks its preserves if
// something throws between the first preserve() and cache().set() -- keep
// that window free of throwing operations (today only set()'s own
// std::bad_alloc can fire in it).
struct CacheEntry {
  std::vector<SEXP> keep;
  std::unique_ptr<EntryData> data;

  // Root `x` for this entry's lifetime. R_NilValue is a no-op.
  SEXP preserve(SEXP x) {
    if (x != R_NilValue) {
      R_PreserveObject(x);
      keep.push_back(x);
    }
    return x;
  }
};

inline void release_entry(CacheEntry& e) {
  for (SEXP s : e.keep) R_ReleaseObject(s);
  e.keep.clear();
}

// What a backend implements so the dispatcher core can stay agnostic. One
// engine instance per Dispatcher (an engine may hold per-dispatcher state,
// e.g. PjrtEngine's reusable execute options, or the canonical-device table).
class Engine {
 public:
  virtual ~Engine();

  // The canonical representative of a device object; its address is the
  // cache key's DeviceToken. Resolution is by object identity first -- free
  // for a backend that interns its devices -- and identical() as a fallback,
  // so a backend handing out equal-but-distinct device objects still lands
  // on one token rather than splitting the cache or conflicting with itself.
  // Canonical objects are preserved for the dispatcher's lifetime (that is
  // what keeps a token's address stable), so a backend should hand out few
  // distinct device objects, not fabricate unequal ones per call. Virtual so
  // a native engine with its own device-identity scheme can replace it.
  virtual SEXP canonical_device(SEXP device);

  // Read one array input into the material the core needs. Nullopt when `leaf`
  // is not an AnvlArray this engine can read, so the core falls through to
  // classify_rdata(). The engine reads only what it uses and only what the
  // leaf's backend permits: PjrtEngine takes dtype/shape off the PJRTBuffer and
  // never consults `$dtype`/`$shape`; ClosureEngine defers to the backend's
  // extractor. Backend-specific structural validation (e.g. "$data must be a
  // PJRTBuffer") is done only once the leaf's backend tag matches this
  // engine's, so the core's plain-reject / backend-match policy still reports
  // first for a foreign leaf. `in_tree`/`leaf_index` name the leaf on
  // rejection.
  virtual std::optional<ArrayLeaf> read_array(SEXP leaf, const RTree& in_tree,
                                              std::size_t leaf_index) = 0;

  // Build the entry's data from the compile callback's result. Must validate
  // and extract everything that can throw BEFORE the first preserve() (`res`
  // keeps the SEXPs rooted meanwhile), so a malformed result never leaks a
  // half-preserved entry.
  virtual void build_entry(const Rcpp::List& res, CacheEntry& e) const = 0;

  // Execute one call against a cached entry and return the finished R value.
  // `inputs` is the call's execute-time inputs, already classified and in
  // order (see ExecInput): the engine supplies exactly these, and knows
  // nothing of cache keys or static arguments. Placing an input on the entry's
  // device under the pin policy is the engine's own business (it is the only
  // party that knows what `$data` is), so the policy is engine state, not a
  // per-call argument.
  //
  // The entry is const: build_entry() constructs it whole, so execution only
  // ever reads it. That is what keeps the preserve() rooting discipline off
  // the hot path -- nothing here can add to the entry's `keep`.
  virtual SEXP run(const CacheEntry& e,
                   const std::vector<ExecInput>& inputs) const = 0;

 private:
  // The distinct devices this dispatcher has seen, each preserved; owned by
  // the default canonical_device().
  std::vector<SEXP> canonical_devices_;
};

// engine_name is the R-facing selector: "pjrt" or "closure"; throws on any
// other value. `backend` is the tag the dispatcher's arrays carry (the engine
// needs it to stamp the arrays it wraps and to recognize its own leaves).
// `move_inputs` is the dispatcher's device policy, fixed for the engine's
// lifetime: under it the entry's device is pinned by the compile callback and
// every engine places its own inputs on it -- PjrtEngine by copying the buffer,
// ClosureEngine by leaving it to the `r_fun` the backend compiled. `extractor`
// is the R closure ClosureEngine calls to read a leaf's metadata via the
// backend's accessors (required for "closure"; ignored, pass R_NilValue, for
// "pjrt", which reads the PJRTBuffer directly). See ?dispatcher.
std::unique_ptr<Engine> make_engine(const std::string& engine_name,
                                    std::string backend, bool move_inputs,
                                    SEXP extractor);

}  // namespace rpjrt

// Native eager-dispatch hot path (see anvl/benchmarks/cpp-hot-path-design.md).
//
// The RTree lives in tree.h (shared with the exposed Rtree API in tree.cpp);
// this file uses it on the stack to flatten a call's arguments to a leaf list
// and to use the structure as cache-key material.

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "buffer.h"
#include "client.h"
#include "device.h"
#include "hash.h"
#include "lru_cache.h"
#include "pjrt_impl.h"
#include "tree.h"

namespace rpjrt {

// Per-leaf abstract value -- mirrors anvl's nv_aval(dtype, shape, ambiguous).
// dtype/shape are genuine PJRT_Buffer properties read natively; `ambiguous` is
// an anvl type-system bit supplied per leaf (pjrt folds it into the key but
// never interprets it). `device` is NOT per-aval: it is a single per-call value
// on the cache_key (matching anvl).
struct aval {
  PJRT_Buffer_Type dtype = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> shape;
  bool ambiguous = false;
};

// One leaf of the cache key.
//   kBuffer    device buffer (xla AnvlArray): keyed by its aval.
//   kStatic    static arg: keyed by value via R's identical().
//   kRData     bare R literal/array: keyed by (default dtype, shape); it is
//              uploaded to the entry's device at execute time (pjrt engine)
//              or passed through as-is (closure engine).
//   kClosureArr R-array-backed AnvlArray (closure engine): keyed by its
//              cached $dtype object (identical()), $shape, and $ambiguous;
//              its $data is the execute-time input.
//   kOpaque    anything else: keyed by value via identical(). Never
//              executable -- the compile callback (the full R prepare +
//              compile path) raises the canonical error for it, so such a
//              call errors on every (always-miss or never-valid) attempt.
struct KeyLeaf {
  enum Kind { kBuffer, kStatic, kRData, kClosureArr, kOpaque };
  Kind kind = kBuffer;
  aval av;                  // kBuffer / kRData / kClosureArr (shape+ambiguous)
  SEXP value = R_NilValue;  // kStatic / kOpaque: the leaf; kClosureArr: $dtype
};

// The executable-cache key -- mirrors anvl's list(in_tree, key_leaves, device).
struct CacheKey {
  RTree in_tree;
  std::vector<KeyLeaf> leaves;
  const void* device = nullptr;  // canonical device id (PJRT_Device*), or null
};

static std::uint64_t aval_hash(const aval& a) {
  std::uint64_t h = static_cast<std::uint64_t>(a.dtype);
  h = hash_combine(h, a.ambiguous ? 1u : 0u);
  h = hash_combine(h, a.shape.size());
  for (int64_t d : a.shape) {
    h = hash_combine(h, static_cast<std::uint64_t>(d));
  }
  return h;
}

static bool aval_eq(const aval& a, const aval& b) {
  return a.dtype == b.dtype && a.ambiguous == b.ambiguous && a.shape == b.shape;
}

// CacheKeyHash and CacheKeyEq are functors (types with operator()) rather than
// plain functions because unordered_map -- and LRUCache, which forwards them --
// take the Hash and Eq as template *type* parameters. Passing them as types
// lets the map default-construct them and inline each call, with no indirect
// call through a function pointer.
struct CacheKeyHash {
  // unordered_map's Hash concept requires std::size_t, so the 64-bit
  // accumulator is narrowed on return (a no-op on the 64-bit platforms we build
  // for).
  std::size_t operator()(const CacheKey& k) const {
    std::uint64_t h = tree_hash(k.in_tree);
    h = hash_combine(h, reinterpret_cast<std::uintptr_t>(k.device));
    h = hash_combine(h, k.leaves.size());
    for (const KeyLeaf& leaf : k.leaves) {
      h = hash_combine(h, static_cast<std::uint64_t>(leaf.kind));
      switch (leaf.kind) {
        case KeyLeaf::kBuffer:
        case KeyLeaf::kRData:
          h = hash_combine(h, aval_hash(leaf.av));
          break;
        case KeyLeaf::kClosureArr:
          // The dtype SEXP gets a cheap discriminator only (exact equality
          // falls back to identical()); shape/ambiguity hash natively.
          h = hash_combine(h, static_cast<std::uint64_t>(TYPEOF(leaf.value)));
          h = hash_combine(h, aval_hash(leaf.av));
          break;
        case KeyLeaf::kStatic:
        case KeyLeaf::kOpaque:
          // Cheap discriminator only; exact equality falls back to
          // identical().
          h = hash_combine(h, 0x57A71Cu);
          h = hash_combine(h, static_cast<std::uint64_t>(TYPEOF(leaf.value)));
          h = hash_combine(h,
                           static_cast<std::uint64_t>(Rf_xlength(leaf.value)));
          break;
      }
    }
    return static_cast<std::size_t>(h);
  }
};

// R's default identical(): R_compute_identical's flag bits are USE bits
// (identical.c); the default sets only IDENT_USE_CLOENV (16), i.e. compare
// closure environments but ignore bytecode/srcref. flags=0 would ignore
// environments and wrongly merge distinct closures.
static inline bool r_identical(SEXP a, SEXP b) {
  return R_compute_identical(a, b, /*flags=*/16);
}

struct CacheKeyEq {
  bool operator()(const CacheKey& a, const CacheKey& b) const {
    if (!tree_eq(a.in_tree, b.in_tree)) return false;
    if (a.device != b.device) return false;
    if (a.leaves.size() != b.leaves.size()) return false;
    for (std::size_t k = 0; k < a.leaves.size(); ++k) {
      const KeyLeaf& x = a.leaves[k];
      const KeyLeaf& y = b.leaves[k];
      if (x.kind != y.kind) return false;
      switch (x.kind) {
        case KeyLeaf::kBuffer:
        case KeyLeaf::kRData:
          if (!aval_eq(x.av, y.av)) return false;
          break;
        case KeyLeaf::kClosureArr:
          if (!aval_eq(x.av, y.av)) return false;
          if (!r_identical(x.value, y.value)) return false;
          break;
        case KeyLeaf::kStatic:
        case KeyLeaf::kOpaque:
          // Match anvl, which compares static args with R's default
          // identical().
          if (!r_identical(x.value, y.value)) return false;
          break;
      }
    }
    return true;
  }
};

// Read (dtype, shape) off a PJRTBuffer xptr and (device) as its canonical id.
// Native reads via pjrt's own buffer class -- no R round-trip. Uses the
// buffer's cached metadata so repeated dispatches of the same buffers skip
// the PJRT C API calls entirely.
static aval aval_from_buffer(SEXP buf_xptr, bool ambiguous,
                             const void** out_device) {
  Rcpp::XPtr<PJRTBuffer> buf(buf_xptr);
  aval a;
  a.dtype = buf->element_type();
  a.shape = buf->dimensions();
  a.ambiguous = ambiguous;
  if (out_device) {
    *out_device = static_cast<const void*>(buf->device_ptr());
  }
  return a;
}

// The relevant fields of an AnvlArray leaf -- an R list with class
// "AnvlArray" from which we read `$data`, `$backend`, `$ambiguous`, and the
// cached `$dtype`/`$shape`.
struct AnvlFields {
  bool is_anvl = false;
  SEXP data = R_NilValue;
  SEXP dtype = R_NilValue;
  SEXP shape = R_NilValue;
  const char* backend = nullptr;
  bool ambiguous = false;
};

static AnvlFields anvl_fields(SEXP leaf) {
  AnvlFields f;
  if (TYPEOF(leaf) != VECSXP || !Rf_inherits(leaf, "AnvlArray")) return f;
  SEXP nms = Rf_getAttrib(leaf, R_NamesSymbol);
  if (nms == R_NilValue) return f;
  SEXP amb = R_NilValue;
  for (R_xlen_t k = 0; k < XLENGTH(leaf); ++k) {
    const char* nm = CHAR(STRING_ELT(nms, k));
    if (!std::strcmp(nm, "data"))
      f.data = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "backend"))
      f.backend = TYPEOF(VECTOR_ELT(leaf, k)) == STRSXP
                      ? CHAR(STRING_ELT(VECTOR_ELT(leaf, k), 0))
                      : nullptr;
    else if (!std::strcmp(nm, "ambiguous"))
      amb = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "dtype"))
      f.dtype = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "shape"))
      f.shape = VECTOR_ELT(leaf, k);
  }
  f.is_anvl = true;
  f.ambiguous = (amb != R_NilValue && Rf_asLogical(amb) == TRUE);
  return f;
}

// Classify a bare (class-less) R value as an uploadable literal/array leaf.
// Mirrors anvl's is_valid_r_lit / is_valid_r_array and its default dtypes
// (double -> f32, integer -> i32, logical -> pred -- also pjrt_scalar()'s
// defaults). Returns false for anything else (NA literals included: they
// have no dtype).
struct RDataInfo {
  bool ok = false;
  PJRT_Buffer_Type dtype = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> shape;  // empty for a rank-0 literal
};

static RDataInfo classify_rdata(SEXP leaf) {
  RDataInfo info;
  const SEXPTYPE t = TYPEOF(leaf);
  if (Rf_isObject(leaf)) return info;  // classed values are not bare R data
  switch (t) {
    case REALSXP:
      info.dtype = PJRT_Buffer_Type_F32;
      break;
    case INTSXP:
      info.dtype = PJRT_Buffer_Type_S32;
      break;
    case LGLSXP:
      info.dtype = PJRT_Buffer_Type_PRED;
      break;
    default:
      return info;
  }
  SEXP dim = Rf_getAttrib(leaf, R_DimSymbol);
  if (dim != R_NilValue) {
    // an array of any rank; NA elements pass through like pjrt_buffer()'s
    const R_xlen_t n = XLENGTH(dim);
    info.shape.reserve(n);
    for (R_xlen_t k = 0; k < n; ++k) info.shape.push_back(INTEGER(dim)[k]);
    info.ok = true;
    return info;
  }
  if (XLENGTH(leaf) != 1) return info;  // bare vector: not a valid input
  // a scalar literal; reject NA (it has no dtype), keep NaN
  switch (t) {
    case REALSXP:
      if (R_IsNA(REAL(leaf)[0])) return info;
      break;
    case INTSXP:
      if (INTEGER(leaf)[0] == NA_INTEGER) return info;
      break;
    case LGLSXP:
      if (LOGICAL(leaf)[0] == NA_LOGICAL) return info;
      break;
    default:
      break;
  }
  info.ok = true;
  return info;
}

// One output-donation phantom buffer to allocate per call (CPU memory mgmt).
struct PhantomSpec {
  std::string dtype;  // pjrt dtype string (e.g. "f32")
  std::vector<int64_t> shape;
};

// What a compiled program needs at execute time -- mirrors anvl's stored cache
// value list(exec, out_tree, const_arrays, ambiguous_out, device,
// phantom_specs). out_tree / ambiguous_out are opaque R objects the caller uses
// to wrap the raw output buffers (this engine does not know AnvlArray layout).
// client / device_xptr are kept to allocate fresh phantom buffers per call.
struct CacheEntry {
  SEXP exec = R_NilValue;   // PJRTLoadedExecutable xptr (preserved)
  SEXP r_fun = R_NilValue;  // closure engine: compiled R closure (preserved)
  std::vector<SEXP> const_arrays;       // (preserved)
  std::vector<SEXP> static_key_values;  // key-leaf SEXPs (preserved)
  std::vector<PhantomSpec> phantom_specs;
  SEXP client = R_NilValue;         // PJRTClient xptr for phantom alloc/uploads
  SEXP device_xptr = R_NilValue;    // PJRTDevice xptr for phantom alloc/uploads
  SEXP out_tree = R_NilValue;       // opaque, for the caller's wrap
  SEXP ambiguous_out = R_NilValue;  // opaque, for the caller's wrap
  // Per-output dtype strings / integer shapes (preserved). Fixed for a given
  // executable, so they are read off the first call's output buffers and
  // served from the entry afterwards.
  SEXP out_dtypes = R_NilValue;
  SEXP out_shapes = R_NilValue;
  const void* device = nullptr;
};

// Release every R object a cache entry preserves (on eviction / teardown).
static void release_entry(CacheEntry& e) {
  if (e.exec != R_NilValue) R_ReleaseObject(e.exec);
  if (e.r_fun != R_NilValue) R_ReleaseObject(e.r_fun);
  for (SEXP c : e.const_arrays) R_ReleaseObject(c);
  for (SEXP c : e.static_key_values) R_ReleaseObject(c);
  if (e.client != R_NilValue) R_ReleaseObject(e.client);
  if (e.device_xptr != R_NilValue) R_ReleaseObject(e.device_xptr);
  if (e.out_tree != R_NilValue) R_ReleaseObject(e.out_tree);
  if (e.ambiguous_out != R_NilValue) R_ReleaseObject(e.ambiguous_out);
  if (e.out_dtypes != R_NilValue) R_ReleaseObject(e.out_dtypes);
  if (e.out_shapes != R_NilValue) R_ReleaseObject(e.out_shapes);
}

// Per-jit pjrt_dispatcher: owns the executable cache, the R cache_miss callback
// (compiles on a miss), and a reusable default-options object. R objects held
// in the cache are R_PreserveObject'd on insert and released on eviction /
// teardown via the cache's on_evict hook + clear().
class pjrt_dispatcher {
 public:
  pjrt_dispatcher(std::size_t capacity, SEXP miss_fn, SEXP opts,
                  std::unordered_set<std::string> static_names,
                  bool closure_engine, bool move_inputs)
      : cache_(capacity, release_entry),
        miss_fn_(miss_fn),
        opts_(opts),
        static_names_(std::move(static_names)),
        closure_engine_(closure_engine),
        move_inputs_(move_inputs) {
    R_PreserveObject(miss_fn_);
    R_PreserveObject(opts_);
  }

  ~pjrt_dispatcher() {
    cache_.clear();
    R_ReleaseObject(miss_fn_);
    R_ReleaseObject(opts_);
  }

  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq>& cache() {
    return cache_;
  }
  SEXP miss_fn() const { return miss_fn_; }
  SEXP opts() const { return opts_; }
  const std::unordered_set<std::string>& static_names() const {
    return static_names_;
  }
  // Closure engine: entries hold a compiled R closure called on the flat
  // leaves (anvl's quickr backend) instead of a PJRT executable.
  bool closure_engine() const { return closure_engine_; }
  // Move policy: a target device is fixed per entry (jit(device = ) /
  // device_arg), so buffer inputs are copied to it at execute time and the
  // key carries no device.
  bool move_inputs() const { return move_inputs_; }

 private:
  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq> cache_;
  SEXP miss_fn_;
  SEXP opts_;
  std::unordered_set<std::string> static_names_;
  bool closure_engine_;
  bool move_inputs_;
};

// An unnamed ListNode over `n` leaf children, in RTree's flat preorder
// encoding: the root, then one leaf node per child. Leaf indices are implicit
// (a leaf's rank in the preorder), so nothing per-leaf is stored.
static RTree flat_leaf_tree(std::size_t n) {
  RTree t;
  t.kind.reserve(n + 1);
  t.n_children.reserve(n + 1);
  t.subtree_nodes.reserve(n + 1);
  t.name_off.reserve(n + 1);
  t.kind.push_back(RTree::ListNode);
  t.n_children.push_back(static_cast<std::int32_t>(n));
  t.subtree_nodes.push_back(static_cast<std::int32_t>(n + 1));
  t.name_off.push_back(-1);
  for (std::size_t k = 0; k < n; ++k) {
    t.kind.push_back(RTree::LeafNode);
    t.n_children.push_back(0);
    t.subtree_nodes.push_back(1);
    t.name_off.push_back(-1);
  }
  return t;
}

// Build a CacheKey from a flat list of dynamic leaves, each a
// list(buffer_xptr, ambiguous), with a flat ListNode in_tree over them.
static CacheKey build_key_from_leaves(Rcpp::List leaves) {
  CacheKey key;
  key.in_tree = flat_leaf_tree(static_cast<std::size_t>(leaves.size()));
  key.leaves.reserve(leaves.size());
  for (R_xlen_t k = 0; k < leaves.size(); ++k) {
    Rcpp::List leaf = leaves[k];
    SEXP buf = leaf[0];
    bool ambiguous = Rcpp::as<bool>(leaf[1]);
    const void* dev = nullptr;
    KeyLeaf kl;
    kl.kind = KeyLeaf::kBuffer;
    kl.av = aval_from_buffer(buf, ambiguous, &dev);
    key.leaves.push_back(std::move(kl));
    // device is a single per-call value; take it from the first leaf.
    if (k == 0) key.device = dev;
  }
  return key;
}

}  // namespace rpjrt

// Self-test for aval/CacheKey hashing. Returns the 64-bit key hash as a decimal
// string (a double would lose the low bits that distinguish keys). Each leaf is
// a list(buffer_xptr, ambiguous).
// [[Rcpp::export]]
std::string impl_dispatch_key_hash(Rcpp::List leaves) {
  rpjrt::CacheKey key = rpjrt::build_key_from_leaves(leaves);
  return std::to_string(rpjrt::CacheKeyHash{}(key));
}

// Self-test for aval/CacheKey equality (what actually governs cache hits).
// [[Rcpp::export]]
bool impl_dispatch_key_eq(Rcpp::List a, Rcpp::List b) {
  rpjrt::CacheKey ka = rpjrt::build_key_from_leaves(a);
  rpjrt::CacheKey kb = rpjrt::build_key_from_leaves(b);
  return rpjrt::CacheKeyEq{}(ka, kb);
}

// Self-test for static-arg cache-key equality. Each element of `a`/`b` is
// treated as a static KeyLeaf value (compared via identical()); the in_tree is
// a flat ListNode over them. Exercises value- and environment-sensitivity.
// [[Rcpp::export]]
bool impl_dispatch_static_key_eq(Rcpp::List a, Rcpp::List b) {
  auto build = [](Rcpp::List vals) {
    rpjrt::CacheKey key;
    key.in_tree = rpjrt::flat_leaf_tree(static_cast<std::size_t>(vals.size()));
    key.leaves.reserve(vals.size());
    for (R_xlen_t k = 0; k < vals.size(); ++k) {
      rpjrt::KeyLeaf kl;
      kl.kind = rpjrt::KeyLeaf::kStatic;
      SEXP v = vals[k];
      kl.value = v;
      key.leaves.push_back(std::move(kl));
    }
    return key;
  };
  rpjrt::CacheKey ka = build(a);
  rpjrt::CacheKey kb = build(b);
  return rpjrt::CacheKeyEq{}(ka, kb);
}

// ---- pjrt_dispatcher: the native eager-dispatch hot path
// -------------------------

// The sentinel returned by impl_dispatch_run() when the call is not handled by
// the native fast path (the R caller must fall back to its slow path). A unique
// symbol, identity-comparable on the R side.
// [[Rcpp::export]]
SEXP impl_dispatch_sentinel() {
  return Rf_install("__pjrt_dispatch_sentinel__");
}

// Create a pjrt_dispatcher for one jitted function. `miss_fn(args)` is the R
// callback that compiles on a cache miss and returns a list with at least
// `exec` (a PJRTLoadedExecutable xptr) and optionally `const_arrays` (a list of
// buffer xptrs prepended to the inputs). `capacity` is the executable-cache
// size. `static_names` are the top-level argument names whose values are
// static (part of the cache key, excluded from execution).
// [[Rcpp::export]]
SEXP impl_dispatch_create(int capacity, SEXP miss_fn, SEXP static_names,
                          std::string engine, bool move_inputs) {
  using namespace rpjrt;
  if (TYPEOF(miss_fn) != CLOSXP && TYPEOF(miss_fn) != BUILTINSXP &&
      TYPEOF(miss_fn) != SPECIALSXP) {
    Rcpp::stop("miss_fn must be a function");
  }
  bool closure_engine;
  if (engine == "pjrt") {
    closure_engine = false;
  } else if (engine == "closure") {
    closure_engine = true;
  } else {
    Rcpp::stop("engine must be \"pjrt\" or \"closure\"");
  }
  std::unordered_set<std::string> statics;
  if (static_names != R_NilValue && TYPEOF(static_names) == STRSXP) {
    const R_xlen_t n = XLENGTH(static_names);
    for (R_xlen_t k = 0; k < n; ++k) {
      statics.insert(std::string(CHAR(STRING_ELT(static_names, k))));
    }
  }
  Rcpp::XPtr<PJRTExecuteOptions> opts =
      impl_execution_options_create(std::vector<int64_t>(), 0);
  auto* d = new pjrt_dispatcher(static_cast<std::size_t>(capacity), miss_fn,
                                static_cast<SEXP>(opts), std::move(statics),
                                closure_engine, move_inputs);
  Rcpp::XPtr<pjrt_dispatcher> ptr(d, true);
  ptr.attr("class") = "Dispatcher";
  return ptr;
}

// Number of compiled executables currently cached by the pjrt_dispatcher.
// [[Rcpp::export]]
int impl_dispatch_size(SEXP dispatcher) {
  Rcpp::XPtr<rpjrt::pjrt_dispatcher> d(dispatcher);
  return static_cast<int>(d->cache().size());
}

// Run the native dispatch for `args` (the evaluated argument list of the
// call). PJRT engine: returns the raw output buffers plus wrap material.
// Closure engine: returns list(value = <closure result>). The sentinel is
// returned only for a device conflict under the infer policy -- the caller
// re-runs its own validation to raise the canonical error.
// [[Rcpp::export]]
SEXP impl_dispatch_run(SEXP dispatcher, Rcpp::List args) {
  using namespace rpjrt;
  Rcpp::XPtr<pjrt_dispatcher> d(dispatcher);
  const std::unordered_set<std::string>& statics = d->static_names();

  // 1. Flatten args into leaves + structure, marking static leaves. Static-ness
  // is decided at the top level by argument name and propagated to every leaf
  // beneath a static arg (mirrors anvl's static-arg marking).
  std::vector<SEXP> leaves;
  std::vector<char> is_static;
  const R_xlen_t n_args = args.size();
  SEXP arg_nms = Rf_getAttrib(args, R_NamesSymbol);
  const bool has_names = (arg_nms != R_NilValue);

  // Emit the root ListNode over the args, then let flatten_rec append each
  // arg's subtree after it (preorder). The root's names are pushed first, so
  // the arg names occupy in_tree.names[0, n_args) even once nested named lists
  // append their own names behind them.
  RTree in_tree;
  in_tree.kind.push_back(RTree::ListNode);
  in_tree.n_children.push_back(static_cast<std::int32_t>(n_args));
  in_tree.subtree_nodes.push_back(0);  // backpatched once the args are emitted
  if (has_names) {
    in_tree.name_off.push_back(0);
    for (R_xlen_t k = 0; k < n_args; ++k) {
      in_tree.names.push_back(std::string(CHAR(STRING_ELT(arg_nms, k))));
    }
  } else {
    in_tree.name_off.push_back(-1);
  }
  for (R_xlen_t k = 0; k < n_args; ++k) {
    bool child_static =
        !statics.empty() && has_names && statics.count(in_tree.names[k]) > 0;
    flatten_rec(VECTOR_ELT(args, k), leaves, in_tree);
    // Static-ness is a dispatch-only per-leaf overlay, not part of the shared
    // Rtree API: mark every leaf this arg just contributed. is_static grows in
    // lockstep with leaves, so resize fills the new tail with child_static
    // (an arg with no leaves -- NULL / empty list -- is a no-op).
    is_static.resize(leaves.size(), child_static ? 1 : 0);
  }
  in_tree.subtree_nodes[0] = static_cast<std::int32_t>(in_tree.kind.size());

  // 2. Classify leaves into key leaves + per-leaf execute material. The only
  // case left to the sentinel is a device conflict under the infer policy
  // (the caller re-runs its R validation to raise the canonical error);
  // everything else either dispatches or errors via the compile callback.
  const bool closure = d->closure_engine();
  const bool move = d->move_inputs();
  CacheKey key;
  key.in_tree = in_tree;
  key.leaves.reserve(leaves.size());
  // Per-leaf execute-time SEXP: the buffer xptr (kBuffer), the backing R
  // array `$data` (kClosureArr), or the leaf itself (everything else).
  std::vector<SEXP> exec_sexp(leaves.size());
  bool have_device = false;
  bool needs_upload = false;
  for (std::size_t k = 0; k < leaves.size(); ++k) {
    SEXP leaf = leaves[k];
    exec_sexp[k] = leaf;
    KeyLeaf kl;
    if (is_static[k]) {
      kl.kind = KeyLeaf::kStatic;
      kl.value = leaf;
      key.leaves.push_back(std::move(kl));
      continue;
    }
    AnvlFields af = anvl_fields(leaf);
    if (af.is_anvl) {
      if (!closure && af.backend != nullptr &&
          std::strcmp(af.backend, "xla") == 0 && TYPEOF(af.data) == EXTPTRSXP &&
          Rf_inherits(af.data, "PJRTBuffer")) {
        const void* dev = nullptr;
        kl.kind = KeyLeaf::kBuffer;
        kl.av = aval_from_buffer(af.data, af.ambiguous, move ? nullptr : &dev);
        if (!move) {
          // Infer policy: the first buffer's device is the call's device;
          // a conflicting input is the caller's validation error.
          if (!have_device) {
            key.device = dev;
            have_device = true;
          } else if (dev != key.device) {
            return impl_dispatch_sentinel();
          }
        }
        exec_sexp[k] = af.data;
        key.leaves.push_back(std::move(kl));
        continue;
      }
      if (closure && af.backend != nullptr &&
          (std::strcmp(af.backend, "quickr") == 0 ||
           std::strcmp(af.backend, "plain") == 0) &&
          af.dtype != R_NilValue && TYPEOF(af.shape) == INTSXP) {
        kl.kind = KeyLeaf::kClosureArr;
        kl.value = af.dtype;
        kl.av.ambiguous = af.ambiguous;
        const R_xlen_t nd = XLENGTH(af.shape);
        kl.av.shape.reserve(nd);
        for (R_xlen_t j = 0; j < nd; ++j) {
          kl.av.shape.push_back(INTEGER(af.shape)[j]);
        }
        exec_sexp[k] = af.data;
        key.leaves.push_back(std::move(kl));
        continue;
      }
      // An AnvlArray this engine cannot execute (wrong backend): opaque.
      kl.kind = KeyLeaf::kOpaque;
      kl.value = leaf;
      key.leaves.push_back(std::move(kl));
      continue;
    }
    RDataInfo rd = classify_rdata(leaf);
    if (rd.ok) {
      kl.kind = KeyLeaf::kRData;
      kl.av.dtype = rd.dtype;
      kl.av.shape = std::move(rd.shape);
      kl.av.ambiguous = true;  // bare R data is dtype-ambiguous (to_avals)
      if (!closure) needs_upload = true;
      key.leaves.push_back(std::move(kl));
      continue;
    }
    kl.kind = KeyLeaf::kOpaque;
    kl.value = leaf;
    key.leaves.push_back(std::move(kl));
  }

  // 4. probe cache; compile via the R miss callback on a miss. The callback
  // returns the compiled artifacts (mirrors anvl's compile result): exec,
  // optional const_arrays, phantom_specs (list(dtype, shape)), client + device
  // xptrs (to allocate phantoms), and opaque out_tree / ambiguous_out.
  CacheEntry* entry = d->cache().get(key);
  if (entry == nullptr) {
    Rcpp::Function miss(d->miss_fn());
    Rcpp::List res = miss(args);

    // Extract everything that can throw FIRST (while `res` keeps the SEXPs
    // rooted), and only then R_PreserveObject into the entry. This keeps the
    // preserve/release balanced even if a malformed callback result throws --
    // a half-built entry with dangling preserves would otherwise leak.
    CacheEntry e;
    auto named = [&](const char* nm) -> SEXP {
      return res.containsElementNamed(nm) ? static_cast<SEXP>(res[nm])
                                          : R_NilValue;
    };
    SEXP exec = named("exec");
    SEXP r_fun = named("r_fun");
    if (closure) {
      if (r_fun == R_NilValue) {
        Rcpp::stop("compile callback must return `r_fun` (closure engine)");
      }
    } else if (exec == R_NilValue) {
      Rcpp::stop("compile callback must return `exec`");
    }
    SEXP consts = named("const_arrays");
    SEXP client = named("client");
    SEXP device_xptr = named("device");
    SEXP out_tree = named("out_tree");
    SEXP ambiguous_out = named("ambiguous_out");
    if (res.containsElementNamed("phantom_specs")) {
      Rcpp::List specs = res["phantom_specs"];
      e.phantom_specs.reserve(specs.size());
      for (R_xlen_t i = 0; i < specs.size(); ++i) {
        Rcpp::List spec = specs[i];
        PhantomSpec ps;
        ps.dtype = Rcpp::as<std::string>(spec["dtype"]);
        // Normalize the boolean aliases the R layer also accepts (a tengen
        // BooleanType stringifies as "bool"; pjrt's canonical name is "pred").
        if (ps.dtype == "bool" || ps.dtype == "i1") ps.dtype = "pred";
        ps.shape = Rcpp::as<std::vector<int64_t>>(spec["shape"]);
        e.phantom_specs.push_back(std::move(ps));
      }
    }

    // No throwing operations past this point: preserve the R objects.
    auto keep = [](SEXP v, SEXP& slot) {
      if (v != R_NilValue) {
        R_PreserveObject(v);
        slot = v;
      }
    };
    keep(exec, e.exec);
    keep(r_fun, e.r_fun);
    if (consts != R_NilValue) {
      Rcpp::List cl(consts);
      e.const_arrays.reserve(cl.size());
      for (R_xlen_t i = 0; i < cl.size(); ++i) {
        SEXP c = cl[i];
        R_PreserveObject(c);
        e.const_arrays.push_back(c);
      }
    }
    keep(client, e.client);
    keep(device_xptr, e.device_xptr);
    keep(out_tree, e.out_tree);
    keep(ambiguous_out, e.ambiguous_out);
    e.device = key.device;

    // Preserve the value-keyed key-leaf SEXPs (static/opaque values, cached
    // dtype objects) so the inserted key outlives this call; released in
    // release_entry on eviction/teardown.
    for (KeyLeaf& kl : key.leaves) {
      if (kl.value != R_NilValue) {
        R_PreserveObject(kl.value);
        e.static_key_values.push_back(kl.value);
      }
    }

    d->cache().set(key, std::move(e));
    entry = d->cache().get(key);
  }

  // 5a. Closure engine (anvl's quickr backend): call the compiled R closure
  // on the full flat leaf list -- array-backed AnvlArrays contribute their
  // backing R array, everything else (static values included; the closure's
  // wrapper drops them) passes through as-is. The closure returns the final
  // (already wrapped) result.
  if (entry->r_fun != R_NilValue) {
    Rcpp::List flat(leaves.size());
    for (std::size_t k = 0; k < leaves.size(); ++k) flat[k] = exec_sexp[k];
    Rcpp::Function fun(entry->r_fun);
    return Rcpp::List::create(Rcpp::Named("value") = fun(flat));
  }

  // 5b. PJRT engine: assemble inputs as const_arrays ++ dynamic leaves ++
  // freshly-allocated phantom donation buffers, then execute. Static and
  // opaque leaves are excluded (statics are baked into the executable as
  // constants). A buffer leaf passes through -- or, under the move policy,
  // is copied to the entry's device when it lives elsewhere; a bare R
  // literal/array leaf is uploaded to the entry's device (same impls and
  // dtype defaults as pjrt_scalar()/pjrt_buffer()).
  // Build the GC-rooted `inputs` list first, then write each allocated
  // buffer (copy, upload, phantom) straight into its slot: it is reachable
  // only through `inputs` (the R GC does not scan C++ locals across the
  // next allocation).
  if ((needs_upload || move) &&
      (entry->client == R_NilValue || entry->device_xptr == R_NilValue)) {
    Rcpp::stop(
        "compile callback must return `client` and `device` for calls with "
        "R-data inputs or a target device");
  }
  std::size_t n_dyn = 0;
  for (const KeyLeaf& kl : key.leaves) {
    if (kl.kind == KeyLeaf::kBuffer || kl.kind == KeyLeaf::kRData) ++n_dyn;
  }
  Rcpp::List inputs(entry->const_arrays.size() + n_dyn +
                    entry->phantom_specs.size());
  R_xlen_t pos = 0;
  for (SEXP c : entry->const_arrays) inputs[pos++] = c;
  for (std::size_t k = 0; k < key.leaves.size(); ++k) {
    const KeyLeaf& kl = key.leaves[k];
    if (kl.kind == KeyLeaf::kBuffer) {
      SEXP b = exec_sexp[k];
      if (move) {
        Rcpp::XPtr<PJRTBuffer> buf(b);
        Rcpp::XPtr<PJRTDevice> dev(entry->device_xptr);
        if (buf->device_ptr() != dev->device) {
          Rcpp::XPtr<PJRTClient> client(entry->client);
          // Same plugin <=> same client (clients are per-platform
          // singletons), so a differing API pointer means a cross-client
          // host-roundtrip copy -- mirrors pjrt::copy_buffer().
          const bool cross = buf->get_api().get() != client->api.get();
          inputs[pos++] = impl_buffer_copy_to_device(buf, dev, client, cross);
          continue;
        }
      }
      inputs[pos++] = b;
    } else if (kl.kind == KeyLeaf::kRData) {
      Rcpp::XPtr<PJRTClient> client(entry->client);
      Rcpp::XPtr<PJRTDevice> dev(entry->device_xptr);
      SEXP leaf = exec_sexp[k];
      switch (TYPEOF(leaf)) {
        case REALSXP:
          inputs[pos++] = impl_client_buffer_from_double(client, dev, leaf,
                                                         kl.av.shape, "f32");
          break;
        case INTSXP:
          inputs[pos++] = impl_client_buffer_from_integer(client, dev, leaf,
                                                          kl.av.shape, "i32");
          break;
        default:
          inputs[pos++] = impl_client_buffer_from_logical(client, dev, leaf,
                                                          kl.av.shape, "pred");
          break;
      }
    }
  }
  if (!entry->phantom_specs.empty()) {
    Rcpp::XPtr<rpjrt::PJRTClient> client(entry->client);
    Rcpp::XPtr<rpjrt::PJRTDevice> device(entry->device_xptr);
    for (const PhantomSpec& ps : entry->phantom_specs) {
      inputs[pos++] =
          impl_client_buffer_empty(client, device, ps.shape, ps.dtype);
    }
  }

  Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> exec(entry->exec);
  Rcpp::XPtr<rpjrt::PJRTExecuteOptions> opts(d->opts());
  Rcpp::List out_bufs = impl_loaded_executable_execute(exec, inputs, opts);

  // Read each output's dtype/shape natively so the caller can build its array
  // wrappers without per-output S3 dtype()/shape()/device() round-trips. The
  // metadata is fixed for a given executable, so it is read off the first
  // call's output buffers and cached on the entry. Shapes are IntegerVectors
  // to match impl_buffer_dimensions (the caller may compare/cache them
  // interchangeably).
  if (entry->out_dtypes == R_NilValue) {
    const R_xlen_t n_out = out_bufs.size();
    Rcpp::CharacterVector out_dtypes(n_out);
    Rcpp::List out_shapes(n_out);
    for (R_xlen_t i = 0; i < n_out; ++i) {
      Rcpp::XPtr<PJRTBuffer> ob(static_cast<SEXP>(out_bufs[i]));
      out_dtypes[i] = PJRTElementType(ob->element_type()).as_string();
      const std::vector<int64_t>& dims = ob->dimensions();
      Rcpp::IntegerVector shp(dims.size());
      for (std::size_t k = 0; k < dims.size(); ++k) {
        shp[k] = static_cast<int>(dims[k]);
      }
      out_shapes[i] = shp;
    }
    R_PreserveObject(out_dtypes);
    entry->out_dtypes = out_dtypes;
    R_PreserveObject(out_shapes);
    entry->out_shapes = out_shapes;
  }

  // Return the raw output buffers plus the wrap material; the caller (anvl)
  // turns these into its array type via out_tree / ambiguous_out / the
  // per-output metadata. `device` is the compile callback's device object
  // (shared by all outputs of one executable).
  return Rcpp::List::create(Rcpp::Named("buffers") = out_bufs,
                            Rcpp::Named("out_tree") = entry->out_tree,
                            Rcpp::Named("ambiguous_out") = entry->ambiguous_out,
                            Rcpp::Named("out_dtypes") = entry->out_dtypes,
                            Rcpp::Named("out_shapes") = entry->out_shapes,
                            Rcpp::Named("device") = entry->device_xptr);
}

// The dispatcher core: the backend-agnostic half of the native eager-dispatch
// hot path. It flattens a call's arguments, validates and classifies the
// leaves, builds the cache key, and drives the LRU cache + compile protocol.
// Everything backend-specific -- reading an aval off an array, what a cache
// entry holds, executing a call, wrapping its outputs -- lives behind the
// Engine interface (dispatch_engine.h); the key material lives in
// dispatch_key.h.
//
// The RTree lives in tree.h (shared with the exposed Rtree API in tree.cpp);
// this file uses it on the stack to flatten a call's arguments to a leaf list
// and to use the structure as cache-key material.

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dispatch_engine.h"
#include "dispatch_key.h"
#include "lru_cache.h"
#include "tree.h"

namespace rpjrt {

// Per-jit Dispatcher: the engine, the executable cache, the R compile
// callback (invoked on a miss), and the dispatch policies. R objects held in
// the cache are R_PreserveObject'd on insert and released on eviction /
// teardown via the cache's on_evict hook + clear().
class Dispatcher {
 public:
  Dispatcher(std::size_t capacity, SEXP miss_fn,
             std::unordered_set<std::string> static_names,
             std::unique_ptr<Engine> engine, std::string backend,
             bool move_inputs, SEXP default_device_fn)
      : cache_(capacity, release_entry),
        miss_fn_(miss_fn),
        default_device_fn_(default_device_fn),
        static_names_(std::move(static_names)),
        engine_(std::move(engine)),
        backend_(std::move(backend)),
        move_inputs_(move_inputs) {
    R_PreserveObject(miss_fn_);
    R_PreserveObject(default_device_fn_);
  }

  ~Dispatcher() {
    cache_.clear();
    R_ReleaseObject(miss_fn_);
    R_ReleaseObject(default_device_fn_);
  }

  // The backend's current default device, as an R object -- the device a call
  // with no array leaves runs on. Resolved afresh per such call: the default
  // can change mid-session, and an entry compiled under one must not serve
  // another. R_NilValue when the dispatcher was given no resolver.
  SEXP default_device_fn() const { return default_device_fn_; }

  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq>& cache() {
    return cache_;
  }
  SEXP miss_fn() const { return miss_fn_; }
  const std::unordered_set<std::string>& static_names() const {
    return static_names_;
  }
  // Non-const: canonical_device() may grow the engine's device table.
  Engine& engine() { return *engine_; }
  // The `$backend` tag every AnvlArray input must carry.
  const std::string& backend() const { return backend_; }
  // Pin policy: a target device is fixed per entry (jit(device = ) /
  // device_arg), so the key carries no device and the engine places the
  // inputs on the entry's device at execute time. The engine holds this too
  // -- the core needs it for the key, the engine for the placing.
  bool move_inputs() const { return move_inputs_; }

 private:
  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq> cache_;
  SEXP miss_fn_;
  SEXP default_device_fn_;
  std::unordered_set<std::string> static_names_;
  std::unique_ptr<Engine> engine_;
  std::string backend_;
  bool move_inputs_;
};

// The per-leaf static mask for a flattened argument list. Static-ness is a
// top-level property: an argument named in `statics` marks every leaf in its
// subtree. `tree` is the one flatten_rec builds for the whole args list, so its
// root's children are the call's arguments. Walk those children and append each
// one's flag once per leaf it contributed -- a leaf is a LeafNode, appended by
// flatten_rec in this same preorder, so the mask lines up with the leaf list
// one-to-one. (A nested list or a NULL is a node but not a leaf, which is why
// leaves are counted rather than nodes.)
inline std::vector<char> static_leaf_mask(
    const RTree& tree, const std::unordered_set<std::string>& statics) {
  std::vector<char> mask;
  const std::int32_t n_args = tree.n_children[0];
  const bool named = tree.is_named(0);
  std::size_t node = 1;  // node 0 is the root; its children start here
  for (std::int32_t k = 0; k < n_args; ++k) {
    const std::int32_t span = tree.subtree_nodes[node];
    const char flag =
        named && statics.count(tree.names[tree.name_off[0] + k]) > 0 ? 1 : 0;
    for (std::int32_t j = 0; j < span; ++j) {
      if (tree.kind[node + j] == RTree::LeafNode) mask.push_back(flag);
    }
    node += span;
  }
  return mask;
}

}  // namespace rpjrt

// ---- Dispatcher: the native eager-dispatch hot path ------------------------

// Create a Dispatcher for one jitted function. `miss_fn(info)` is the R
// callback that compiles on a cache miss and returns the engine's entry
// material (see ?dispatcher). `capacity` is the executable-cache size.
// `static_names` are the top-level argument names whose values are static
// (part of the cache key, excluded from execution). `backend` is the
// `$backend` tag the call's AnvlArray inputs must carry.
// [[Rcpp::export]]
SEXP impl_dispatch_create(int capacity, SEXP miss_fn, SEXP static_names,
                          std::string engine, std::string backend,
                          bool move_inputs, SEXP default_device_fn) {
  using namespace rpjrt;
  // A zero-capacity LRU evicts every entry as it is inserted, so the compile
  // path would insert and then dereference a null entry.
  if (capacity < 1) Rcpp::stop("capacity must be at least 1");
  if (TYPEOF(miss_fn) != CLOSXP) {
    Rcpp::stop("miss_fn must be a function");
  }
  if (backend.empty()) Rcpp::stop("backend must be a non-empty string");
  std::unique_ptr<Engine> eng = make_engine(engine, backend, move_inputs);
  if (default_device_fn != R_NilValue && TYPEOF(default_device_fn) != CLOSXP) {
    Rcpp::stop("default_device must be a function or NULL");
  }
  std::unordered_set<std::string> statics;
  if (static_names != R_NilValue && TYPEOF(static_names) == STRSXP) {
    const R_xlen_t n = XLENGTH(static_names);
    for (R_xlen_t k = 0; k < n; ++k) {
      statics.insert(std::string(CHAR(STRING_ELT(static_names, k))));
    }
  }
  auto* d = new Dispatcher(static_cast<std::size_t>(capacity), miss_fn,
                           std::move(statics), std::move(eng),
                           std::move(backend), move_inputs, default_device_fn);
  Rcpp::XPtr<Dispatcher> ptr(d, true);
  ptr.attr("class") = "Dispatcher";
  return ptr;
}

// Number of compiled executables currently cached by the Dispatcher.
// [[Rcpp::export]]
int impl_dispatcher_size(SEXP dispatcher) {
  Rcpp::XPtr<rpjrt::Dispatcher> d(dispatcher);
  return static_cast<int>(d->cache().size());
}

// Run the native dispatch for `args` (the evaluated argument list of the
// call) and return the call's finished result. Every input is validated here,
// by name; a call that returns has been dispatched.
// [[Rcpp::export]]
SEXP impl_dispatch_run(SEXP dispatcher, Rcpp::List args) {
  using namespace rpjrt;
  Rcpp::XPtr<Dispatcher> d(dispatcher);
  const std::unordered_set<std::string>& statics = d->static_names();
  Engine& engine = d->engine();

  // 1. Flatten args into leaves + structure, and mark the static leaves.
  // flatten_rec encodes the argument list as the root ListNode, so its children
  // are the call's arguments; static_leaf_mask overlays the dispatch-only
  // static bit on the leaves each static-named argument contributed (this
  // overlay is not part of the shared Rtree API).
  std::vector<SEXP> leaves;
  RTree in_tree;
  flatten_rec(args, leaves, in_tree);
  const std::vector<char> is_static = static_leaf_mask(in_tree, statics);

  // 2. Validate and classify leaves into key leaves + per-leaf execute
  // material. Every rejection happens here, named after the offending argument:
  // an input that survives this loop is one the engine can execute, so the
  // compile callback is only ever asked to compile, never to validate.
  const bool move = d->move_inputs();
  const char* call_backend = d->backend().c_str();
  CacheKey key;
  key.in_tree = in_tree;
  // Reserved to the exact leaf count and never grown past it, which is what
  // keeps the `&key.leaves.back().av` pointers in exec_inputs valid: no
  // reallocation can happen while the two are built side by side.
  key.leaves.reserve(leaves.size());
  // The call's execute-time inputs, in program order. A static leaf
  // contributes a key leaf but no input -- it is a constant in the compiled
  // program -- so the engine is handed only what it must actually supply.
  std::vector<ExecInput> exec_inputs;
  exec_inputs.reserve(leaves.size());
  bool have_device = false;
  for (std::size_t k = 0; k < leaves.size(); ++k) {
    SEXP leaf = leaves[k];
    KeyLeaf kl;
    if (is_static[k]) {
      // A static is baked into the executable as a constant and compared with
      // identical() on every call. An array is neither: it would key the cache
      // on its contents, and the compile callback would trace it as a real
      // input that execution then never supplies.
      if (Rf_inherits(leaf, "AnvlArray")) {
        Rcpp::stop(
            "invalid static %s: a static argument must not be an AnvlArray",
            leaf_subject(in_tree, k));
      }
      kl.kind = KeyLeaf::kStatic;
      kl.value = leaf;
      key.leaves.push_back(std::move(kl));
      continue;
    }
    if (std::optional<AnvlFields> af = anvl_fields(leaf)) {
      const char* leaf_backend =
          af->backend != nullptr ? af->backend : "<none>";
      // "plain" AnvlArrays capture trace-time constants in a backend-agnostic
      // way; no engine can execute one, whichever engine is asking.
      if (!std::strcmp(leaf_backend, "plain")) {
        Rcpp::stop(
            "invalid %s: an AnvlArray of the \"plain\" backend captures a "
            "trace-time constant and is not a call argument",
            leaf_subject(in_tree, k));
      }
      // Every array leaf must carry the dispatcher's backend. Mixing backends
      // in one call can therefore never happen; it is a per-dispatcher
      // property, not a per-leaf one.
      if (std::strcmp(leaf_backend, call_backend)) {
        Rcpp::stop(
            "invalid %s: expected an AnvlArray of backend \"%s\"; got \"%s\"",
            leaf_subject(in_tree, k), call_backend, leaf_backend);
      }
      kl.kind = KeyLeaf::kArray;
      kl.av = engine.array_aval(*af, in_tree, k);
      if (af->device == R_NilValue) {
        Rcpp::stop("invalid %s: an AnvlArray must carry $device",
                   leaf_subject(in_tree, k));
      }
      // Under the pin policy the entry's device is fixed and the engine
      // places the inputs on it, so no leaf's device reaches the key and
      // inputs may be spread across devices. Otherwise the leaf's `$device`
      // is canonicalized by the engine -- equal-but-distinct device objects
      // collapse to one token, whose object the engine keeps alive -- and the
      // infer policy applies: the first array's device is the call's device,
      // and every later array must agree with it.
      if (!move) {
        const DeviceToken leaf_device =
            static_cast<DeviceToken>(engine.canonical_device(af->device));
        if (!have_device) {
          key.device = leaf_device;
          have_device = true;
        } else if (leaf_device != key.device) {
          Rcpp::stop(
              "invalid %s: it lives on a different device than an earlier "
              "input; all inputs must share a device unless a target device "
              "is fixed",
              leaf_subject(in_tree, k));
        }
      }
      key.leaves.push_back(std::move(kl));
      exec_inputs.push_back({af->data, &key.leaves.back().av, false});
      continue;
    }
    std::optional<RDataInfo> rd = classify_rdata(leaf);
    if (!rd) {
      Rcpp::stop(
          "invalid %s: expected an AnvlArray, a length-1 atomic scalar, or an "
          "is.array() value; got <%s> of length %d",
          leaf_subject(in_tree, k), r_class_name(leaf),
          static_cast<long long>(Rf_xlength(leaf)));
    }
    kl.kind = KeyLeaf::kRData;
    kl.av.dtype = rd->dtype;
    kl.av.shape = std::move(rd->shape);
    kl.av.ambiguous = true;  // bare R data is dtype-ambiguous (to_avals)
    key.leaves.push_back(std::move(kl));
    exec_inputs.push_back({leaf, &key.leaves.back().av, true});
  }

  // 3. No array leaf named a device, so the call runs on the backend's
  // *current* default. That device still binds the entry the callback will
  // compile, so the key must name it: without this, `f(1)` compiled under one
  // default device would be served back after the default changed. Resolved per
  // call, since the default can change mid-session, and never under the move
  // policy, where the entry's device is fixed and the call's own must stay out
  // of the key.
  Rcpp::RObject default_device;  // the resolved object, for the callback
  if (!move && !have_device) {
    if (d->default_device_fn() == R_NilValue) {
      Rcpp::stop(
          "this dispatcher cannot dispatch a call with no array inputs: it was "
          "created without a `default_device` resolver");
    }
    Rcpp::Function resolve(d->default_device_fn());
    // Canonicalized like a leaf's device, so a resolver that returns a fresh
    // (equal) object per call still lands on the entry it compiled.
    default_device = engine.canonical_device(resolve());
    key.device = static_cast<DeviceToken>(SEXP(default_device));
  }

  // 4. Probe the cache; compile via the R miss callback on a miss.
  CacheEntry* entry = d->cache().get(key);
  if (entry == nullptr) {
    // Hand the callback the material this call already derived -- the tree, the
    // flat leaves, the static mask, and each dynamic leaf's aval -- rather than
    // the bare `args` for it to classify a second time. The avals here are the
    // ones the cache key was built from, so the program the callback compiles
    // cannot disagree with the key it gets filed under.
    const R_xlen_t n_leaves = static_cast<R_xlen_t>(leaves.size());
    Rcpp::List leaf_list(n_leaves);
    Rcpp::LogicalVector static_mask(n_leaves);
    Rcpp::List avals(n_leaves);  // NULL at a static leaf: it has no aval
    for (R_xlen_t i = 0; i < n_leaves; ++i) {
      leaf_list[i] = leaves[i];
      static_mask[i] = is_static[i] ? TRUE : FALSE;
      const KeyLeaf& kl = key.leaves[i];
      if (kl.kind == KeyLeaf::kStatic) continue;
      Rcpp::IntegerVector shp(kl.av.shape.size());
      for (std::size_t j = 0; j < kl.av.shape.size(); ++j) {
        shp[j] = static_cast<int>(kl.av.shape[j]);
      }
      // Every dtype has a canonical name -- a leaf with none was rejected -- so
      // the callback always sees a string, whichever backend the leaf is from.
      avals[i] = Rcpp::List::create(
          Rcpp::Named("dtype") = anvl_dtype_name(kl.av.dtype),
          Rcpp::Named("shape") = shp,
          Rcpp::Named("ambiguous") = kl.av.ambiguous);
    }
    Rcpp::List info = Rcpp::List::create(
        Rcpp::Named("args") = args,
        Rcpp::Named("in_tree") = tree_xptr(new RTree(in_tree)),
        Rcpp::Named("leaves") = leaf_list,
        Rcpp::Named("is_static") = static_mask, Rcpp::Named("avals") = avals,
        // The device this call resolved when no array named one -- the device
        // the key was built on, so the callback compiles for it rather than
        // resolving a default of its own. NULL when the leaves named a device,
        // or under the move policy.
        Rcpp::Named("default_device") = default_device);

    Rcpp::Function miss(d->miss_fn());
    Rcpp::List res = miss(info);

    // The engine validates the result and builds its entry material; entry
    // SEXPs are preserved only after everything that can throw has thrown.
    CacheEntry e;
    engine.build_entry(res, e);

    // Preserve every SEXP the inserted key holds, so it outlives this call;
    // released in release_entry on eviction/teardown. The key's device token
    // needs no preserving here: it is the address of a canonical device
    // object the engine keeps alive for the dispatcher's lifetime.
    for (KeyLeaf& kl : key.leaves) e.preserve(kl.value);

    d->cache().set(key, std::move(e));
    entry = d->cache().get(key);
  }

  // 5. The engine runs the call and returns the finished value.
  return engine.run(*entry, exec_inputs);
}

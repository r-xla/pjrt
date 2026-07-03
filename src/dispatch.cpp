// Native eager-dispatch hot path (see anvl/benchmarks/cpp-hot-path-design.md).
//
// The pytree `Node` tree lives in tree.h (shared with the exposed tree API in
// tree.cpp); this file uses it on the stack to flatten a call's arguments to a
// leaf list and to use the structure as cache-key material.

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "buffer.h"
#include "client.h"
#include "device.h"
#include "tree.h"

namespace rpjrt {

// Per-leaf abstract value -- mirrors anvl's nv_aval(dtype, shape, ambiguous).
// dtype/shape are genuine PJRT_Buffer properties read natively; `ambiguous` is
// an anvl type-system bit supplied per leaf (pjrt folds it into the key but
// never interprets it). `device` is NOT per-aval: it is a single per-call value
// on the cache_key (matching anvl).
struct aval {
  int dtype = 0;  // PJRT_Buffer_Type enum
  std::vector<int64_t> shape;
  bool ambiguous = false;
};

// One leaf of the cache key. Dynamic leaves carry an `aval`; static-arg leaves
// carry the R value itself (compared via R's identical()).
struct KeyLeaf {
  bool is_static = false;
  aval av;                  // when !is_static
  SEXP value = R_NilValue;  // when is_static (protected by the cache entry)
};

// The executable-cache key -- mirrors anvl's list(in_tree, key_leaves, device).
struct CacheKey {
  Node in_tree;
  std::vector<KeyLeaf> leaves;
  const void* device = nullptr;  // canonical device id (PJRT_Device*), or null
};

static std::size_t aval_hash(const aval& a) {
  std::size_t h = static_cast<std::size_t>(a.dtype);
  hash_combine(h, a.ambiguous ? 1u : 0u);
  hash_combine(h, a.shape.size());
  for (int64_t d : a.shape) {
    hash_combine(h, static_cast<std::size_t>(d));
  }
  return h;
}

static bool aval_eq(const aval& a, const aval& b) {
  return a.dtype == b.dtype && a.ambiguous == b.ambiguous && a.shape == b.shape;
}

struct CacheKeyHash {
  std::size_t operator()(const CacheKey& k) const {
    std::size_t h = node_hash(k.in_tree);
    hash_combine(h, reinterpret_cast<std::size_t>(k.device));
    hash_combine(h, k.leaves.size());
    for (const KeyLeaf& leaf : k.leaves) {
      if (leaf.is_static) {
        // Cheap discriminator only; exact equality falls back to identical().
        hash_combine(h, 0x57A71Cu);
        hash_combine(h, static_cast<std::size_t>(TYPEOF(leaf.value)));
        hash_combine(h, static_cast<std::size_t>(Rf_xlength(leaf.value)));
      } else {
        hash_combine(h, aval_hash(leaf.av));
      }
    }
    return h;
  }
};

struct CacheKeyEq {
  bool operator()(const CacheKey& a, const CacheKey& b) const {
    if (!node_eq(a.in_tree, b.in_tree)) return false;
    if (a.device != b.device) return false;
    if (a.leaves.size() != b.leaves.size()) return false;
    for (std::size_t k = 0; k < a.leaves.size(); ++k) {
      const KeyLeaf& x = a.leaves[k];
      const KeyLeaf& y = b.leaves[k];
      if (x.is_static != y.is_static) return false;
      if (x.is_static) {
        // Match anvl, which compares static args with R's default identical().
        // R_compute_identical's flag bits are USE bits (identical.c): default
        // identical() sets only IDENT_USE_CLOENV (16), i.e. compare closure
        // environments but ignore bytecode/srcref. flags=0 would ignore
        // environments and wrongly merge distinct closures.
        if (!R_compute_identical(x.value, y.value, /*flags=*/16)) return false;
      } else if (!aval_eq(x.av, y.av)) {
        return false;
      }
    }
    return true;
  }
};

// A least-recently-used cache: a hashmap for lookup plus a doubly-linked list
// ordering entries most- to least-recently-used (mirrors xlamisc::LRUCache, the
// R cache anvl's jit uses). `get`/`set` move the touched entry to the front;
// `set` evicts from the back when over capacity. `on_evict` runs on the evicted
// value before it is dropped -- used to release R objects held in the value.
template <typename K, typename V, typename Hash, typename Eq>
class LRUCache {
 public:
  explicit LRUCache(std::size_t capacity,
                    std::function<void(V&)> on_evict = nullptr)
      : capacity_(capacity), on_evict_(std::move(on_evict)) {}

  // Returns a pointer to the value (now MRU), or nullptr on miss.
  V* get(const K& key) {
    auto it = index_.find(key);
    if (it == index_.end()) return nullptr;
    order_.splice(order_.begin(), order_, it->second);
    return &it->second->value;
  }

  void set(const K& key, V value) {
    auto it = index_.find(key);
    if (it != index_.end()) {
      if (on_evict_) on_evict_(it->second->value);
      it->second->value = std::move(value);
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    order_.push_front(Entry{key, std::move(value)});
    index_.emplace(key, order_.begin());
    if (index_.size() > capacity_) {
      Entry& victim = order_.back();
      if (on_evict_) on_evict_(victim.value);
      index_.erase(victim.key);
      order_.pop_back();
    }
  }

  std::size_t size() const { return index_.size(); }

  // Run on_evict over every entry and drop them (used on dispatcher teardown so
  // cached R objects are released, not leaked).
  void clear() {
    if (on_evict_) {
      for (Entry& e : order_) on_evict_(e.value);
    }
    order_.clear();
    index_.clear();
  }

 private:
  struct Entry {
    K key;
    V value;
  };
  std::list<Entry> order_;  // front = MRU, back = LRU
  std::unordered_map<K, typename std::list<Entry>::iterator, Hash, Eq> index_;
  std::size_t capacity_;
  std::function<void(V&)> on_evict_;
};

// Read (dtype, shape) off a PJRTBuffer xptr and (device) as its canonical id.
// Native reads via pjrt's own buffer class -- no R round-trip.
static aval aval_from_buffer(SEXP buf_xptr, bool ambiguous,
                             const void** out_device) {
  Rcpp::XPtr<PJRTBuffer> buf(buf_xptr);
  aval a;
  a.dtype = static_cast<int>(buf->element_type());
  a.shape = buf->dimensions();
  a.ambiguous = ambiguous;
  if (out_device) {
    std::unique_ptr<PJRTDevice> dev = buf->device();
    *out_device = static_cast<const void*>(dev->device);
  }
  return a;
}

// Extract the input buffer + ambiguity from a leaf. A leaf is dispatchable iff
// it is a bare PJRTBuffer xptr (ambiguous = false) or an xla AnvlArray -- an R
// list with class "AnvlArray" whose `$backend` is "xla" -- from which we read
// `$data` (the buffer) and `$ambiguous`. Anything else is not dispatchable.
struct LeafBuf {
  bool ok = false;
  SEXP buffer = R_NilValue;
  bool ambiguous = false;
};

static LeafBuf extract_leaf(SEXP leaf) {
  LeafBuf lb;
  // Only an xla AnvlArray is dispatchable: a named list of class "AnvlArray"
  // with $backend == "xla", from which we read $data (a PJRTBuffer) and
  // $ambiguous. Everything else -- a bare buffer (the anvl compile callback
  // cannot turn it into an aval), a non-xla backend, a literal, etc. -- is not
  // dispatchable and falls back to the R path.
  if (TYPEOF(leaf) != VECSXP || !Rf_inherits(leaf, "AnvlArray")) return lb;
  SEXP nms = Rf_getAttrib(leaf, R_NamesSymbol);
  if (nms == R_NilValue) return lb;
  SEXP data = R_NilValue, backend = R_NilValue, amb = R_NilValue;
  for (R_xlen_t k = 0; k < XLENGTH(leaf); ++k) {
    const char* nm = CHAR(STRING_ELT(nms, k));
    if (!std::strcmp(nm, "data"))
      data = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "backend"))
      backend = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "ambiguous"))
      amb = VECTOR_ELT(leaf, k);
  }
  if (backend == R_NilValue || TYPEOF(backend) != STRSXP ||
      std::strcmp(CHAR(STRING_ELT(backend, 0)), "xla") != 0) {
    return lb;  // non-xla backend -> not dispatchable here
  }
  if (data == R_NilValue || TYPEOF(data) != EXTPTRSXP ||
      !Rf_inherits(data, "PJRTBuffer")) {
    return lb;
  }
  lb.ok = true;
  lb.buffer = data;
  lb.ambiguous = (amb != R_NilValue && Rf_asLogical(amb) == TRUE);
  return lb;
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
  SEXP exec = R_NilValue;               // PJRTLoadedExecutable xptr (preserved)
  std::vector<SEXP> const_arrays;       // (preserved)
  std::vector<SEXP> static_key_values;  // static key-leaf SEXPs (preserved)
  std::vector<PhantomSpec> phantom_specs;
  SEXP client = R_NilValue;         // PJRTClient xptr for phantom alloc
  SEXP device_xptr = R_NilValue;    // PJRTDevice xptr for phantom alloc
  SEXP out_tree = R_NilValue;       // opaque, for the caller's wrap
  SEXP ambiguous_out = R_NilValue;  // opaque, for the caller's wrap
  const void* device = nullptr;
};

// Release every R object a cache entry preserves (on eviction / teardown).
static void release_entry(CacheEntry& e) {
  if (e.exec != R_NilValue) R_ReleaseObject(e.exec);
  for (SEXP c : e.const_arrays) R_ReleaseObject(c);
  for (SEXP c : e.static_key_values) R_ReleaseObject(c);
  if (e.client != R_NilValue) R_ReleaseObject(e.client);
  if (e.device_xptr != R_NilValue) R_ReleaseObject(e.device_xptr);
  if (e.out_tree != R_NilValue) R_ReleaseObject(e.out_tree);
  if (e.ambiguous_out != R_NilValue) R_ReleaseObject(e.ambiguous_out);
}

// Per-jit dispatcher: owns the executable cache, the R cache_miss callback
// (compiles on a miss), and a reusable default-options object. R objects held
// in the cache are R_PreserveObject'd on insert and released on eviction /
// teardown via the cache's on_evict hook + clear().
class Dispatcher {
 public:
  Dispatcher(std::size_t capacity, SEXP miss_fn, SEXP opts,
             std::unordered_set<std::string> static_names)
      : cache_(capacity, release_entry),
        miss_fn_(miss_fn),
        opts_(opts),
        static_names_(std::move(static_names)) {
    R_PreserveObject(miss_fn_);
    R_PreserveObject(opts_);
  }

  ~Dispatcher() {
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

 private:
  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq> cache_;
  SEXP miss_fn_;
  SEXP opts_;
  std::unordered_set<std::string> static_names_;
};

}  // namespace rpjrt

// These are defined in pjrt.cpp; the dispatcher reuses them directly so the
// keepalive/donation/options logic lives in one place.
Rcpp::List impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input,
    Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options);
Rcpp::XPtr<rpjrt::PJRTExecuteOptions> impl_execution_options_create(
    std::vector<int64_t> non_donatable_input_indices, int launch_id);
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_empty(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    std::vector<int64_t> dims, std::string dtype);

// Self-test entry: flatten `x`, rebuild it, and hash the tree. Lets the R tests
// check the native Node against R's flatten()/unflatten() on real inputs.
// [[Rcpp::export]]
Rcpp::List impl_dispatch_node_selftest(SEXP x) {
  using namespace rpjrt;
  std::vector<SEXP> leaves;
  std::vector<char> is_static;
  Node tree;
  int counter = 0;
  flatten_rec(x, leaves, is_static, tree, counter, /*inherited_static=*/false);

  SEXP leaves_list = PROTECT(Rf_allocVector(VECSXP, leaves.size()));
  for (std::size_t k = 0; k < leaves.size(); ++k) {
    SET_VECTOR_ELT(leaves_list, k, leaves[k]);
  }
  SEXP rebuilt = PROTECT(unflatten_rec(tree, leaves));

  Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("n_leaves") = static_cast<int>(leaves.size()),
      Rcpp::Named("leaves") = leaves_list, Rcpp::Named("rebuilt") = rebuilt,
      Rcpp::Named("hash") = static_cast<double>(node_hash(tree)));
  UNPROTECT(2);
  return out;
}

namespace rpjrt {

// Build a CacheKey from a flat list of dynamic leaves, each a
// list(buffer_xptr, ambiguous), with a flat ListNode in_tree over them.
static CacheKey build_key_from_leaves(Rcpp::List leaves) {
  CacheKey key;
  key.in_tree.kind = Node::ListNode;
  key.in_tree.nodes.resize(leaves.size());
  key.leaves.reserve(leaves.size());
  for (R_xlen_t k = 0; k < leaves.size(); ++k) {
    Rcpp::List leaf = leaves[k];
    SEXP buf = leaf[0];
    bool ambiguous = Rcpp::as<bool>(leaf[1]);
    const void* dev = nullptr;
    KeyLeaf kl;
    kl.is_static = false;
    kl.av = aval_from_buffer(buf, ambiguous, &dev);
    key.leaves.push_back(std::move(kl));
    // device is a single per-call value; take it from the first leaf.
    if (k == 0) key.device = dev;
    Node child;
    child.kind = Node::LeafNode;
    child.i = static_cast<int>(k + 1);
    key.in_tree.nodes[k] = child;
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
    key.in_tree.kind = rpjrt::Node::ListNode;
    key.in_tree.nodes.resize(vals.size());
    key.leaves.reserve(vals.size());
    for (R_xlen_t k = 0; k < vals.size(); ++k) {
      rpjrt::KeyLeaf kl;
      kl.is_static = true;
      SEXP v = vals[k];
      kl.value = v;
      key.leaves.push_back(std::move(kl));
      rpjrt::Node child;
      child.kind = rpjrt::Node::LeafNode;
      child.i = static_cast<int>(k + 1);
      key.in_tree.nodes[k] = child;
    }
    return key;
  };
  rpjrt::CacheKey ka = build(a);
  rpjrt::CacheKey kb = build(b);
  return rpjrt::CacheKeyEq{}(ka, kb);
}

// Self-test for the LRU cache: capacity 2, exercise recency + eviction.
// Returns c(get1, has1, has2, has3, size, n_evicted); expect c(10,1,0,1,2,1).
// [[Rcpp::export]]
Rcpp::IntegerVector impl_dispatch_lru_selftest() {
  int evicted = 0;
  rpjrt::LRUCache<int, int, std::hash<int>, std::equal_to<int>> cache(
      2, [&](int&) { ++evicted; });
  cache.set(1, 10);
  cache.set(2, 20);
  int g1 = *cache.get(1);  // 10; touching 1 makes it MRU, 2 is LRU
  cache.set(3, 30);        // over capacity -> evicts 2
  bool has1 = cache.get(1) != nullptr;
  bool has2 = cache.get(2) != nullptr;  // evicted
  bool has3 = cache.get(3) != nullptr;
  return Rcpp::IntegerVector::create(g1, has1, has2, has3,
                                     static_cast<int>(cache.size()), evicted);
}

// ---- Dispatcher: the native eager-dispatch hot path -------------------------

// The sentinel returned by impl_dispatch_run() when the call is not handled by
// the native fast path (the R caller must fall back to its slow path). A unique
// symbol, identity-comparable on the R side.
// [[Rcpp::export]]
SEXP impl_dispatch_sentinel() {
  return Rf_install("__pjrt_dispatch_sentinel__");
}

// Create a dispatcher for one jitted function. `miss_fn(args)` is the R
// callback that compiles on a cache miss and returns a list with at least
// `exec` (a PJRTLoadedExecutable xptr) and optionally `const_arrays` (a list of
// buffer xptrs prepended to the inputs). `capacity` is the executable-cache
// size. `static_names` are the top-level argument names whose values are
// static (part of the cache key, excluded from execution).
// [[Rcpp::export]]
SEXP impl_dispatch_create(int capacity, SEXP miss_fn, SEXP static_names) {
  using namespace rpjrt;
  if (TYPEOF(miss_fn) != CLOSXP && TYPEOF(miss_fn) != BUILTINSXP &&
      TYPEOF(miss_fn) != SPECIALSXP) {
    Rcpp::stop("miss_fn must be a function");
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
  auto* d = new Dispatcher(static_cast<std::size_t>(capacity), miss_fn,
                           static_cast<SEXP>(opts), std::move(statics));
  Rcpp::XPtr<Dispatcher> ptr(d, true);
  ptr.attr("class") = "PJRTDispatcher";
  return ptr;
}

// Number of compiled executables currently cached by the dispatcher.
// [[Rcpp::export]]
int impl_dispatch_size(SEXP handle) {
  Rcpp::XPtr<rpjrt::Dispatcher> d(handle);
  return static_cast<int>(d->cache().size());
}

// Run the native dispatch for `args` (the evaluated argument list of the call).
// Returns the raw output buffers (a list of PJRTBuffer xptrs) on success, or
// the dispatch sentinel when the call is not handled natively (caller falls
// back).
// [[Rcpp::export]]
SEXP impl_dispatch_run(SEXP handle, Rcpp::List args) {
  using namespace rpjrt;
  Rcpp::XPtr<Dispatcher> d(handle);
  const std::unordered_set<std::string>& statics = d->static_names();

  // 1. Flatten args into leaves + structure, marking static leaves. Static-ness
  // is decided at the top level by argument name and propagated to every leaf
  // beneath a static arg (mirrors anvl's static-arg marking).
  std::vector<SEXP> leaves;
  std::vector<char> is_static;
  Node in_tree;
  int counter = 0;
  in_tree.kind = Node::ListNode;
  const R_xlen_t n_args = args.size();
  in_tree.nodes.resize(n_args);
  SEXP arg_nms = Rf_getAttrib(args, R_NamesSymbol);
  in_tree.has_names = (arg_nms != R_NilValue);
  if (in_tree.has_names) {
    in_tree.names.resize(n_args);
    for (R_xlen_t k = 0; k < n_args; ++k) {
      in_tree.names[k] = std::string(CHAR(STRING_ELT(arg_nms, k)));
    }
  }
  for (R_xlen_t k = 0; k < n_args; ++k) {
    bool child_static = !statics.empty() && in_tree.has_names &&
                        statics.count(in_tree.names[k]) > 0;
    flatten_rec(VECTOR_ELT(args, k), leaves, is_static, in_tree.nodes[k],
                counter, child_static);
  }

  // 2. Classify leaves. Static leaves become value-keyed KeyLeafs; dynamic
  // leaves must be xla AnvlArrays (else bail). Collect dynamic buffers (in
  // flatten order) for execution and read avals + device natively. Every
  // dynamic leaf must live on the same device, else bail so the R path
  // produces its multi-device error.
  CacheKey key;
  key.in_tree = in_tree;
  key.leaves.reserve(leaves.size());
  std::vector<LeafBuf> bufs;  // dynamic buffers only
  bufs.reserve(leaves.size());
  bool have_device = false;
  for (std::size_t k = 0; k < leaves.size(); ++k) {
    KeyLeaf kl;
    if (is_static[k]) {
      kl.is_static = true;
      kl.value = leaves[k];
      key.leaves.push_back(std::move(kl));
      continue;
    }
    LeafBuf lb = extract_leaf(leaves[k]);
    if (!lb.ok) return impl_dispatch_sentinel();
    const void* dev = nullptr;
    kl.is_static = false;
    kl.av = aval_from_buffer(lb.buffer, lb.ambiguous, &dev);
    if (!have_device) {
      key.device = dev;
      have_device = true;
    } else if (dev != key.device) {
      return impl_dispatch_sentinel();
    }
    key.leaves.push_back(std::move(kl));
    bufs.push_back(lb);
  }

  // No dynamic buffers -> no device to run on natively; let R infer from the
  // trace (e.g. all-static or zero-arg constructor calls).
  if (bufs.empty()) return impl_dispatch_sentinel();

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
    SEXP exec = res["exec"];  // required
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

    // Preserve the static key values so the inserted key's SEXPs outlive this
    // call; released in release_entry on eviction/teardown.
    for (KeyLeaf& kl : key.leaves) {
      if (kl.is_static) {
        R_PreserveObject(kl.value);
        e.static_key_values.push_back(kl.value);
      }
    }

    d->cache().set(key, std::move(e));
    entry = d->cache().get(key);
  }

  // 5. assemble inputs: const_arrays ++ dynamic buffers ++ freshly-allocated
  // phantom donation buffers, then execute. Static leaves are excluded (they
  // are baked into the executable as constants).
  // Build the GC-rooted `inputs` list first, then allocate each phantom buffer
  // straight into its slot. A phantom is reachable only through `inputs` (the
  // R GC does not scan a std::vector<SEXP>), and impl_client_buffer_empty
  // allocates -- so writing it into the rooted list immediately, before the
  // next allocation, is required to avoid a GC use-after-free.
  Rcpp::List inputs(entry->const_arrays.size() + bufs.size() +
                    entry->phantom_specs.size());
  R_xlen_t pos = 0;
  for (SEXP c : entry->const_arrays) inputs[pos++] = c;
  for (const LeafBuf& lb : bufs) inputs[pos++] = lb.buffer;
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

  // Return the raw output buffers plus the opaque wrap material; the caller
  // (anvl) turns these into its array type via out_tree / ambiguous_out.
  return Rcpp::List::create(
      Rcpp::Named("buffers") = out_bufs,
      Rcpp::Named("out_tree") = entry->out_tree,
      Rcpp::Named("ambiguous_out") = entry->ambiguous_out);
}

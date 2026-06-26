// Native eager-dispatch hot path (see anvl/benchmarks/cpp-hot-path-design.md).
//
// Slice 1: the `Node` tree -- a native port of anvl's R/flatten.R. It captures
// the pytree structure of a call's arguments so the dispatch can flatten args
// to a leaf list, reconstruct outputs, and use the structure as cache-key
// material. Names and semantics mirror flatten.R exactly:
//   * NULL              -> NullNode (no leaf, no flat index, but kept in tree)
//   * bare list         -> ListNode (recurses; remembers names)
//   * anything else     -> LeafNode (a classed object or atomic is a leaf)
// "bare list" == VECSXP without a class attribute, matching R's is.object().

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "buffer.h"
#include "client.h"
#include "device.h"

namespace rpjrt {

struct Node {
  enum Kind { LeafNode, ListNode, NullNode };
  Kind kind;
  int i = 0;                // LeafNode: 1-based index into the flat leaves
  std::vector<Node> nodes;  // ListNode: children
  bool has_names = false;   // ListNode: whether names() was non-NULL
  std::vector<std::string>
      names;  // ListNode: child names ("" for unnamed slots)
};

// A bare list (recurse) vs a leaf (classed object / atomic / function / ...).
// Rf_isObject(x) == OBJECT(x): true iff x carries a class attribute, matching
// R's is.object() used by flatten()/build_tree() to decide leaf vs list.
static inline bool is_bare_list(SEXP x) {
  return TYPEOF(x) == VECSXP && !Rf_isObject(x);
}

// Flatten `x` into `leaves` (in order) and fill `node` with its structure.
// `counter` assigns 1-based leaf indices in flatten order (matches build_tree).
static void flatten_rec(SEXP x, std::vector<SEXP>& leaves, Node& node,
                        int& counter) {
  if (x == R_NilValue) {
    node.kind = Node::NullNode;
    return;
  }
  if (is_bare_list(x)) {
    node.kind = Node::ListNode;
    const R_xlen_t n = XLENGTH(x);
    SEXP nms = Rf_getAttrib(x, R_NamesSymbol);
    node.has_names = (nms != R_NilValue);
    node.nodes.resize(n);
    if (node.has_names) {
      node.names.resize(n);
      for (R_xlen_t k = 0; k < n; ++k) {
        node.names[k] = std::string(CHAR(STRING_ELT(nms, k)));
      }
    }
    for (R_xlen_t k = 0; k < n; ++k) {
      flatten_rec(VECTOR_ELT(x, k), leaves, node.nodes[k], counter);
    }
    return;
  }
  node.kind = Node::LeafNode;
  node.i = ++counter;
  leaves.push_back(x);
}

// Reconstruct an R object from a tree and a flat leaf list (mirrors unflatten).
static SEXP unflatten_rec(const Node& node, const std::vector<SEXP>& leaves) {
  switch (node.kind) {
    case Node::NullNode:
      return R_NilValue;
    case Node::LeafNode:
      return leaves[node.i - 1];
    case Node::ListNode: {
      const std::size_t n = node.nodes.size();
      SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
      for (std::size_t k = 0; k < n; ++k) {
        SET_VECTOR_ELT(out, k, unflatten_rec(node.nodes[k], leaves));
      }
      if (node.has_names) {
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
        for (std::size_t k = 0; k < n; ++k) {
          SET_STRING_ELT(nms, k, Rf_mkChar(node.names[k].c_str()));
        }
        Rf_setAttrib(out, R_NamesSymbol, nms);
        UNPROTECT(1);
      }
      UNPROTECT(1);
      return out;
    }
  }
  return R_NilValue;  // unreachable
}

// Structural hash / equality. Two trees are equal iff identical kind, child
// structure, and names -- the same notion R's identical() uses on the Node.
static void hash_combine(std::size_t& seed, std::size_t v) {
  seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

static std::size_t node_hash(const Node& node) {
  std::size_t h = static_cast<std::size_t>(node.kind);
  switch (node.kind) {
    case Node::NullNode:
      break;
    case Node::LeafNode:
      hash_combine(h, static_cast<std::size_t>(node.i));
      break;
    case Node::ListNode:
      hash_combine(h, node.nodes.size());
      hash_combine(h, node.has_names ? 1u : 0u);
      for (const auto& nm : node.names) {
        hash_combine(h, std::hash<std::string>{}(nm));
      }
      for (const auto& child : node.nodes) {
        hash_combine(h, node_hash(child));
      }
      break;
  }
  return h;
}

static bool node_eq(const Node& a, const Node& b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case Node::NullNode:
      return true;
    case Node::LeafNode:
      return a.i == b.i;
    case Node::ListNode:
      if (a.nodes.size() != b.nodes.size()) return false;
      if (a.has_names != b.has_names) return false;
      if (a.has_names && a.names != b.names) return false;
      for (std::size_t k = 0; k < a.nodes.size(); ++k) {
        if (!node_eq(a.nodes[k], b.nodes[k])) return false;
      }
      return true;
  }
  return false;
}

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
        if (!R_compute_identical(x.value, y.value, /*flags=*/0)) return false;
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
  if (TYPEOF(leaf) == EXTPTRSXP && Rf_inherits(leaf, "PJRTBuffer")) {
    lb.ok = true;
    lb.buffer = leaf;
    return lb;
  }
  if (TYPEOF(leaf) == VECSXP && Rf_inherits(leaf, "AnvlArray")) {
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
    if (data == R_NilValue || TYPEOF(data) != EXTPTRSXP) return lb;
    lb.ok = true;
    lb.buffer = data;
    lb.ambiguous = (amb != R_NilValue && Rf_asLogical(amb) == TRUE);
    return lb;
  }
  return lb;
}

// What a compiled program needs at execute time -- mirrors anvl's stored cache
// value list(exec, const_arrays, phantom_specs, device, ...). The structural
// out_tree / ambiguity are carried as opaque R objects for the caller to use
// when wrapping outputs (this engine returns raw output buffers).
struct CacheEntry {
  SEXP exec = R_NilValue;          // PJRTLoadedExecutable xptr (preserved)
  std::vector<SEXP> const_arrays;  // (preserved)
  const void* device = nullptr;
};

// Per-jit dispatcher: owns the executable cache, the R cache_miss callback
// (compiles on a miss), and a reusable default-options object. R objects held
// in the cache are R_PreserveObject'd on insert and released on eviction /
// teardown via the cache's on_evict hook + clear().
class Dispatcher {
 public:
  Dispatcher(std::size_t capacity, SEXP miss_fn, SEXP opts)
      : cache_(capacity,
               [](CacheEntry& e) {
                 if (e.exec != R_NilValue) R_ReleaseObject(e.exec);
                 for (SEXP c : e.const_arrays) R_ReleaseObject(c);
               }),
        miss_fn_(miss_fn),
        opts_(opts) {
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

 private:
  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq> cache_;
  SEXP miss_fn_;
  SEXP opts_;
};

}  // namespace rpjrt

// These are defined in pjrt.cpp; the dispatcher reuses them directly so the
// keepalive/donation/options logic lives in one place.
Rcpp::List impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input,
    Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options);
Rcpp::XPtr<rpjrt::PJRTExecuteOptions> impl_execution_options_create(
    std::vector<int64_t> non_donatable_input_indices, int launch_id);

// Self-test entry: flatten `x`, rebuild it, and hash the tree. Lets the R tests
// check the native Node against R's flatten()/unflatten() on real inputs.
// [[Rcpp::export]]
Rcpp::List impl_dispatch_node_selftest(SEXP x) {
  using namespace rpjrt;
  std::vector<SEXP> leaves;
  Node tree;
  int counter = 0;
  flatten_rec(x, leaves, tree, counter);

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
// size.
// [[Rcpp::export]]
SEXP impl_dispatch_create(int capacity, SEXP miss_fn) {
  using namespace rpjrt;
  if (TYPEOF(miss_fn) != CLOSXP && TYPEOF(miss_fn) != BUILTINSXP &&
      TYPEOF(miss_fn) != SPECIALSXP) {
    Rcpp::stop("miss_fn must be a function");
  }
  Rcpp::XPtr<PJRTExecuteOptions> opts =
      impl_execution_options_create(std::vector<int64_t>(), 0);
  auto* d = new Dispatcher(static_cast<std::size_t>(capacity), miss_fn,
                           static_cast<SEXP>(opts));
  Rcpp::XPtr<Dispatcher> ptr(d, true);
  ptr.attr("class") = "PJRTDispatcher";
  return ptr;
}

// Run the native dispatch for `args` (the evaluated argument list of the call).
// Returns the raw output buffers (a list of PJRTBuffer xptrs) on success, or
// the dispatch sentinel when the call is not handled natively (caller falls
// back).
// [[Rcpp::export]]
SEXP impl_dispatch_run(SEXP handle, Rcpp::List args) {
  using namespace rpjrt;
  Rcpp::XPtr<Dispatcher> d(handle);

  // 1. flatten args into leaves + structure.
  std::vector<SEXP> leaves;
  Node in_tree;
  int counter = 0;
  flatten_rec(args, leaves, in_tree, counter);

  // 2. classify every leaf; bail to the R slow path on anything unhandled.
  std::vector<LeafBuf> bufs;
  bufs.reserve(leaves.size());
  for (SEXP leaf : leaves) {
    LeafBuf lb = extract_leaf(leaf);
    if (!lb.ok) return impl_dispatch_sentinel();
    bufs.push_back(lb);
  }
  if (bufs.empty()) return impl_dispatch_sentinel();

  // 3. build the cache key (device from the first leaf; avals read natively).
  CacheKey key;
  key.in_tree = in_tree;
  key.leaves.reserve(bufs.size());
  for (std::size_t k = 0; k < bufs.size(); ++k) {
    const void* dev = nullptr;
    KeyLeaf kl;
    kl.is_static = false;
    kl.av = aval_from_buffer(bufs[k].buffer, bufs[k].ambiguous,
                             k == 0 ? &dev : nullptr);
    if (k == 0) key.device = dev;
    key.leaves.push_back(std::move(kl));
  }

  // 4. probe cache; compile via the R miss callback on a miss.
  CacheEntry* entry = d->cache().get(key);
  if (entry == nullptr) {
    Rcpp::Function miss(d->miss_fn());
    Rcpp::List res = miss(args);
    CacheEntry e;
    e.exec = res["exec"];
    R_PreserveObject(e.exec);
    if (res.containsElementNamed("const_arrays")) {
      Rcpp::List consts = res["const_arrays"];
      for (R_xlen_t i = 0; i < consts.size(); ++i) {
        SEXP c = consts[i];
        R_PreserveObject(c);
        e.const_arrays.push_back(c);
      }
    }
    e.device = key.device;
    d->cache().set(key, std::move(e));
    entry = d->cache().get(key);
  }

  // 5. assemble inputs: const_arrays ++ leaf buffers, then execute.
  Rcpp::List inputs(entry->const_arrays.size() + bufs.size());
  R_xlen_t pos = 0;
  for (SEXP c : entry->const_arrays) inputs[pos++] = c;
  for (const LeafBuf& lb : bufs) inputs[pos++] = lb.buffer;

  Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> exec(entry->exec);
  Rcpp::XPtr<rpjrt::PJRTExecuteOptions> opts(d->opts());
  return impl_loaded_executable_execute(exec, inputs, opts);
}

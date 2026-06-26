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
#include <functional>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

}  // namespace rpjrt

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

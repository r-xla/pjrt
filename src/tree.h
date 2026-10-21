// The Rtree module: capture the nesting structure of an R object as a tree so
// a flat leaf list can be extracted (flatten) and the structure restored
// (unflatten). This is pjrt's R analog of JAX's pytree (hence "Rtree").
// Semantics:
//   * NULL              -> NullNode (no leaf, but kept in the structure)
//   * bare list         -> ListNode (recurses; remembers names)
//   * object of a registered class
//                       -> ObjectNode (recurses into the children the class's
//                          `flatten` returns; remembers their names and the
//                          class, whose `unflatten` rebuilds the object)
//   * anything else     -> LeafNode (a classed object or atomic is a leaf)
// "bare list" == VECSXP without a class attribute, matching R's is.object().
// Classes are registered with register_tree_node() (see tree_node_class()).
//
// Representation: a tree is stored as a flat structure-of-arrays, one entry per
// node in preorder (the flatten / DFS order). This keeps a whole tree in a
// handful of contiguous vectors -- O(1) heap allocations regardless of node
// count -- and makes equality and hashing linear, cache-friendly scans. A
// leaf's index is implicit (its rank in the preorder), so no per-node index is
// stored.
//
// This is the single source of truth for the flatten semantics; tree.cpp
// exposes it to R behind an external pointer (S3 class "RTree").

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rpjrt {

struct RTree {
  enum Kind : std::uint8_t { LeafNode, ListNode, NullNode, ObjectNode };

  // Parallel arrays, indexed by node in preorder. Preorder means each node is
  // emitted before its children (which follow left-to-right), so a parent sits
  // immediately before its subtree and the whole subtree is one contiguous run.
  std::vector<std::uint8_t> kind;
  // ListNode / ObjectNode: #children; else 0
  std::vector<std::int32_t> n_children;
  std::vector<std::int32_t> subtree_nodes;  // #nodes in this subtree incl. self
  // index into the `names` array, i.e. the start index.
  // named list of length 0 still has a start index but because n_children==0
  // the sequence is empty and hence there are no names. this is needed to
  // distinguish it from a unnamed list of length 0, which has name_off = -1
  std::vector<std::int32_t> name_off;
  // All child-name strings, concatenated in node preorder: the names of an
  // earlier (by node index) named list node come before those of a later one.
  std::vector<std::string> names;
  // ObjectNode: the registered class of the object; "" for any other node.
  std::vector<std::string> node_class;

  bool is_named(std::size_t pos) const { return name_off[pos] != -1; }
  // A node with children: a list or an object.
  bool has_children(std::size_t pos) const {
    return kind[pos] == ListNode || kind[pos] == ObjectNode;
  }
};

inline bool is_bare_list(SEXP x) {
  return TYPEOF(x) == VECSXP && !Rf_isObject(x);
}

// A class registered with register_tree_node(): `flatten(x)` returns the
// children of an object of the class as a bare list, and `unflatten(children)`
// rebuilds an object from such a list.
struct TreeNodeType {
  Rcpp::RObject flatten;
  Rcpp::RObject unflatten;
};

// The registered classes, by class name.
std::unordered_map<std::string, TreeNodeType>& tree_node_types();

// The registered class `x` is an object of (its first class that is
// registered), or nullptr. Cheap when nothing is registered.
inline const std::pair<const std::string, TreeNodeType>* tree_node_class(
    SEXP x) {
  auto& types = tree_node_types();
  if (types.empty() || !Rf_isObject(x)) return nullptr;
  SEXP cls = Rf_getAttrib(x, R_ClassSymbol);
  for (R_xlen_t k = 0; k < XLENGTH(cls); ++k) {
    auto it = types.find(CHAR(STRING_ELT(cls, k)));
    if (it != types.end()) return &*it;
  }
  return nullptr;
}

inline void flatten_rec(SEXP x, std::vector<SEXP>& leaves, RTree& tree,
                        std::vector<Rcpp::RObject>& keep);

// Append a node with the children `x` (a list, possibly named) to `tree` and
// recurse into them; `kind` and `cls` say what node it is.
inline void flatten_children(SEXP x, RTree::Kind kind, const std::string& cls,
                             std::vector<SEXP>& leaves, RTree& tree,
                             std::vector<Rcpp::RObject>& keep) {
  const std::size_t pos = tree.kind.size();
  const R_xlen_t n = XLENGTH(x);
  SEXP nms = Rf_getAttrib(x, R_NamesSymbol);
  const bool has_names = (nms != R_NilValue);
  tree.kind.push_back(kind);
  tree.n_children.push_back(static_cast<std::int32_t>(n));
  tree.subtree_nodes.push_back(0);  // place-holder. Set correctly at the end
  tree.node_class.push_back(cls);
  if (has_names) {
    tree.name_off.push_back(static_cast<std::int32_t>(tree.names.size()));
    for (R_xlen_t k = 0; k < n; ++k) {
      SEXP nm = STRING_ELT(nms, k);
      // NA_STRING would round-trip through CHAR() as the literal "NA",
      // silently conflating an NA name with a "NA" name (and poisoning
      // tree_equal); reject it instead.
      if (nm == NA_STRING) Rcpp::stop("tree names must not be NA");
      tree.names.push_back(std::string(CHAR(nm)));
    }
  } else {
    tree.name_off.push_back(-1);
  }
  for (R_xlen_t k = 0; k < n; ++k) {
    flatten_rec(VECTOR_ELT(x, k), leaves, tree, keep);
  }
  tree.subtree_nodes[pos] = static_cast<std::int32_t>(tree.kind.size() - pos);
}

// Append the preorder encoding of `x` to `t`, collecting leaves in order. The
// children a registered class's `flatten` returns are kept alive in `keep`, as
// the leaves may point into them.
inline void flatten_rec(SEXP x, std::vector<SEXP>& leaves, RTree& tree,
                        std::vector<Rcpp::RObject>& keep) {
  if (x == R_NilValue) {
    tree.kind.push_back(RTree::NullNode);
    tree.n_children.push_back(0);
    tree.subtree_nodes.push_back(1);
    tree.name_off.push_back(-1);
    tree.node_class.push_back(std::string());
    return;
  }
  if (is_bare_list(x)) {
    flatten_children(x, RTree::ListNode, std::string(), leaves, tree, keep);
    return;
  }
  if (const auto* type = tree_node_class(x)) {
    Rcpp::Function flatten(type->second.flatten);
    Rcpp::RObject children = flatten(x);
    if (!is_bare_list(children)) {
      Rcpp::stop("the `flatten` registered for class \"%s\" must return a list",
                 type->first.c_str());
    }
    keep.push_back(children);
    flatten_children(children, RTree::ObjectNode, type->first, leaves, tree,
                     keep);
    return;
  }
  tree.kind.push_back(RTree::LeafNode);
  tree.n_children.push_back(0);
  tree.subtree_nodes.push_back(1);
  tree.name_off.push_back(-1);
  tree.node_class.push_back(std::string());
  leaves.push_back(x);
}

// Reconstruct an R object from the subtree at `pos`, consuming leaves in order
// via `li`. `pos` advances past the whole subtree (children follow their parent
// contiguously in preorder). The caller must size `leaves` to the leaf count.
inline SEXP unflatten_rec(const RTree& tree, const std::vector<SEXP>& leaves,
                          std::size_t& pos, std::size_t& li) {
  const std::size_t node = pos++;
  switch (tree.kind[node]) {
    case RTree::NullNode:
      return R_NilValue;
    case RTree::LeafNode:
      return leaves[li++];
    case RTree::ListNode:
    case RTree::ObjectNode: {
      const int n = tree.n_children[node];
      SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
      for (int k = 0; k < n; ++k) {
        SET_VECTOR_ELT(out, k, unflatten_rec(tree, leaves, pos, li));
      }
      if (tree.is_named(node)) {
        const std::int32_t off = tree.name_off[node];
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
        for (int k = 0; k < n; ++k) {
          SET_STRING_ELT(nms, k, Rf_mkChar(tree.names[off + k].c_str()));
        }
        Rf_setAttrib(out, R_NamesSymbol, nms);
        UNPROTECT(1);
      }
      if (tree.kind[node] == RTree::ObjectNode) {
        auto it = tree_node_types().find(tree.node_class[node]);
        if (it == tree_node_types().end()) {
          UNPROTECT(1);
          Rcpp::stop("the tree node class \"%s\" is no longer registered",
                     tree.node_class[node].c_str());
        }
        Rcpp::Function unflatten(it->second.unflatten);
        out = unflatten(out);
      }
      UNPROTECT(1);
      return out;
    }
  }
  return R_NilValue;  // unreachable
}

// Number of leaves in the subtree rooted at `pos` (== length of its flat list).
inline int subtree_leaf_count(const RTree& tree, std::size_t pos) {
  const std::size_t end =
      pos + static_cast<std::size_t>(tree.subtree_nodes[pos]);
  int c = 0;
  for (std::size_t q = pos; q < end; ++q) {
    if (tree.kind[q] == RTree::LeafNode) ++c;
  }
  return c;
}

// Number of leaves in the whole tree.
inline int tree_size_rec(const RTree& tree) {
  return tree.kind.empty() ? 0 : subtree_leaf_count(tree, 0);
}

// The preorder node indices of node `pos`'s direct children (found by skipping
// each child's subtree via subtree_nodes).
inline std::vector<std::size_t> child_nodes(const RTree& tree,
                                            std::size_t pos) {
  std::vector<std::size_t> out;
  out.reserve(static_cast<std::size_t>(tree.n_children[pos]));
  std::size_t c = pos + 1;
  for (int k = 0; k < tree.n_children[pos]; ++k) {
    out.push_back(c);
    c += static_cast<std::size_t>(tree.subtree_nodes[c]);
  }
  return out;
}

// Structural equality: identical kinds, child counts, name offsets, child
// names and object classes -- together the full canonical encoding.
// subtree_nodes is derived from the shape, so it needs no separate check; and
// name_off is canonical (a named node's offset is fixed by the shape, and -1
// exactly when unnamed), so comparing it directly also captures each node's
// named-ness.
inline bool tree_eq(const RTree& a, const RTree& b) {
  return a.kind == b.kind && a.n_children == b.n_children &&
         a.name_off == b.name_off && a.names == b.names &&
         a.node_class == b.node_class;
}

// Structural hash of an RTree, consistent with tree_eq: trees that compare
// equal hash equally. Defined out-of-line in tree.cpp.
std::uint64_t tree_hash(const RTree& tree);

// The R accessor path of the `i`-th leaf (1-based, in preorder): "x$w[[2]]" and
// the like, or "" for a leaf at the root. Used to name the offending argument
// when the dispatcher rejects an input. Errors if `i` is out of range. Defined
// out-of-line in tree.cpp.
std::string tree_path(const RTree& tree, int i);

// Hand a heap-allocated RTree to R as a classed external pointer, transferring
// ownership (a finalizer deletes it on GC). The dispatcher uses this to pass
// the tree it already built to its compile callback. Defined in tree.cpp.
SEXP tree_xptr(RTree* n);

}  // namespace rpjrt

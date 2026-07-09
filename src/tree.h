// The Rtree module: capture the nesting structure of an R object as a tree so
// a flat leaf list can be extracted (flatten) and the structure restored
// (unflatten). This is pjrt's R analog of JAX's pytree (hence "Rtree").
// Semantics:
//   * NULL              -> NullNode (no leaf, but kept in the structure)
//   * bare list         -> ListNode (recurses; remembers names)
//   * anything else     -> LeafNode (a classed object or atomic is a leaf)
// "bare list" == VECSXP without a class attribute, matching R's is.object().
//
// Representation: a tree is stored as a flat structure-of-arrays, one entry per
// node in preorder (the flatten / DFS order). This keeps a whole tree in a
// handful of contiguous vectors -- O(1) heap allocations regardless of node
// count -- and makes equality/hashing linear, cache-friendly scans. A leaf's
// index is implicit (its rank in the preorder), so no per-node index is stored.
//
// This is the single source of truth for the flatten semantics; tree.cpp
// exposes it to R behind an external pointer (S3 class "RTree").

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rpjrt {

struct RTree {
  enum Kind : std::uint8_t { LeafNode, ListNode, NullNode };

  // Parallel arrays, indexed by node in preorder.
  std::vector<std::uint8_t> kind;
  std::vector<std::int32_t> n_children;     // ListNode: #children; else 0
  std::vector<std::int32_t> subtree_nodes;  // #nodes in this subtree incl. self
  // ListNode carrying names: start index of its child names in `names`; else
  // -1. The named/unnamed distinction is load-bearing for an *empty* list (it
  // tells `list()` from `structure(list(), names = character(0))`).
  std::vector<std::int32_t> name_off;
  std::vector<std::string> names;  // child-name strings, grouped by node

  std::size_t size() const { return kind.size(); }
  bool is_named(std::size_t p) const { return name_off[p] >= 0; }
};

// A bare list (recurse) vs a leaf (classed object / atomic / function / ...).
// Rf_isObject(x) == OBJECT(x): true iff x carries a class attribute, matching
// R's is.object() used by flatten()/build_tree() to decide leaf vs list.
inline bool is_bare_list(SEXP x) {
  return TYPEOF(x) == VECSXP && !Rf_isObject(x);
}

// Append the preorder encoding of `x` to `t`, collecting leaves in order.
inline void flatten_rec(SEXP x, std::vector<SEXP>& leaves, RTree& t) {
  const std::size_t p = t.kind.size();
  if (x == R_NilValue) {
    t.kind.push_back(RTree::NullNode);
    t.n_children.push_back(0);
    t.subtree_nodes.push_back(1);
    t.name_off.push_back(-1);
    return;
  }
  if (is_bare_list(x)) {
    const R_xlen_t n = XLENGTH(x);
    SEXP nms = Rf_getAttrib(x, R_NamesSymbol);
    const bool has_names = (nms != R_NilValue);
    t.kind.push_back(RTree::ListNode);
    t.n_children.push_back(static_cast<std::int32_t>(n));
    t.subtree_nodes.push_back(0);  // backpatched once the subtree is emitted
    if (has_names) {
      t.name_off.push_back(static_cast<std::int32_t>(t.names.size()));
      for (R_xlen_t k = 0; k < n; ++k) {
        t.names.push_back(std::string(CHAR(STRING_ELT(nms, k))));
      }
    } else {
      t.name_off.push_back(-1);
    }
    for (R_xlen_t k = 0; k < n; ++k) {
      flatten_rec(VECTOR_ELT(x, k), leaves, t);
    }
    t.subtree_nodes[p] = static_cast<std::int32_t>(t.kind.size() - p);
    return;
  }
  t.kind.push_back(RTree::LeafNode);
  t.n_children.push_back(0);
  t.subtree_nodes.push_back(1);
  t.name_off.push_back(-1);
  leaves.push_back(x);
}

// Reconstruct an R object from the subtree at `p`, consuming leaves in order
// via `li`. `p` advances past the whole subtree (children follow their parent
// contiguously in preorder). The caller must size `leaves` to the leaf count.
inline SEXP unflatten_rec(const RTree& t, const std::vector<SEXP>& leaves,
                          std::size_t& p, std::size_t& li) {
  const std::size_t node = p++;
  switch (t.kind[node]) {
    case RTree::NullNode:
      return R_NilValue;
    case RTree::LeafNode:
      return leaves[li++];
    case RTree::ListNode: {
      const int n = t.n_children[node];
      SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
      for (int k = 0; k < n; ++k) {
        SET_VECTOR_ELT(out, k, unflatten_rec(t, leaves, p, li));
      }
      if (t.is_named(node)) {
        const std::int32_t off = t.name_off[node];
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
        for (int k = 0; k < n; ++k) {
          SET_STRING_ELT(nms, k, Rf_mkChar(t.names[off + k].c_str()));
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

// Number of leaves in the subtree rooted at `p` (== length of its flat list).
inline int subtree_leaf_count(const RTree& t, std::size_t p) {
  const std::size_t end = p + static_cast<std::size_t>(t.subtree_nodes[p]);
  int c = 0;
  for (std::size_t q = p; q < end; ++q) {
    if (t.kind[q] == RTree::LeafNode) ++c;
  }
  return c;
}

// Number of leaves in the whole tree.
inline int tree_size_rec(const RTree& t) {
  return t.kind.empty() ? 0 : subtree_leaf_count(t, 0);
}

// The preorder node indices of node `p`'s direct children (found by skipping
// each child's subtree via subtree_nodes).
inline std::vector<std::size_t> child_nodes(const RTree& t, std::size_t p) {
  std::vector<std::size_t> out;
  out.reserve(static_cast<std::size_t>(t.n_children[p]));
  std::size_t c = p + 1;
  for (int k = 0; k < t.n_children[p]; ++k) {
    out.push_back(c);
    c += static_cast<std::size_t>(t.subtree_nodes[c]);
  }
  return out;
}

// Structural equality: identical shape (kinds + child counts), identical
// named-ness at each node, and identical child names. (Leaf positions and
// subtree sizes are implied by the shape, so they need no separate check.)
inline bool tree_eq(const RTree& a, const RTree& b) {
  if (a.kind != b.kind) return false;
  if (a.n_children != b.n_children) return false;
  if (a.names != b.names) return false;
  for (std::size_t p = 0; p < a.name_off.size(); ++p) {
    if ((a.name_off[p] < 0) != (b.name_off[p] < 0)) return false;
  }
  return true;
}

// Structural hash of an RTree, consistent with tree_eq: trees that compare
// equal hash equally. Defined out-of-line in tree.cpp.
std::size_t tree_hash(const RTree& tree);

}  // namespace rpjrt

// The Rtree module: capture the nesting structure of an R object as a tree so
// a flat leaf list can be extracted (flatten) and the structure restored
// (unflatten). This is pjrt's R analog of JAX's pytree (hence "Rtree").
// Semantics:
//   * NULL              -> NullNode (no leaf, no flat index, but kept in tree)
//   * bare list         -> ListNode (recurses; remembers names)
//   * anything else     -> LeafNode (a classed object or atomic is a leaf)
// "bare list" == VECSXP without a class attribute, matching R's is.object().
//
// This is the single source of truth for the flatten semantics; tree.cpp
// exposes it to R behind an external pointer (S3 class "RTree").

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <string>
#include <vector>

namespace rpjrt {

struct RTree {
  enum Kind { LeafNode, ListNode, NullNode };
  Kind kind;
  std::vector<RTree> children;  // ListNode: child subtrees
  // ListNode: whether names() was non-NULL. Redundant with !names.empty() for
  // a non-empty list, but load-bearing for an *empty* named list (`names` is
  // empty either way there), so it distinguishes `list()` from
  // `structure(list(), names = character(0))` on round-trip and in equality.
  bool has_names = false;
  std::vector<std::string>
      names;  // ListNode: child names ("" for unnamed slots)
};

// A bare list (recurse) vs a leaf (classed object / atomic / function / ...).
// Rf_isObject(x) == OBJECT(x): true iff x carries a class attribute, matching
// R's is.object() used by flatten()/build_tree() to decide leaf vs list.
inline bool is_bare_list(SEXP x) {
  return TYPEOF(x) == VECSXP && !Rf_isObject(x);
}

// Flatten `x` into `leaves` (in order) and fill `tree` with its structure.
// A leaf's index is implicit: its position in `leaves` (i.e. its rank in an
// in-order traversal), so the tree stores no explicit leaf index.
inline void flatten_rec(SEXP x, std::vector<SEXP>& leaves, RTree& tree) {
  if (x == R_NilValue) {
    tree.kind = RTree::NullNode;
    return;
  }
  if (is_bare_list(x)) {
    tree.kind = RTree::ListNode;
    const R_xlen_t n = XLENGTH(x);
    SEXP nms = Rf_getAttrib(x, R_NamesSymbol);
    tree.has_names = (nms != R_NilValue);
    tree.children.resize(n);
    if (tree.has_names) {
      tree.names.resize(n);
      for (R_xlen_t k = 0; k < n; ++k) {
        tree.names[k] = std::string(CHAR(STRING_ELT(nms, k)));
      }
    }
    for (R_xlen_t k = 0; k < n; ++k) {
      flatten_rec(VECTOR_ELT(x, k), leaves, tree.children[k]);
    }
    return;
  }
  tree.kind = RTree::LeafNode;
  leaves.push_back(x);
}

// Reconstruct an R object from a tree and a flat leaf list (mirrors unflatten).
// Leaves are consumed left-to-right via `pos`; the caller must ensure
// `leaves.size()` equals the tree's leaf count.
inline SEXP unflatten_rec(const RTree& tree, const std::vector<SEXP>& leaves,
                          std::size_t& pos) {
  switch (tree.kind) {
    case RTree::NullNode:
      return R_NilValue;
    case RTree::LeafNode:
      return leaves[pos++];
    case RTree::ListNode: {
      const std::size_t n = tree.children.size();
      SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
      for (std::size_t k = 0; k < n; ++k) {
        SET_VECTOR_ELT(out, k, unflatten_rec(tree.children[k], leaves, pos));
      }
      if (tree.has_names) {
        SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
        for (std::size_t k = 0; k < n; ++k) {
          SET_STRING_ELT(nms, k, Rf_mkChar(tree.names[k].c_str()));
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

// Structural equality: two trees are equal iff identical kind, child
// structure, and names. (Leaf positions are implicit in the traversal, so two
// leaves are always structurally equal.)
inline bool tree_eq(const RTree& a, const RTree& b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case RTree::NullNode:
      return true;
    case RTree::LeafNode:
      return true;
    case RTree::ListNode:
      if (a.children.size() != b.children.size()) return false;
      if (a.has_names != b.has_names) return false;
      if (a.has_names && a.names != b.names) return false;
      for (std::size_t k = 0; k < a.children.size(); ++k) {
        if (!tree_eq(a.children[k], b.children[k])) return false;
      }
      return true;
  }
  return false;
}

// Structural hash of an RTree, consistent with tree_eq: trees that compare
// equal (same kind, child structure, leaf indices, names) hash equally. Used
// as dispatch cache-key material; defined out-of-line in tree.cpp.
std::size_t tree_hash(const RTree& tree);

// Number of leaves under `tree` (== length of the flat list).
inline int tree_size_rec(const RTree& tree) {
  switch (tree.kind) {
    case RTree::NullNode:
      return 0;
    case RTree::LeafNode:
      return 1;
    case RTree::ListNode: {
      int n = 0;
      for (const RTree& child : tree.children) n += tree_size_rec(child);
      return n;
    }
  }
  return 0;  // unreachable
}

}  // namespace rpjrt

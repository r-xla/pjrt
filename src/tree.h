// The pytree `Node` module: capture the nesting structure of an R object as a
// tree so a flat leaf list can be extracted (flatten) and the structure
// restored (unflatten). Semantics:
//   * NULL              -> NullNode (no leaf, no flat index, but kept in tree)
//   * bare list         -> ListNode (recurses; remembers names)
//   * anything else     -> LeafNode (a classed object or atomic is a leaf)
// "bare list" == VECSXP without a class attribute, matching R's is.object().
//
// This is the single source of truth for the flatten semantics: the dispatch
// hot path (dispatch.cpp) uses it on the stack, and tree.cpp exposes it to R
// behind an external pointer (S3 class "PJRTNode").

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <string>
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
inline bool is_bare_list(SEXP x) {
  return TYPEOF(x) == VECSXP && !Rf_isObject(x);
}

// Flatten `x` into `leaves` (in order), fill `node` with its structure, and push
// each leaf's static-ness into `is_static`. `inherited_static` is propagated to
// all descendants (a static top-level arg makes all its leaves static). `counter`
// assigns 1-based leaf indices in flatten order (matches build_tree).
inline void flatten_rec(SEXP x, std::vector<SEXP>& leaves,
                        std::vector<char>& is_static, Node& node, int& counter,
                        bool inherited_static) {
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
      flatten_rec(VECTOR_ELT(x, k), leaves, is_static, node.nodes[k], counter,
                  inherited_static);
    }
    return;
  }
  node.kind = Node::LeafNode;
  node.i = ++counter;
  leaves.push_back(x);
  is_static.push_back(inherited_static ? 1 : 0);
}

// Reconstruct an R object from a tree and a flat leaf list (mirrors unflatten).
inline SEXP unflatten_rec(const Node& node, const std::vector<SEXP>& leaves) {
  switch (node.kind) {
    case Node::NullNode:
      return R_NilValue;
    case Node::LeafNode:
      if (node.i < 1 || static_cast<std::size_t>(node.i) > leaves.size()) {
        Rf_error("unflatten: leaf index %d out of bounds (%d leaves)", node.i,
                 static_cast<int>(leaves.size()));
      }
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
// structure, leaf indices, and names.
inline void hash_combine(std::size_t& seed, std::size_t v) {
  seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

inline std::size_t node_hash(const Node& node) {
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

inline bool node_eq(const Node& a, const Node& b) {
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

// Number of leaves under `node` (== length of the flat list).
inline int tree_size_rec(const Node& node) {
  switch (node.kind) {
    case Node::NullNode:
      return 0;
    case Node::LeafNode:
      return 1;
    case Node::ListNode: {
      int n = 0;
      for (const Node& child : node.nodes) n += tree_size_rec(child);
      return n;
    }
  }
  return 0;  // unreachable
}

}  // namespace rpjrt

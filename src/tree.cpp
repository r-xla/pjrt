// The R-exposed pytree API over the Node module in tree.h. A tree handed to R
// is a heap Node behind an Rcpp::XPtr with S3 class "PJRTNode"; it is opaque
// to R -- every operation that walks the tree walks it here and returns either
// a fully-owned result Node or a plain R value (mask, sizes, names, strings).
// No sub-node handles escape to R (a child handle would alias into a
// parent-owned subtree).

#include "tree.h"

#include <Rcpp.h>

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace rpjrt {

static SEXP node_xptr(Node* n) {
  Rcpp::XPtr<Node> ptr(n, true);
  ptr.attr("class") = "PJRTNode";
  return ptr;
}

// Reassign leaf indices in structure order so they form a contiguous sequence
// continuing from `counter` (used after filtering / when concatenating trees).
static void reindex_rec(Node& node, int& counter) {
  switch (node.kind) {
    case Node::NullNode:
      break;
    case Node::LeafNode:
      node.i = ++counter;
      break;
    case Node::ListNode:
      for (Node& child : node.nodes) reindex_rec(child, counter);
      break;
  }
}

// The canonical structural string: "*" for a leaf, "NULL" for a null node,
// "list(a = *, list(*, NULL))" for lists ("" names are printed positionally).
static void repr_rec(const Node& node, std::string& out) {
  switch (node.kind) {
    case Node::LeafNode:
      out += "*";
      return;
    case Node::NullNode:
      out += "NULL";
      return;
    case Node::ListNode: {
      if (node.nodes.empty()) {
        out += "list()";
        return;
      }
      out += "list(";
      for (std::size_t k = 0; k < node.nodes.size(); ++k) {
        if (k > 0) out += ", ";
        if (node.has_names && !node.names[k].empty()) {
          out += node.names[k];
          out += " = ";
        }
        repr_rec(node.nodes[k], out);
      }
      out += ")";
      return;
    }
  }
}

static std::string node_repr(const Node& node) {
  std::string out;
  repr_rec(node, out);
  return out;
}

// The path suffix for child j of a list: "$name" / "name" (at the root) for a
// named slot, "[[j]]" for an unnamed one.
static std::string path_suffix(const Node& parent, std::size_t j,
                               const std::string& prefix) {
  const std::string nm = parent.has_names ? parent.names[j] : std::string();
  if (!nm.empty()) {
    return prefix.empty() ? nm : "$" + nm;
  }
  return "[[" + std::to_string(j + 1) + "]]";
}

// Find the path of the leaf with flat index `i`; only descends into the branch
// containing it. A leaf at the root has path "" (there is nothing to name).
static bool tree_path_rec(const Node& node, int i, const std::string& prefix,
                          std::string& out) {
  switch (node.kind) {
    case Node::LeafNode:
      out = prefix;
      return true;
    case Node::NullNode:
      return false;
    case Node::ListNode:
      for (std::size_t j = 0; j < node.nodes.size(); ++j) {
        const Node& child = node.nodes[j];
        const std::string child_prefix = prefix + path_suffix(node, j, prefix);
        if (child.kind == Node::LeafNode) {
          if (child.i == i) {
            out = child_prefix;
            return true;
          }
        } else if (tree_path_rec(child, i, child_prefix, out)) {
          return true;
        }
      }
      return false;
  }
  return false;
}

// Walk two trees in parallel; report the first structural divergence as the
// path prefix plus pointers to the two diverging subtrees.
static bool tree_diff_rec(const Node& a, const Node& b,
                          const std::string& prefix, std::string& out_prefix,
                          const Node** out_a, const Node** out_b) {
  auto diverge = [&]() {
    out_prefix = prefix;
    *out_a = &a;
    *out_b = &b;
    return true;
  };
  if (a.kind != b.kind) return diverge();
  switch (a.kind) {
    case Node::NullNode:
      return false;
    case Node::LeafNode:
      return a.i == b.i ? false : diverge();
    case Node::ListNode: {
      if (a.nodes.size() != b.nodes.size() || a.has_names != b.has_names ||
          (a.has_names && a.names != b.names)) {
        return diverge();
      }
      for (std::size_t j = 0; j < a.nodes.size(); ++j) {
        if (tree_diff_rec(a.nodes[j], b.nodes[j],
                          prefix + path_suffix(a, j, prefix), out_prefix, out_a,
                          out_b)) {
          return true;
        }
      }
      return false;
    }
  }
  return false;
}

static const Node& as_node(SEXP handle) {
  Rcpp::XPtr<Node> ptr(handle);
  return *ptr;
}

static void require_list(const Node& node, const char* what) {
  if (node.kind != Node::ListNode) {
    Rcpp::stop("%s requires a tree whose root is a list node", what);
  }
}

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP impl_tree_build(SEXP x) {
  using namespace rpjrt;
  auto* node = new Node();
  std::vector<SEXP> leaves;
  std::vector<char> is_static;
  int counter = 0;
  flatten_rec(x, leaves, is_static, *node, counter,
              /*inherited_static=*/false);
  return node_xptr(node);
}

// [[Rcpp::export]]
Rcpp::List impl_tree_flatten(SEXP x) {
  using namespace rpjrt;
  Node node;
  std::vector<SEXP> leaves;
  std::vector<char> is_static;
  int counter = 0;
  flatten_rec(x, leaves, is_static, node, counter, /*inherited_static=*/false);
  Rcpp::List out(leaves.size());
  for (std::size_t k = 0; k < leaves.size(); ++k) out[k] = leaves[k];
  return out;
}

// [[Rcpp::export]]
SEXP impl_tree_unflatten(SEXP node, Rcpp::List x) {
  using namespace rpjrt;
  std::vector<SEXP> leaves(x.begin(), x.end());
  return unflatten_rec(as_node(node), leaves);
}

// [[Rcpp::export]]
int impl_tree_size(SEXP node) {
  return rpjrt::tree_size_rec(rpjrt::as_node(node));
}

// [[Rcpp::export]]
bool impl_tree_equal(SEXP a, SEXP b) {
  return rpjrt::node_eq(rpjrt::as_node(a), rpjrt::as_node(b));
}

// [[Rcpp::export]]
std::string impl_tree_kind(SEXP node) {
  using namespace rpjrt;
  switch (as_node(node).kind) {
    case Node::LeafNode:
      return "leaf";
    case Node::ListNode:
      return "list";
    case Node::NullNode:
      return "null";
  }
  return "";  // unreachable
}

// [[Rcpp::export]]
SEXP impl_tree_names(SEXP node) {
  using namespace rpjrt;
  const Node& n = as_node(node);
  if (n.kind != Node::ListNode || !n.has_names) return R_NilValue;
  return Rcpp::wrap(n.names);
}

// [[Rcpp::export]]
Rcpp::CharacterVector impl_tree_child_kinds(SEXP node) {
  using namespace rpjrt;
  const Node& n = as_node(node);
  require_list(n, "child_kinds()");
  Rcpp::CharacterVector out(n.nodes.size());
  for (std::size_t k = 0; k < n.nodes.size(); ++k) {
    switch (n.nodes[k].kind) {
      case Node::LeafNode:
        out[k] = "leaf";
        break;
      case Node::ListNode:
        out[k] = "list";
        break;
      case Node::NullNode:
        out[k] = "null";
        break;
    }
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::IntegerVector impl_tree_child_sizes(SEXP node) {
  using namespace rpjrt;
  const Node& n = as_node(node);
  require_list(n, "child_sizes()");
  Rcpp::IntegerVector out(n.nodes.size());
  for (std::size_t k = 0; k < n.nodes.size(); ++k) {
    out[k] = tree_size_rec(n.nodes[k]);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::CharacterVector impl_tree_flat_names(SEXP node) {
  using namespace rpjrt;
  const Node& n = as_node(node);
  require_list(n, "flat_names()");
  Rcpp::CharacterVector out(tree_size_rec(n));
  R_xlen_t pos = 0;
  for (std::size_t k = 0; k < n.nodes.size(); ++k) {
    const int sz = tree_size_rec(n.nodes[k]);
    const std::string nm = n.has_names ? n.names[k] : std::string();
    for (int j = 0; j < sz; ++j) out[pos++] = nm;
  }
  return out;
}

// [[Rcpp::export]]
std::string impl_tree_path(SEXP node, int i) {
  using namespace rpjrt;
  std::string out;
  if (!tree_path_rec(as_node(node), i, "", out)) {
    Rcpp::stop("tree_path(): no leaf with index %d in the tree", i);
  }
  return out;
}

// [[Rcpp::export]]
SEXP impl_tree_filter_by_names(SEXP node, Rcpp::CharacterVector names) {
  using namespace rpjrt;
  const Node& n = as_node(node);
  require_list(n, "filter_by_names()");
  if (!n.has_names) Rcpp::stop("filter_by_names(): tree must have names");
  std::unordered_set<std::string> keep(names.size());
  for (R_xlen_t k = 0; k < names.size(); ++k) {
    keep.insert(std::string(names[k]));
  }
  auto* out = new Node();
  out->kind = Node::ListNode;
  out->has_names = true;
  int counter = 0;
  for (std::size_t k = 0; k < n.nodes.size(); ++k) {
    if (keep.count(n.names[k]) == 0) continue;
    out->nodes.push_back(n.nodes[k]);  // copy the kept subtree
    out->names.push_back(n.names[k]);
    reindex_rec(out->nodes.back(), counter);
  }
  return node_xptr(out);
}

// [[Rcpp::export]]
SEXP impl_tree_concat(Rcpp::List nodes, SEXP names) {
  using namespace rpjrt;
  auto* out = new Node();
  out->kind = Node::ListNode;
  out->nodes.reserve(nodes.size());
  int counter = 0;
  for (R_xlen_t k = 0; k < nodes.size(); ++k) {
    out->nodes.push_back(as_node(nodes[k]));  // copy the child tree
    reindex_rec(out->nodes.back(), counter);
  }
  if (names != R_NilValue) {
    Rcpp::CharacterVector nms(names);
    if (nms.size() != nodes.size()) {
      Rcpp::stop("tree_concat(): names must match the number of nodes");
    }
    out->has_names = true;
    out->names.resize(nms.size());
    for (R_xlen_t k = 0; k < nms.size(); ++k) {
      out->names[k] = std::string(nms[k]);
    }
  }
  return node_xptr(out);
}

// [[Rcpp::export]]
Rcpp::LogicalVector impl_tree_mask_from_names(SEXP node,
                                              Rcpp::CharacterVector names) {
  using namespace rpjrt;
  const Node& n = as_node(node);
  require_list(n, "mask_from_names()");
  std::unordered_set<std::string> marked(names.size());
  for (R_xlen_t k = 0; k < names.size(); ++k) {
    marked.insert(std::string(names[k]));
  }
  Rcpp::LogicalVector out(tree_size_rec(n));
  R_xlen_t pos = 0;
  for (std::size_t k = 0; k < n.nodes.size(); ++k) {
    const bool m = n.has_names && marked.count(n.names[k]) > 0;
    const int sz = tree_size_rec(n.nodes[k]);
    for (int j = 0; j < sz; ++j) out[pos++] = m;
  }
  return out;
}

// [[Rcpp::export]]
std::string impl_tree_repr(SEXP node) {
  return rpjrt::node_repr(rpjrt::as_node(node));
}

// [[Rcpp::export]]
SEXP impl_tree_diff(SEXP a, SEXP b) {
  using namespace rpjrt;
  std::string prefix;
  const Node* sub_a = nullptr;
  const Node* sub_b = nullptr;
  if (!tree_diff_rec(as_node(a), as_node(b), "", prefix, &sub_a, &sub_b)) {
    return R_NilValue;
  }
  return Rcpp::List::create(Rcpp::Named("prefix") = prefix,
                            Rcpp::Named("a") = node_repr(*sub_a),
                            Rcpp::Named("b") = node_repr(*sub_b));
}

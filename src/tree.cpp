// The R-exposed Rtree API over the RTree module in tree.h. A tree handed to R
// is a heap RTree behind an Rcpp::XPtr with S3 class "RTree"; it is opaque
// to R -- every operation that walks the tree walks it here and returns either
// a fully-owned result RTree or a plain R value (mask, sizes, names, strings).
// No sub-tree handles escape to R (a child handle would alias into a
// parent-owned subtree).

#include "tree.h"

#include <Rcpp.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "hash.h"

namespace rpjrt {

// Structural hash of an RTree, consistent with tree_eq (declared in tree.h).
// Used as dispatch cache-key material: trees that compare equal hash equally.
std::size_t tree_hash(const RTree& tree) {
  std::size_t h = static_cast<std::size_t>(tree.kind);
  switch (tree.kind) {
    case RTree::NullNode:
      break;
    case RTree::LeafNode:
      hash_combine(h, static_cast<std::size_t>(tree.i));
      break;
    case RTree::ListNode:
      hash_combine(h, tree.children.size());
      hash_combine(h, tree.has_names ? 1u : 0u);
      for (const auto& nm : tree.names) {
        hash_combine(h, std::hash<std::string>{}(nm));
      }
      for (const auto& child : tree.children) {
        hash_combine(h, tree_hash(child));
      }
      break;
  }
  return h;
}

// Wrap a heap-allocated RTree in an external pointer handed to R. Ownership
// transfers to R: `true` registers a delete finalizer, so ~RTree() runs
// (freeing the whole subtree) when the handle is garbage-collected.
static SEXP tree_xptr(RTree* n) {
  Rcpp::XPtr<RTree> ptr(n, true);  // true: delete `n` on GC
  ptr.attr("class") = "RTree";     // tag with the S3 class R dispatches on
  return ptr;
}

// Reassign leaf indices in structure order so they form a contiguous sequence
// continuing from `counter` (used after filtering / when concatenating trees).
static void reindex_rec(RTree& tree, int& counter) {
  switch (tree.kind) {
    case RTree::NullNode:
      break;
    case RTree::LeafNode:
      tree.i = ++counter;
      break;
    case RTree::ListNode:
      for (RTree& child : tree.children) reindex_rec(child, counter);
      break;
  }
}

// The canonical structural string: "*" for a leaf, "NULL" for a null tree,
// "list(a = *, list(*, NULL))" for lists ("" names are printed positionally).
// Printer does not distinguish between empty named list and list
static void repr_rec(const RTree& tree, std::string& out) {
  switch (tree.kind) {
    case RTree::LeafNode:
      out += "*";
      return;
    case RTree::NullNode:
      out += "NULL";
      return;
    case RTree::ListNode: {
      if (tree.children.empty()) {
        out += "list()";
        return;
      }
      out += "list(";
      for (std::size_t k = 0; k < tree.children.size(); ++k) {
        if (k > 0) out += ", ";
        if (tree.has_names && !tree.names[k].empty()) {
          out += tree.names[k];
          out += " = ";
        }
        repr_rec(tree.children[k], out);
      }
      out += ")";
      return;
    }
  }
}

static std::string tree_repr(const RTree& tree) {
  std::string out;
  repr_rec(tree, out);
  return out;
}

// The path suffix for child j of a list: "$name" / "name" (at the root) for a
// named slot, "[[j]]" for an unnamed one.
static std::string path_suffix(const RTree& parent, std::size_t j,
                               const std::string& prefix) {
  const std::string nm = parent.has_names ? parent.names[j] : std::string();
  if (!nm.empty()) {
    return prefix.empty() ? nm : "$" + nm;
  }
  return "[[" + std::to_string(j + 1) + "]]";
}

// Find the path of the leaf with flat index `i`; only descends into the branch
// containing it. A leaf at the root has path "" (there is nothing to name).
static bool tree_path_rec(const RTree& tree, int i, const std::string& prefix,
                          std::string& out) {
  switch (tree.kind) {
    case RTree::LeafNode:
      if (tree.i != i) return false;
      out = prefix;
      return true;
    case RTree::NullNode:
      return false;
    case RTree::ListNode:
      for (std::size_t j = 0; j < tree.children.size(); ++j) {
        const RTree& child = tree.children[j];
        const std::string child_prefix = prefix + path_suffix(tree, j, prefix);
        if (child.kind == RTree::LeafNode) {
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
static bool tree_diff_rec(const RTree& a, const RTree& b,
                          const std::string& prefix, std::string& out_prefix,
                          const RTree** out_a, const RTree** out_b) {
  auto diverge = [&]() {
    out_prefix = prefix;
    *out_a = &a;
    *out_b = &b;
    return true;
  };
  if (a.kind != b.kind) return diverge();
  switch (a.kind) {
    case RTree::NullNode:
      return false;
    case RTree::LeafNode:
      return a.i == b.i ? false : diverge();
    case RTree::ListNode: {
      if (a.children.size() != b.children.size() ||
          a.has_names != b.has_names || (a.has_names && a.names != b.names)) {
        return diverge();
      }
      for (std::size_t j = 0; j < a.children.size(); ++j) {
        if (tree_diff_rec(a.children[j], b.children[j],
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

// Recover the RTree behind an external pointer handed back from R. Guards
// against foreign or plain SEXPs: without the type/class/NULL checks below, a
// non-RTree argument (e.g. a PJRTBuffer handle or a bare `1L`) would be
// reinterpreted as an `RTree*` and dereferenced, segfaulting the R session.
static const RTree& as_tree(SEXP handle) {
  if (TYPEOF(handle) != EXTPTRSXP || !Rf_inherits(handle, "RTree")) {
    Rcpp::stop("expected an `RTree` (as returned by `build_tree()`)");
  }
  if (R_ExternalPtrAddr(handle) == nullptr) {
    Rcpp::stop("`RTree` external pointer is NULL (already released?)");
  }
  Rcpp::XPtr<RTree> ptr(handle);
  return *ptr;
}

static void require_list(const RTree& tree, const char* what) {
  if (tree.kind != RTree::ListNode) {
    Rcpp::stop("%s requires a tree whose root is a list node", what);
  }
}

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP impl_tree_build(SEXP x) {
  using namespace rpjrt;
  auto tree = std::make_unique<RTree>();
  std::vector<SEXP> leaves;
  int counter = 0;
  flatten_rec(x, leaves, *tree, counter);
  return tree_xptr(tree.release());
}

// [[Rcpp::export]]
Rcpp::List impl_tree_flatten(SEXP x) {
  using namespace rpjrt;
  RTree tree;
  std::vector<SEXP> leaves;
  int counter = 0;
  flatten_rec(x, leaves, tree, counter);
  Rcpp::List out(leaves.size());
  for (std::size_t k = 0; k < leaves.size(); ++k) out[k] = leaves[k];
  return out;
}

// Build the tree and extract the leaves in a single traversal; returns
// `list(tree = <RTree>, leaves = <list>)`. map_tree()/pmap_tree() use this to
// avoid walking the same object twice (once to build, once to flatten).
// [[Rcpp::export]]
Rcpp::List impl_tree_build_flatten(SEXP x) {
  using namespace rpjrt;
  auto tree = std::make_unique<RTree>();
  std::vector<SEXP> leaves;
  int counter = 0;
  flatten_rec(x, leaves, *tree, counter);
  Rcpp::List out(leaves.size());
  for (std::size_t k = 0; k < leaves.size(); ++k) out[k] = leaves[k];
  return Rcpp::List::create(Rcpp::Named("tree") = tree_xptr(tree.release()),
                            Rcpp::Named("leaves") = out);
}

// [[Rcpp::export]]
SEXP impl_tree_unflatten(SEXP tree, Rcpp::List x) {
  using namespace rpjrt;
  const RTree& t = as_tree(tree);
  const int need = tree_size_rec(t);
  if (static_cast<int>(x.size()) != need) {
    Rcpp::stop("unflatten(): got %d leaves but the tree has %d",
               static_cast<int>(x.size()), need);
  }
  std::vector<SEXP> leaves(x.begin(), x.end());
  return unflatten_rec(t, leaves);
}

// [[Rcpp::export]]
int impl_tree_size(SEXP tree) {
  return rpjrt::tree_size_rec(rpjrt::as_tree(tree));
}

// [[Rcpp::export]]
bool impl_tree_equal(SEXP a, SEXP b) {
  return rpjrt::tree_eq(rpjrt::as_tree(a), rpjrt::as_tree(b));
}

// The std::size_t hash is returned as a decimal string: R has no native 64-bit
// integer, and the value is only ever compared for (in)equality.
// [[Rcpp::export]]
std::string impl_tree_hash(SEXP tree) {
  return std::to_string(rpjrt::tree_hash(rpjrt::as_tree(tree)));
}

// [[Rcpp::export]]
std::string impl_tree_kind(SEXP tree) {
  using namespace rpjrt;
  switch (as_tree(tree).kind) {
    case RTree::LeafNode:
      return "leaf";
    case RTree::ListNode:
      return "list";
    case RTree::NullNode:
      return "null";
  }
  return "";  // unreachable
}

// [[Rcpp::export]]
SEXP impl_tree_names(SEXP tree) {
  using namespace rpjrt;
  const RTree& n = as_tree(tree);
  if (n.kind != RTree::ListNode || !n.has_names) return R_NilValue;
  return Rcpp::wrap(n.names);
}

// [[Rcpp::export]]
Rcpp::CharacterVector impl_tree_child_kinds(SEXP tree) {
  using namespace rpjrt;
  const RTree& n = as_tree(tree);
  require_list(n, "child_kinds()");
  Rcpp::CharacterVector out(n.children.size());
  for (std::size_t k = 0; k < n.children.size(); ++k) {
    switch (n.children[k].kind) {
      case RTree::LeafNode:
        out[k] = "leaf";
        break;
      case RTree::ListNode:
        out[k] = "list";
        break;
      case RTree::NullNode:
        out[k] = "null";
        break;
    }
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::IntegerVector impl_tree_child_sizes(SEXP tree) {
  using namespace rpjrt;
  const RTree& n = as_tree(tree);
  require_list(n, "child_sizes()");
  Rcpp::IntegerVector out(n.children.size());
  for (std::size_t k = 0; k < n.children.size(); ++k) {
    out[k] = tree_size_rec(n.children[k]);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::CharacterVector impl_tree_flat_names(SEXP tree) {
  using namespace rpjrt;
  const RTree& n = as_tree(tree);
  require_list(n, "flat_names()");
  Rcpp::CharacterVector out(tree_size_rec(n));
  R_xlen_t pos = 0;
  for (std::size_t k = 0; k < n.children.size(); ++k) {
    const int sz = tree_size_rec(n.children[k]);
    const std::string nm = n.has_names ? n.names[k] : std::string();
    for (int j = 0; j < sz; ++j) out[pos++] = nm;
  }
  return out;
}

// [[Rcpp::export]]
std::string impl_tree_path(SEXP tree, int i) {
  using namespace rpjrt;
  std::string out;
  if (!tree_path_rec(as_tree(tree), i, "", out)) {
    Rcpp::stop("tree_path(): no leaf with index %d in the tree", i);
  }
  return out;
}

// [[Rcpp::export]]
SEXP impl_tree_filter_by_names(SEXP tree, Rcpp::CharacterVector names) {
  using namespace rpjrt;
  const RTree& n = as_tree(tree);
  require_list(n, "filter_by_names()");
  if (!n.has_names) Rcpp::stop("filter_by_names(): tree must have names");
  std::unordered_set<std::string> keep(names.size());
  for (R_xlen_t k = 0; k < names.size(); ++k) {
    keep.insert(std::string(names[k]));
  }
  auto out = std::make_unique<RTree>();
  out->kind = RTree::ListNode;
  out->has_names = true;
  int counter = 0;
  for (std::size_t k = 0; k < n.children.size(); ++k) {
    if (keep.count(n.names[k]) == 0) continue;
    out->children.push_back(n.children[k]);  // copy the kept subtree
    out->names.push_back(n.names[k]);
    reindex_rec(out->children.back(), counter);
  }
  return tree_xptr(out.release());
}

// [[Rcpp::export]]
SEXP impl_tree_concat(Rcpp::List children, SEXP names) {
  using namespace rpjrt;
  // RAII: as_tree() below rejects a non-RTree child by throwing, so `out` must
  // free itself if we never reach tree_xptr() (which transfers it to R's GC).
  auto out = std::make_unique<RTree>();
  out->kind = RTree::ListNode;
  out->children.reserve(children.size());
  int counter = 0;
  for (R_xlen_t k = 0; k < children.size(); ++k) {
    out->children.push_back(as_tree(children[k]));  // copy the child tree
    reindex_rec(out->children.back(), counter);
  }
  if (names != R_NilValue) {
    Rcpp::CharacterVector nms(names);
    if (nms.size() != children.size()) {
      Rcpp::stop("tree_concat(): names must match the number of children");
    }
    out->has_names = true;
    out->names.resize(nms.size());
    for (R_xlen_t k = 0; k < nms.size(); ++k) {
      out->names[k] = std::string(nms[k]);
    }
  }
  return tree_xptr(out.release());
}

// [[Rcpp::export]]
Rcpp::LogicalVector impl_tree_mask_from_names(SEXP tree,
                                              Rcpp::CharacterVector names) {
  using namespace rpjrt;
  const RTree& n = as_tree(tree);
  require_list(n, "mask_from_names()");
  std::unordered_set<std::string> marked(names.size());
  for (R_xlen_t k = 0; k < names.size(); ++k) {
    marked.insert(std::string(names[k]));
  }
  Rcpp::LogicalVector out(tree_size_rec(n));
  R_xlen_t pos = 0;
  for (std::size_t k = 0; k < n.children.size(); ++k) {
    const bool m = n.has_names && marked.count(n.names[k]) > 0;
    const int sz = tree_size_rec(n.children[k]);
    for (int j = 0; j < sz; ++j) out[pos++] = m;
  }
  return out;
}

// [[Rcpp::export]]
std::string impl_tree_repr(SEXP tree) {
  return rpjrt::tree_repr(rpjrt::as_tree(tree));
}

// [[Rcpp::export]]
SEXP impl_tree_diff(SEXP a, SEXP b) {
  using namespace rpjrt;
  std::string prefix;
  const RTree* sub_a = nullptr;
  const RTree* sub_b = nullptr;
  if (!tree_diff_rec(as_tree(a), as_tree(b), "", prefix, &sub_a, &sub_b)) {
    return R_NilValue;
  }
  return Rcpp::List::create(Rcpp::Named("prefix") = prefix,
                            Rcpp::Named("a") = tree_repr(*sub_a),
                            Rcpp::Named("b") = tree_repr(*sub_b));
}

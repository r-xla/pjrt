// Catch unit tests for the RTree flatten/unflatten/equality ops (src/tree.h).

#include <Rcpp.h>
#include <testthat.h>

#include <vector>

#include "tree.h"

using rpjrt::flatten_rec;
using rpjrt::RTree;
using rpjrt::tree_eq;
using rpjrt::tree_size_rec;
using rpjrt::unflatten_rec;

namespace {
// Flatten `x` into a fresh (tree, leaves) pair. `x` must outlive `leaves`,
// which holds borrowed SEXPs into it.
RTree build(SEXP x, std::vector<SEXP>& leaves) {
  RTree t;
  int counter = 0;
  flatten_rec(x, leaves, t, counter);
  return t;
}
}  // namespace

context("RTree flatten / unflatten") {
  test_that("a scalar is a single leaf") {
    Rcpp::RObject x = Rcpp::wrap(1.0);
    std::vector<SEXP> leaves;
    RTree t = build(x, leaves);
    expect_true(t.kind == RTree::LeafNode);
    expect_true(tree_size_rec(t) == 1);
    expect_true(leaves.size() == 1u);
  }

  test_that("NULL is a NullNode contributing no leaves") {
    std::vector<SEXP> leaves;
    RTree t = build(R_NilValue, leaves);
    expect_true(t.kind == RTree::NullNode);
    expect_true(tree_size_rec(t) == 0);
    expect_true(leaves.empty());
  }

  test_that("a nested list flattens in order and round-trips") {
    Rcpp::List x = Rcpp::List::create(
        Rcpp::wrap(1.0), Rcpp::List::create(Rcpp::wrap(2.0), Rcpp::wrap(3.0)));
    std::vector<SEXP> leaves;
    RTree t = build(x, leaves);
    expect_true(t.kind == RTree::ListNode);
    expect_true(tree_size_rec(t) == 3);
    expect_true(leaves.size() == 3u);

    // unflatten then re-flatten must yield a structurally equal tree
    Rcpp::RObject y = unflatten_rec(t, leaves);
    std::vector<SEXP> leaves2;
    RTree t2 = build(y, leaves2);
    expect_true(tree_eq(t, t2));
    expect_true(leaves2.size() == 3u);
  }

  test_that("list names are captured") {
    Rcpp::List x = Rcpp::List::create(Rcpp::Named("a") = Rcpp::wrap(1.0),
                                      Rcpp::Named("b") = Rcpp::wrap(2.0));
    std::vector<SEXP> leaves;
    RTree t = build(x, leaves);
    expect_true(t.has_names);
    expect_true(t.names.size() == 2u);
    expect_true(t.names[0] == "a");
    expect_true(t.names[1] == "b");
  }
}

context("RTree equality") {
  test_that("equality is structural: values don't matter, shape does") {
    Rcpp::List xa = Rcpp::List::create(Rcpp::wrap(1.0), Rcpp::wrap(2.0));
    Rcpp::List xb = Rcpp::List::create(Rcpp::wrap(9.0), Rcpp::wrap(9.0));
    Rcpp::List xc = Rcpp::List::create(Rcpp::wrap(1.0),
                                       Rcpp::List::create(Rcpp::wrap(2.0)));
    std::vector<SEXP> la, lb, lc;
    RTree a = build(xa, la);
    RTree b = build(xb, lb);
    RTree c = build(xc, lc);
    expect_true(tree_eq(a, b));
    expect_false(tree_eq(a, c));
  }

  test_that("an empty list differs from a named empty list") {
    Rcpp::List empty(0);
    Rcpp::List named_empty(0);
    named_empty.attr("names") = Rcpp::CharacterVector(0);
    std::vector<SEXP> l1, l2;
    RTree a = build(empty, l1);
    RTree b = build(named_empty, l2);
    expect_false(tree_eq(a, b));
  }

  test_that("differing names are not equal") {
    Rcpp::List xa = Rcpp::List::create(Rcpp::Named("a") = Rcpp::wrap(1.0));
    Rcpp::List xb = Rcpp::List::create(Rcpp::Named("b") = Rcpp::wrap(1.0));
    std::vector<SEXP> l1, l2;
    RTree a = build(xa, l1);
    RTree b = build(xb, l2);
    expect_false(tree_eq(a, b));
  }
}

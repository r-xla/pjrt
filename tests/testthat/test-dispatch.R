# Native eager-dispatch building blocks (src/dispatch.cpp).

test_that("native Node flattens and unflattens like a pytree", {
  st <- function(x) impl_dispatch_node_selftest(x)

  # flat list of leaves
  r <- st(list(1, 2, 3))
  expect_equal(r$n_leaves, 3L)
  expect_identical(r$rebuilt, list(1, 2, 3))

  # nested + names
  x <- list(a = 1, b = list(c = 2, d = 3))
  r <- st(x)
  expect_equal(r$n_leaves, 3L)
  expect_identical(r$rebuilt, x)

  # NULL is a structural (NullNode) element: no leaf, but preserved
  x <- list(p = 1, q = NULL, r = list(2, NULL, 3))
  r <- st(x)
  expect_equal(r$n_leaves, 3L) # the three numbers; the two NULLs contribute none
  expect_identical(r$rebuilt, x)

  # a classed object / atomic vector is a single leaf (not recursed)
  expect_equal(st(1:5)$n_leaves, 1L)
  expect_equal(st(list(matrix(1:4, 2, 2), "hi"))$n_leaves, 2L)

  # top-level NULL
  expect_equal(st(NULL)$n_leaves, 0L)
  expect_null(st(NULL)$rebuilt)
})

test_that("native Node hashing reflects structure", {
  h <- function(x) impl_dispatch_node_selftest(x)$hash
  expect_equal(h(list(1, 2)), h(list(9, 9))) # same structure -> same hash
  expect_false(h(list(1, 2)) == h(list(1, NULL))) # leaf vs null
  expect_false(h(list(1, 2)) == h(list(a = 1, b = 2))) # unnamed vs named
  expect_false(h(list(1, 2)) == h(list(1, 2, 3))) # arity
})

test_that("native LRU cache: recency, eviction, on_evict hook", {
  # capacity 2: set 1,2; touch 1; set 3 -> evicts 2 (LRU). on_evict fires once.
  # returns c(get1, has1, has2, has3, size, n_evicted)
  expect_identical(
    as.integer(impl_dispatch_lru_selftest()),
    c(10L, 1L, 0L, 1L, 2L, 1L)
  )
})

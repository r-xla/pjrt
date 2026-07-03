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

test_that("native cache key distinguishes dtype/shape/ambiguity/arity", {
  skip_if_not(plugins_downloaded())
  leaf <- function(buf, ambiguous = FALSE) list(buf, ambiguous)
  x <- pjrt_buffer(c(1, 2, 3), dtype = "f32")
  x2 <- pjrt_buffer(c(4, 5, 6), dtype = "f32") # same dtype/shape/device
  xi <- pjrt_buffer(c(1L, 2L, 3L), dtype = "i32") # different dtype
  xs <- pjrt_buffer(c(1, 2), dtype = "f32") # different shape

  H <- function(...) impl_dispatch_key_hash(list(...))
  E <- function(a, b) impl_dispatch_key_eq(a, b)

  # equal signatures (even from different buffers) hash and compare equal
  expect_equal(H(leaf(x), leaf(x)), H(leaf(x2), leaf(x2)))
  expect_true(E(list(leaf(x), leaf(x)), list(leaf(x2), leaf(x2))))

  # each distinguishing property changes both hash and equality
  expect_false(H(leaf(x)) == H(leaf(xi))) # dtype
  expect_false(E(list(leaf(x)), list(leaf(xi))))
  expect_false(H(leaf(x)) == H(leaf(xs))) # shape
  expect_false(E(list(leaf(x)), list(leaf(xs))))
  expect_false(H(leaf(x, TRUE)) == H(leaf(x))) # ambiguity
  expect_false(E(list(leaf(x, TRUE)), list(leaf(x))))
  expect_false(H(leaf(x), leaf(x)) == H(leaf(x))) # arity
  expect_false(E(list(leaf(x), leaf(x)), list(leaf(x))))
})

test_that("native static cache key compares values via identical() (env-sensitive)", {
  E <- function(a, b) impl_dispatch_static_key_eq(a, b)

  # same value -> equal; different value -> not equal
  expect_true(E(list(1L), list(1L)))
  expect_true(E(list("a", TRUE), list("a", TRUE)))
  expect_false(E(list(1L), list(2L)))
  expect_false(E(list("a"), list("b")))
  expect_false(E(list(1L, 2L), list(1L))) # arity

  # Two closures with identical body/formals but different environments must
  # NOT be merged: R's default identical() has ignore.environment = FALSE.
  mk <- function() function() NULL
  f1 <- mk()
  f2 <- mk()
  expect_true(E(list(f1), list(f1)))
  expect_false(E(list(f1), list(f2)))
  expect_false(identical(f1, f2)) # the R reference behavior being mirrored

  # ...but bytecode and srcref differences are ignored, like default identical().
  f3 <- compiler::cmpfun(f1)
  expect_true(identical(f1, f3))
  expect_true(E(list(f1), list(f3)))
  env <- new.env()
  g1 <- eval(parse(text = "function() NULL", keep.source = TRUE), envir = env)
  g2 <- eval(parse(text = "function() NULL", keep.source = TRUE), envir = env)
  expect_false(identical(attr(g1, "srcref"), attr(g2, "srcref"), ignore.srcref = FALSE))
  expect_true(identical(g1, g2))
  expect_true(E(list(g1), list(g2)))
})

test_that("native dispatcher caches, executes, and falls back", {
  skip_if_not(plugins_downloaded())
  # run() returns list(buffers, out_tree, ambiguous_out); take the first buffer.
  out <- function(res) as.numeric(tengen::as_array(await(res$buffers[[1]])))

  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec2 <- pjrt_compile(pjrt_program(src = add_src))

  n_miss <- 0L
  d <- impl_dispatch_create(10L, function(args) {
    n_miss <<- n_miss + 1L
    list(exec = exec2)
  }, character(0))

  # Leaves are xla AnvlArray-shaped lists (list(data=buffer, backend="xla", ...)).
  arr <- function(buf) {
    structure(list(data = buf, ambiguous = FALSE, backend = "xla"), class = "AnvlArray")
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- arr(pjrt_buffer(c(3, 4), dtype = "f32"))

  r1 <- impl_dispatch_run(d, list(x, y)) # miss -> compile -> execute
  r2 <- impl_dispatch_run(d, list(x, y)) # cache hit -> execute
  expect_equal(out(r1), c(4, 6))
  expect_equal(out(r2), c(4, 6))
  expect_equal(n_miss, 1L) # compiled once, then served from cache

  # non-dispatchable inputs return the sentinel (caller falls back)
  sentinel <- impl_dispatch_sentinel()
  expect_identical(impl_dispatch_run(d, list(x, 42L)), sentinel) # literal arg
  expect_identical(
    impl_dispatch_run(d, list(pjrt_buffer(c(1, 2), dtype = "f32"))),
    sentinel
  ) # bare buffer (not an AnvlArray) is not dispatchable
  quirk <- structure(list(data = x$data, ambiguous = FALSE, backend = "quickr"), class = "AnvlArray")
  expect_identical(impl_dispatch_run(d, list(quirk, quirk)), sentinel)

  # GC-correct: many dispatches with periodic gc(), then teardown
  for (i in 1:300) {
    r <- impl_dispatch_run(d, list(x, y))
    if (i %% 100 == 0) {
      gc()
    }
    expect_equal(out(r), c(4, 6))
  }
  rm(d)
  gc()
  expect_true(TRUE) # reached teardown without crashing
})

test_that("dispatcher with static names still dispatches a pure-dynamic call", {
  skip_if_not(plugins_downloaded())
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec2 <- pjrt_compile(pjrt_program(src = add_src))
  arr <- function(buf) {
    structure(list(data = buf, ambiguous = FALSE, backend = "xla"), class = "AnvlArray")
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- arr(pjrt_buffer(c(3, 4), dtype = "f32"))

  # static name "flag" declared, but this call has no such arg -> all dynamic.
  d <- pjrt_dispatcher(10L, function(args) list(exec = exec2), static = "flag")
  res <- pjrt_dispatch(d, list(x = x, y = y))
  expect_false(identical(res, pjrt_dispatch_sentinel()))
  expect_equal(as.numeric(tengen::as_array(await(res$buffers[[1]]))), c(4, 6))
})

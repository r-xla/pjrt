# The dispatcher exercised the way it is meant to be used: behind anvl's
# jit(). test-dispatch.R drives the building blocks directly with hand-built
# array leaves; these tests drive the whole loop -- anvl traces and compiles in
# the miss callback, the dispatcher caches, executes, and wraps -- and assert
# on what a user of jit() observes. anvl sits in Suggests, so this file is a
# no-op wherever anvl is not installed.

skip_if_not_installed("anvl")
skip_if_not(plugins_downloaded())

# The dispatcher a jitted function dispatches through (anvl stores it in the
# closure environment of the function's fast entry).
jit_dispatcher <- function(f) {
  environment(attr(f, "jit_run_args"))$dispatcher
}

test_that("jit() dispatches, caches, and returns wrapped arrays", {
  f <- anvl::jit(function(x, y) x + y)
  x <- anvl::nv_array(c(1, 2, 3), dtype = "f32")
  y <- anvl::nv_array(c(10, 20, 30), dtype = "f32")

  r1 <- f(x, y)
  # The result is a fully wrapped array: the dispatcher built it natively.
  expect_s3_class(r1, "AnvlArray")
  expect_identical(as.character(tengen::dtype(r1)), "f32")
  expect_identical(tengen::shape(r1), 3L)
  expect_s3_class(r1$data, "PJRTBuffer")
  expect_identical(r1$backend, "xla")
  expect_equal(as.numeric(tengen::as_array(r1)), c(11, 22, 33))

  # A second call of the same signature is a cache hit...
  r2 <- f(x, y)
  expect_equal(as.numeric(tengen::as_array(r2)), c(11, 22, 33))
  d <- jit_dispatcher(f)
  expect_s3_class(d, "Dispatcher")
  expect_equal(dispatcher_size(d), 1L)

  # ...an output feeds back in as an input without re-compiling...
  r3 <- f(r1, y)
  expect_equal(as.numeric(tengen::as_array(r3)), c(21, 42, 63))
  expect_equal(dispatcher_size(d), 1L)

  # ...and a new shape is a new cache entry.
  invisible(f(anvl::nv_array(1, dtype = "f32"), anvl::nv_array(2, dtype = "f32")))
  expect_equal(dispatcher_size(d), 2L)
})

test_that("jit() preserves nested output structure and names", {
  f <- anvl::jit(function(x) {
    list(sum = x + x, nested = list(sq = x * x))
  })
  x <- anvl::nv_array(c(2, 3), dtype = "f32")
  res <- f(x)
  expect_named(res, c("sum", "nested"))
  expect_named(res$nested, "sq")
  expect_equal(as.numeric(tengen::as_array(res$sum)), c(4, 6))
  expect_equal(as.numeric(tengen::as_array(res$nested$sq)), c(4, 9))
})

test_that("jit() with static args compiles per static value", {
  f <- anvl::jit(
    function(x, flag) if (flag) x + 1 else x * 2,
    static = "flag"
  )
  x <- anvl::nv_array(3, dtype = "f32")
  expect_equal(as.numeric(tengen::as_array(f(x, TRUE))), 4)
  expect_equal(as.numeric(tengen::as_array(f(x, FALSE))), 6)
  expect_equal(as.numeric(tengen::as_array(f(x, TRUE))), 4)
  expect_equal(dispatcher_size(jit_dispatcher(f)), 2L)
})

test_that("jit() accepts bare literals; a buffer of the same aval shares the entry", {
  f <- anvl::jit(function(x, y) x + y)
  x <- anvl::nv_array(c(1, 2), dtype = "f32")
  expect_equal(as.numeric(tengen::as_array(f(x, 5))), c(6, 7))
  # An ambiguous rank-0 f32 array has the literal's aval: same cache entry.
  # (nv_scalar(5) would not share it -- it commits ambiguous = FALSE.)
  amb <- anvl::nv_scalar(5, ambiguous = TRUE)
  expect_equal(as.numeric(tengen::as_array(f(x, amb))), c(6, 7))
  expect_equal(dispatcher_size(jit_dispatcher(f)), 1L)
})

test_that("the ambiguity of an output survives the native wrap", {
  # x + 1 keeps a committed f32's dtype: the output is unambiguous. A literal
  # alone stays ambiguous. Both bits are stamped by the dispatcher's wrap.
  f <- anvl::jit(function(x) x + 1)
  committed <- f(anvl::nv_array(1, dtype = "f32"))
  expect_false(committed$ambiguous)
  g <- anvl::jit(function(x) x + 1)
  floating <- g(2)
  expect_true(floating$ambiguous)
})

test_that("invalid jit() inputs are rejected natively, naming the argument", {
  f <- anvl::jit(function(x, y) x + y)
  x <- anvl::nv_array(c(1, 2), dtype = "f32")
  d_before <- jit_dispatcher(f)
  expect_error(f(x, "nope"), "invalid input `y`")
  expect_error(f(x, c(1, 2, 3)), "invalid input `y`.*<numeric> of length 3")
  expect_equal(dispatcher_size(d_before), 0L) # rejected before any compile
})

test_that("jit(device = ) fixes the entry's device and moves inputs", {
  f <- anvl::jit(function(x) x + 1, device = "cpu:0")
  x <- anvl::nv_array(c(1, 2), dtype = "f32")
  res <- f(x)
  expect_equal(as.numeric(tengen::as_array(res)), c(2, 3))
  # Devices are interned, so the wrapped output carries the very object.
  expect_identical(tengen::device(res), pjrt_device("cpu:0"))
})

test_that("a jitted function with no array inputs keys on the default device", {
  f <- anvl::jit(function(n) n + 1)
  expect_equal(as.numeric(tengen::as_array(f(41))), 42)
  expect_equal(as.numeric(tengen::as_array(f(41))), 42)
  expect_equal(dispatcher_size(jit_dispatcher(f)), 1L)
})

test_that("the quickr backend dispatches through the closure engine", {
  skip_if_not_installed("quickr")
  anvl::with_backend("quickr", {
    f <- anvl::jit(function(x, y) x + y)
    x <- anvl::nv_array(c(1, 2), dtype = "f64")
    y <- anvl::nv_array(c(10, 20), dtype = "f64")
    r1 <- f(x, y)
    expect_s3_class(r1, "AnvlArray")
    expect_identical(r1$backend, "quickr")
    expect_equal(as.numeric(tengen::as_array(r1)), c(11, 22))
    invisible(f(x, y))
    expect_equal(dispatcher_size(jit_dispatcher(f)), 1L)
  })
})

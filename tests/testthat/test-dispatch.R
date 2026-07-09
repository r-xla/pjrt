# Native eager-dispatch building blocks (src/dispatch.cpp).

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
  d <- impl_dispatch_create(
    10L,
    function(args) {
      n_miss <<- n_miss + 1L
      list(exec = exec2)
    },
    character(0),
    "pjrt",
    FALSE
  )

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
  d <- dispatcher(10L, function(args) list(exec = exec2), static = "flag")
  res <- dispatch(d, list(x = x, y = y))
  expect_false(identical(res, dispatch_sentinel()))
  expect_equal(as.numeric(tengen::as_array(await(res$buffers[[1]]))), c(4, 6))
})

test_that("native dispatcher keys static args by value and excludes them from execute", {
  skip_if_not(plugins_downloaded())
  # Identity executable over ONE f32 input. If a static leaf were ever sent to
  # execute, the input arity (2) would mismatch @main (1) and execution would
  # error -- so reaching the asserts proves the static leaf was excluded.
  id_src <- 'func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    "func.return"(%x): (tensor<2xf32>) -> ()
  }'
  exec_id <- pjrt_compile(pjrt_program(src = id_src))

  seen <- list()
  d <- impl_dispatch_create(
    10L,
    function(args) {
      seen[[length(seen) + 1L]] <<- args$flag
      list(exec = exec_id)
    },
    "flag",
    "pjrt",
    FALSE
  )

  arr <- function(buf) {
    structure(list(data = buf, ambiguous = FALSE, backend = "xla"), class = "AnvlArray")
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))
  out <- function(res) as.numeric(tengen::as_array(await(res$buffers[[1]])))

  r1 <- impl_dispatch_run(d, list(x = x, flag = TRUE)) # miss (flag = TRUE)
  r2 <- impl_dispatch_run(d, list(x = x, flag = FALSE)) # miss (flag = FALSE)
  r3 <- impl_dispatch_run(d, list(x = x, flag = TRUE)) # hit  (flag = TRUE)

  expect_equal(out(r1), c(1, 2))
  expect_equal(out(r2), c(1, 2))
  expect_equal(out(r3), c(1, 2))
  expect_equal(length(seen), 2L) # two distinct static values compiled
  expect_identical(seen, list(TRUE, FALSE))
  expect_equal(impl_dispatch_size(d), 2L)

  # GC-correct with static keys: the preserved static values survive gc().
  for (i in 1:50) {
    r <- impl_dispatch_run(d, list(x = x, flag = TRUE))
    if (i %% 25 == 0) {
      gc()
    }
    expect_equal(out(r), c(1, 2))
  }
  expect_equal(impl_dispatch_size(d), 2L)
  rm(d)
  gc()

  # An all-static / zero-dynamic call dispatches natively too: the entry has
  # no dynamic inputs and its device comes from the compile callback.
  zero_src <- 'func.func @main() -> tensor<2xf32> {
    %0 = "stablehlo.constant"() { value = dense<[7.0, 8.0]> : tensor<2xf32> } : () -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec_zero <- pjrt_compile(pjrt_program(src = zero_src))
  n2 <- 0L
  d2 <- impl_dispatch_create(
    10L,
    function(args) {
      n2 <<- n2 + 1L
      list(exec = exec_zero)
    },
    "flag",
    "pjrt",
    FALSE
  )
  z1 <- impl_dispatch_run(d2, list(flag = TRUE))
  z2 <- impl_dispatch_run(d2, list(flag = TRUE))
  expect_equal(out(z1), c(7, 8))
  expect_equal(out(z2), c(7, 8))
  expect_equal(n2, 1L)
})

test_that("native dispatcher uploads bare R literals and arrays (pjrt engine)", {
  skip_if_not(plugins_downloaded())
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<f32>) -> tensor<2xf32> {
    %0 = "stablehlo.broadcast_in_dim"(%y) { broadcast_dimensions = array<i64> } : (tensor<f32>) -> tensor<2xf32>
    %1 = "stablehlo.add"(%x, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%1): (tensor<2xf32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = add_src))
  client <- pjrt_client("cpu")
  device <- pjrt_device("cpu")
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(args) {
      n_miss <<- n_miss + 1L
      list(exec = exec, client = client, device = device)
    },
    character(0),
    "pjrt",
    FALSE
  )
  arr <- function(buf) {
    structure(list(data = buf, ambiguous = FALSE, backend = "xla"), class = "AnvlArray")
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))
  out <- function(res) as.numeric(tengen::as_array(await(res$buffers[[1]])))

  # a bare double literal is uploaded as a rank-0 f32 buffer per call
  r1 <- impl_dispatch_run(d, list(x, 10))
  r2 <- impl_dispatch_run(d, list(x, 20)) # same signature -> cache hit
  expect_equal(out(r1), c(11, 12))
  expect_equal(out(r2), c(21, 22))
  expect_equal(n_miss, 1L)

  # a literal missing client/device in the entry is a clear error
  d_bad <- impl_dispatch_create(
    10L,
    function(args) list(exec = exec),
    character(0),
    "pjrt",
    FALSE
  )
  expect_error(impl_dispatch_run(d_bad, list(x, 10)), "client")

  # an R array leaf uploads column-major like pjrt_buffer()
  id2_src <- 'func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    "func.return"(%x): (tensor<2x2xf32>) -> ()
  }'
  exec_id2 <- pjrt_compile(pjrt_program(src = id2_src))
  d2 <- impl_dispatch_create(
    10L,
    function(args) list(exec = exec_id2, client = client, device = device),
    character(0),
    "pjrt",
    FALSE
  )
  m <- matrix(c(1, 2, 3, 4), nrow = 2)
  r3 <- impl_dispatch_run(d2, list(m))
  expect_equal(tengen::as_array(await(r3$buffers[[1]])), m)
})

test_that("native dispatcher moves buffers to the target device (move_inputs)", {
  skip_if_not(plugins_downloaded())
  id_src <- 'func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    "func.return"(%x): (tensor<2xf32>) -> ()
  }'
  client <- pjrt_client("cpu")
  target <- pjrt_device("cpu:0")
  exec_id <- pjrt_compile(pjrt_program(src = id_src), device = target)
  d <- impl_dispatch_create(
    10L,
    function(args) list(exec = exec_id, client = client, device = target),
    character(0),
    "pjrt",
    TRUE
  )
  arr <- function(buf) {
    structure(list(data = buf, ambiguous = FALSE, backend = "xla"), class = "AnvlArray")
  }
  out <- function(res) as.numeric(tengen::as_array(await(res$buffers[[1]])))

  # same-device input passes through
  x0 <- arr(pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0"))
  expect_equal(out(impl_dispatch_run(d, list(x0))), c(1, 2))

  # an input on another device is copied to the target (needs >= 2 devices)
  cpus <- devices(client)
  skip_if(length(cpus) < 2L, "needs a second cpu device")
  x1 <- arr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:1"))
  res <- impl_dispatch_run(d, list(x1))
  expect_equal(out(res), c(3, 4))
  expect_equal(impl_dispatch_size(d), 1L) # device is not part of the key
})

test_that("native dispatcher sentinels only on a device conflict (infer policy)", {
  skip_if_not(plugins_downloaded())
  cpus <- devices(pjrt_client("cpu"))
  skip_if(length(cpus) < 2L, "needs a second cpu device")
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec2 <- pjrt_compile(pjrt_program(src = add_src))
  d <- impl_dispatch_create(
    10L,
    function(args) list(exec = exec2),
    character(0),
    "pjrt",
    FALSE
  )
  arr <- function(buf) {
    structure(list(data = buf, ambiguous = FALSE, backend = "xla"), class = "AnvlArray")
  }
  x0 <- arr(pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0"))
  y1 <- arr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:1"))
  expect_identical(impl_dispatch_run(d, list(x0, y1)), dispatch_sentinel())
})

test_that("closure engine dispatches through a compiled R closure", {
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(args) {
      n_miss <<- n_miss + 1L
      # `r_fun` receives the flat leaves: quickr/plain AnvlArrays contribute
      # their $data, everything else (statics included) passes through.
      list(r_fun = function(flat) list(sum = flat[[1]] + flat[[2]], flag = flat[[3]]))
    },
    "flag",
    "closure",
    FALSE
  )
  qarr <- function(v) {
    structure(
      list(
        data = v,
        dtype = tengen::as_dtype("f64"),
        shape = as.integer(length(v)),
        ambiguous = FALSE,
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }
  a <- qarr(c(1, 2))
  b <- qarr(c(10, 20))
  r1 <- impl_dispatch_run(d, list(x = a, y = b, flag = TRUE))
  expect_identical(r1$value, list(sum = c(11, 22), flag = TRUE))
  r2 <- impl_dispatch_run(d, list(x = a, y = b, flag = TRUE))
  expect_identical(r2$value, list(sum = c(11, 22), flag = TRUE))
  expect_equal(n_miss, 1L) # same signature -> hit
  expect_equal(impl_dispatch_size(d), 1L)

  # a different dtype or shape is a new signature
  qi <- structure(
    list(
      data = c(1L, 2L),
      dtype = tengen::as_dtype("i32"),
      shape = 2L,
      ambiguous = FALSE,
      backend = "quickr"
    ),
    class = "AnvlArray"
  )
  invisible(impl_dispatch_run(d, list(x = qi, y = b, flag = TRUE)))
  expect_equal(n_miss, 2L)

  # a bare R literal passes through to the closure as-is
  d2 <- impl_dispatch_create(
    10L,
    function(args) list(r_fun = function(flat) flat[[1]] * flat[[2]]),
    character(0),
    "closure",
    FALSE
  )
  expect_identical(impl_dispatch_run(d2, list(a, 3))$value, c(3, 6))
})

# Native eager-dispatch building blocks (src/dispatch.cpp, dispatch_engine.cpp).

# The `default_device` resolvers a dispatcher needs when a call has no array
# input to read a device from. Devices are canonicalized per dispatcher
# (identity first, identical() fallback), so interning -- one object per
# device, as pjrt does for PJRTDevice and `test_device()` mimics here -- is
# the fast path rather than a requirement.
test_xla_device <- function() pjrt_device("cpu:0")
the_test_devices <- new.env(parent = emptyenv())
test_device <- function(id = "cpu") {
  dev <- the_test_devices[[id]]
  if (is.null(dev)) {
    dev <- structure(list(device = id), class = "QuickrDevice")
    the_test_devices[[id]] <- dev
  }
  dev
}
test_quickr_device <- function() test_device("cpu")

# One output aval, as the compile callback declares it: the dtype/shape/
# ambiguous pjrt stamps on that output's wrapper. Same shape as the input avals
# the callback receives in `info$avals`.
oav <- function(dtype = "f32", shape = 2L, ambiguous = FALSE) {
  list(dtype = dtype, shape = as.integer(shape), ambiguous = ambiguous)
}

# The pjrt engine's full compile-callback contract for a single-output
# executable: exec, client + device (uploads, moves, phantoms, and the wrapped
# outputs' $device), a leaf out_tree (a single un-nested output), and the
# out_avals the outputs are wrapped from. The default is the tests' most common
# program: one f32 output of shape 2.
xla_entry <- function(
  exec,
  ...,
  out_tree = build_tree(0),
  out_avals = list(oav()),
  device = pjrt_device("cpu:0")
) {
  list(
    exec = exec,
    client = pjrt_client("cpu"),
    device = device,
    out_tree = out_tree,
    out_avals = out_avals,
    ...
  )
}

# With a leaf out_tree, dispatch() returns the single output as one wrapped
# array: an "AnvlArray" list whose $data is the output buffer.
out <- function(res) as.numeric(tengen::as_array(await(res$data)))

# The cache key's structure -- avals, device token, the kArray/kRData merge,
# and the hash/equality contract -- is tested in C++, in src/test-dispatch.cpp,
# where a device token can be fabricated and a leaf built directly. What is
# tested here is what only R can express: how the key treats real R values as
# static arguments. Each test drives the actual dispatcher and counts compiles,
# because a cache entry per distinct key is the behaviour that matters, not a
# hash the caller never sees.

# A dispatcher over one static argument `s`; `n()` reports how many times the
# compile callback ran, i.e. how many distinct keys were seen.
static_dispatcher <- function() {
  n <- 0L
  d <- impl_dispatch_create(
    50L,
    function(info) {
      n <<- n + 1L
      list(r_fun = function(flat) list(v = 1))
    },
    "s",
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  list(
    compiles_for = function(...) {
      for (v in list(...)) {
        invisible(impl_dispatch_run(d, list(s = v)))
      }
      n
    }
  )
}

# TRUE iff the two static values are one cache key.
same_key <- function(a, b) static_dispatcher()$compiles_for(a, b) == 1L

test_that("static args are keyed with identical(), environment included", {
  expect_true(same_key(1L, 1L))
  expect_true(same_key("a", "a"))
  expect_false(same_key(1L, 2L))
  expect_false(same_key("a", "b"))

  # Two closures with identical body/formals but different environments must
  # NOT be merged: R's default identical() has ignore.environment = FALSE.
  mk <- function() function() NULL
  f1 <- mk()
  f2 <- mk()
  expect_false(identical(f1, f2)) # the R reference behaviour being mirrored
  expect_true(same_key(f1, f1))
  expect_false(same_key(f1, f2))

  # ...but bytecode and srcref differences are ignored, like default identical().
  f3 <- compiler::cmpfun(f1)
  expect_true(identical(f1, f3))
  expect_true(same_key(f1, f3))

  env <- new.env()
  g1 <- eval(parse(text = "function() NULL", keep.source = TRUE), envir = env)
  g2 <- eval(parse(text = "function() NULL", keep.source = TRUE), envir = env)
  expect_false(identical(attr(g1, "srcref"), attr(g2, "srcref"), ignore.srcref = FALSE))
  expect_true(identical(g1, g2))
  expect_true(same_key(g1, g2))
})

test_that("static args that identical() joins share one cache entry", {
  # The contract: keys the dispatcher calls equal MUST hash alike, or the map
  # stores two entries for one key. Two compiles here would mean a hash that
  # disagrees with the equality.
  utf8 <- "é"
  latin1 <- iconv(utf8, "UTF-8", "latin1")
  expect_true(same_key(utf8, latin1)) # same string, different bytes
  expect_true(same_key(1.5, 1.5))
  expect_true(same_key(NaN, NaN))
  expect_true(same_key(1:3, c(1L, 2L, 3L))) # ALTREP compact seq vs materialized
})

test_that("static numbers are keyed bitwise: +0 and -0 are distinct", {
  # A literal `-0` is constant-folded to `+0` by R's byte compiler, so build it
  # from a variable -- otherwise this would quietly compare 0 against 0.
  zero <- 0
  neg_zero <- -1 * zero
  expect_identical(sprintf("%a", neg_zero), "-0x0p+0")
  expect_false(same_key(zero, neg_zero))
})

test_that("distinct static values never merge", {
  expect_false(same_key(1L, 1)) # type is folded before the contents
  expect_false(same_key(TRUE, FALSE))
  expect_false(same_key(NaN, NA_real_))
  expect_false(same_key(c(1, 2), c(2, 1)))
  expect_false(same_key(1 + 2i, 1 + 3i))
  expect_false(same_key(as.raw(1), as.raw(2)))
  expect_false(same_key(NA_character_, "NA"))
})

test_that("native dispatcher caches, executes, and wraps the outputs", {
  skip_if_not(plugins_downloaded())
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec2 <- pjrt_compile(pjrt_program(src = add_src))

  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      xla_entry(exec2)
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )

  # Leaves are xla AnvlArray-shaped lists (list(data=buffer, backend="xla", ...)).
  arr <- function(buf) {
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- arr(pjrt_buffer(c(3, 4), dtype = "f32"))

  r1 <- impl_dispatch_run(d, list(x, y)) # miss -> compile -> execute
  r2 <- impl_dispatch_run(d, list(x, y)) # cache hit -> execute
  expect_equal(out(r1), c(4, 6))
  expect_equal(out(r2), c(4, 6))
  expect_equal(n_miss, 1L) # compiled once, then served from cache

  # The output is wrapped natively: the very array-leaf layout the dispatcher
  # itself accepts as an input, stamped with the entry's device and the
  # dispatcher's backend.
  expect_s3_class(r1, "AnvlArray")
  expect_named(r1, c("data", "dtype", "shape", "device", "ambiguous", "backend"))
  expect_s3_class(r1$data, "PJRTBuffer")
  expect_identical(as.character(r1$dtype), "f32")
  expect_identical(r1$shape, 2L)
  expect_s3_class(r1$device, "PJRTDevice")
  expect_identical(r1$ambiguous, FALSE)
  expect_identical(r1$backend, "xla")
  # ...so an output feeds straight back in as an input.
  expect_equal(out(impl_dispatch_run(d, list(r1, y))), c(7, 10))

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

test_that("outputs are re-nested by out_tree; out_avals reach the wrap", {
  skip_if_not(plugins_downloaded())
  two_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = "stablehlo.multiply"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0, %1): (tensor<2xf32>, tensor<2xf32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = two_src))
  d <- impl_dispatch_create(
    10L,
    function(info) {
      xla_entry(
        exec,
        out_tree = build_tree(list(sum = 0, rest = list(prod = 0))),
        out_avals = list(oav(ambiguous = TRUE), oav())
      )
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  arr <- function(v) {
    buf <- pjrt_buffer(v, dtype = "f32")
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  res <- impl_dispatch_run(d, list(arr(c(1, 2)), arr(c(3, 4))))
  expect_named(res, c("sum", "rest"))
  expect_named(res$rest, "prod")
  expect_equal(out(res$sum), c(4, 6))
  expect_equal(out(res$rest$prod), c(3, 8))
  # every wrapped field comes from the declared aval, not from the buffer
  expect_identical(res$sum$ambiguous, TRUE)
  expect_identical(res$rest$prod$ambiguous, FALSE)
  expect_identical(res$sum$shape, 2L)
  expect_identical(res$sum$dtype, tengen::as_dtype("f32"))

  # An out_tree whose leaf count disagrees with the executable's actual output
  # count is the one half of the callback's claim pjrt can still settle, and it
  # does -- on execution, against the real outputs.
  d_bad <- impl_dispatch_create(
    10L,
    function(info) {
      xla_entry(
        exec,
        out_tree = build_tree(list(0, 0, 0)),
        out_avals = list(oav(), oav(), oav())
      )
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  expect_error(
    impl_dispatch_run(d_bad, list(arr(c(1, 2)), arr(c(3, 4)))),
    "out_tree has 3 leaves but the executable returned 2 outputs"
  )
  # An out_avals that disagrees with out_tree is caught at compile time,
  # before the entry is ever cached.
  d_bad2 <- impl_dispatch_create(
    10L,
    function(info) {
      xla_entry(exec, out_tree = build_tree(list(0, 0)), out_avals = list(oav()))
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  expect_error(
    impl_dispatch_run(d_bad2, list(arr(c(1, 2)), arr(c(3, 4)))),
    "out_avals has length 1 but out_tree has 2 leaves"
  )
})

test_that("dispatcher with static names still dispatches a pure-dynamic call", {
  skip_if_not(plugins_downloaded())
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec2 <- pjrt_compile(pjrt_program(src = add_src))
  arr <- function(buf) {
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- arr(pjrt_buffer(c(3, 4), dtype = "f32"))

  # static name "flag" declared, but this call has no such arg -> all dynamic.
  d <- dispatcher(
    10L,
    function(info) xla_entry(exec2),
    static = "flag",
    default_device = test_xla_device
  )
  res <- dispatch(d, list(x = x, y = y))
  expect_equal(out(res), c(4, 6))
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
    function(info) {
      seen[[length(seen) + 1L]] <<- info$args$flag
      xla_entry(exec_id)
    },
    "flag",
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )

  arr <- function(buf) {
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))

  r1 <- impl_dispatch_run(d, list(x = x, flag = TRUE)) # miss (flag = TRUE)
  r2 <- impl_dispatch_run(d, list(x = x, flag = FALSE)) # miss (flag = FALSE)
  r3 <- impl_dispatch_run(d, list(x = x, flag = TRUE)) # hit  (flag = TRUE)

  expect_equal(out(r1), c(1, 2))
  expect_equal(out(r2), c(1, 2))
  expect_equal(out(r3), c(1, 2))
  expect_equal(length(seen), 2L) # two distinct static values compiled
  expect_identical(seen, list(TRUE, FALSE))
  expect_equal(impl_dispatcher_size(d), 2L)

  # GC-correct with static keys: the preserved static values survive gc().
  for (i in 1:50) {
    r <- impl_dispatch_run(d, list(x = x, flag = TRUE))
    if (i %% 25 == 0) {
      gc()
    }
    expect_equal(out(r), c(1, 2))
  }
  expect_equal(impl_dispatcher_size(d), 2L)
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
    function(info) {
      n2 <<- n2 + 1L
      xla_entry(exec_zero)
    },
    "flag",
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
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
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      xla_entry(exec)
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  arr <- function(buf) {
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  x <- arr(pjrt_buffer(c(1, 2), dtype = "f32"))

  # a bare double literal is uploaded as a rank-0 f32 buffer per call
  r1 <- impl_dispatch_run(d, list(x, 10))
  r2 <- impl_dispatch_run(d, list(x, 20)) # same signature -> cache hit
  expect_equal(out(r1), c(11, 12))
  expect_equal(out(r2), c(21, 22))
  expect_equal(n_miss, 1L)

  # The pjrt engine's compile-callback contract is validated up front: a
  # missing client (needed for uploads, phantoms, and the wrap's device) is a
  # clear error, not a crash at input-assembly time.
  d_bad <- impl_dispatch_create(
    10L,
    function(info) list(exec = exec, device = pjrt_device("cpu:0"), out_tree = build_tree(0)),
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  expect_error(impl_dispatch_run(d_bad, list(x, 10)), "must return `client`")

  # So is a const_arrays element that is not a PJRTBuffer: execute would
  # reinterpret the external pointer blindly and segfault, so it must be
  # rejected when the entry is built (here: the exec itself, a plausible slip).
  d_bad2 <- impl_dispatch_create(
    10L,
    function(info) xla_entry(exec, const_arrays = list(exec)),
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  expect_error(
    impl_dispatch_run(d_bad2, list(x, 10)),
    "const_arrays\\[\\[1\\]\\]` must be a PJRTBuffer"
  )

  # an R array leaf uploads column-major like pjrt_buffer()
  id2_src <- 'func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    "func.return"(%x): (tensor<2x2xf32>) -> ()
  }'
  exec_id2 <- pjrt_compile(pjrt_program(src = id2_src))
  d2 <- impl_dispatch_create(
    10L,
    function(info) xla_entry(exec_id2, out_avals = list(oav(shape = c(2L, 2L)))),
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  m <- matrix(c(1, 2, 3, 4), nrow = 2)
  r3 <- impl_dispatch_run(d2, list(m))
  expect_equal(tengen::as_array(await(r3$data)), m)
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
    function(info) xla_entry(exec_id, device = target),
    character(0),
    "pjrt",
    "xla",
    TRUE,
    test_xla_device
  )
  arr <- function(buf) {
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }

  # same-device input passes through
  x0 <- arr(pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0"))
  expect_equal(out(impl_dispatch_run(d, list(x0))), c(1, 2))

  # an input on another device is copied to the target (needs >= 2 devices)
  cpus <- devices(client)
  skip_if(length(cpus) < 2L, "needs a second cpu device")
  x1 <- arr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:1"))
  res <- impl_dispatch_run(d, list(x1))
  expect_equal(out(res), c(3, 4))
  expect_equal(impl_dispatcher_size(d), 1L) # device is not part of the key
})

test_that("native dispatcher rejects a device conflict (infer policy), naming the input", {
  skip_if_not(plugins_downloaded())
  cpus <- devices(pjrt_client("cpu"))
  skip_if(length(cpus) < 2L, "needs a second cpu device")
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec2 <- pjrt_compile(pjrt_program(src = add_src))
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      xla_entry(exec2)
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  arr <- function(buf) {
    structure(
      list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  x0 <- arr(pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0"))
  y1 <- arr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:1"))

  # The conflict is caught natively, before the cache is probed: the compile
  # callback is never reached, and the message names the offending argument.
  expect_error(
    impl_dispatch_run(d, list(x = x0, y = y1)),
    "invalid input `y`.*different device"
  )
  expect_equal(n_miss, 0L)

  # Under the move policy a target device is fixed, so the same call dispatches.
  d_move <- impl_dispatch_create(
    10L,
    function(info) xla_entry(exec2),
    character(0),
    "pjrt",
    "xla",
    TRUE,
    test_xla_device
  )
  expect_no_error(impl_dispatch_run(d_move, list(x = x0, y = y1)))
})

test_that("closure engine dispatches through a compiled R closure", {
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      # `r_fun` receives the call's inputs: "quickr" AnvlArrays contribute
      # their $data, bare R data passes through. The static `flag` is not among
      # them -- it is a constant of this very closure, which closes over the
      # value it was compiled for. Its return value is the dispatch result.
      flag <- info$args$flag
      list(r_fun = function(flat) {
        list(n_inputs = length(flat), sum = flat[[1]] + flat[[2]], flag = flag)
      })
    },
    "flag",
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  qarr <- function(v) {
    structure(
      list(
        data = v,
        dtype = tengen::as_dtype("f64"),
        shape = as.integer(length(v)),
        ambiguous = FALSE,
        device = test_quickr_device(),
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }
  a <- qarr(c(1, 2))
  b <- qarr(c(10, 20))
  # Only the two arrays are inputs; the static `flag` never reaches `r_fun`.
  r1 <- impl_dispatch_run(d, list(x = a, y = b, flag = TRUE))
  expect_identical(r1, list(n_inputs = 2L, sum = c(11, 22), flag = TRUE))
  r2 <- impl_dispatch_run(d, list(x = a, y = b, flag = TRUE))
  expect_identical(r2, list(n_inputs = 2L, sum = c(11, 22), flag = TRUE))
  expect_equal(n_miss, 1L) # same signature -> hit
  expect_equal(impl_dispatcher_size(d), 1L)

  # a different dtype or shape is a new signature
  qi <- structure(
    list(
      data = c(1L, 2L),
      dtype = tengen::as_dtype("i32"),
      shape = 2L,
      ambiguous = FALSE,
      device = test_quickr_device(),
      backend = "quickr"
    ),
    class = "AnvlArray"
  )
  invisible(impl_dispatch_run(d, list(x = qi, y = b, flag = TRUE)))
  expect_equal(n_miss, 2L)

  # a bare R literal passes through to the closure as-is
  d2 <- impl_dispatch_create(
    10L,
    function(info) list(r_fun = function(flat) flat[[1]] * flat[[2]]),
    character(0),
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  expect_identical(impl_dispatch_run(d2, list(a, 3)), c(3, 6))
})

test_that("the closure engine serves a backend pjrt has never heard of", {
  # The dispatcher's `backend` is a parameter, not a hardcoded pair: a new
  # backend brings interned devices, AnvlArrays tagged with its own name, and
  # a compile callback -- and dispatches natively with no C++ of its own.
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      list(r_fun = function(flat) flat[[1]] * 2)
    },
    character(0),
    "closure",
    "mybackend",
    FALSE,
    function() test_device("mydev")
  )
  myarr <- structure(
    list(
      data = c(1, 2),
      dtype = tengen::as_dtype("f64"),
      shape = 2L,
      ambiguous = FALSE,
      device = test_device("mydev"),
      backend = "mybackend"
    ),
    class = "AnvlArray"
  )
  expect_identical(impl_dispatch_run(d, list(myarr)), c(2, 4))
  invisible(impl_dispatch_run(d, list(myarr)))
  expect_equal(n_miss, 1L)

  # ...and an array of any other backend is rejected by name.
  qarr <- myarr
  qarr$backend <- "quickr"
  expect_error(
    impl_dispatch_run(d, list(x = qarr)),
    'expected an AnvlArray of backend "mybackend"; got "quickr"'
  )
})

test_that("phantom_specs allocate donation buffers of the requested dtype", {
  skip_if_not(plugins_downloaded())
  # An identity executable whose only input is supplied by the phantom spec,
  # so the call has zero dynamic leaves and the output dtype is the spec's.
  run <- function(mlir_ty, spec_dtype) {
    src <- sprintf(
      'func.func @main(%%x: tensor<2x%s>) -> tensor<2x%s> {
        "func.return"(%%x): (tensor<2x%s>) -> ()
      }',
      mlir_ty,
      mlir_ty,
      mlir_ty
    )
    d <- impl_dispatch_create(
      4L,
      function(info) {
        xla_entry(
          pjrt_compile(pjrt_program(src = src)),
          out_avals = list(oav(dtype = spec_dtype)),
          phantom_specs = list(list(dtype = spec_dtype, shape = 2L))
        )
      },
      "flag",
      "pjrt",
      "xla",
      FALSE,
      test_xla_device
    )
    impl_dispatch_run(d, list(flag = TRUE))
  }

  for (dt in c("f32", "f64", "i32")) {
    res <- run(dt, dt)
    expect_identical(as.character(res$dtype), dt)
    expect_identical(res$shape, 2L)
  }

  # "bool" is the canonical AnvlDtype name; "pred" is pjrt's C-API spelling and
  # "i1" the MLIR one. All three must land on the same wrapped dtype.
  for (alias in c("bool", "i1", "pred")) {
    expect_identical(as.character(run("i1", alias)$dtype), "bool")
  }

  expect_error(run("f32", "nonsense"), "Unsupported type")
})

test_that("a buffer and a literal of the same aval share one executable", {
  skip_if_not(plugins_downloaded())
  # kArray and kRData are the same key material: both trace to the same aval, so
  # they compile to the same program. Only where execution gets the input from
  # differs -- the leaf's buffer, or an upload of the leaf -- and that is decided
  # per call. Keying them apart would compile the identical program twice.
  add_src <- 'func.func @main(%x: tensor<f32>, %y: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%0): (tensor<f32>) -> ()
  }'
  exec1 <- pjrt_compile(pjrt_program(src = add_src))
  n_miss <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      xla_entry(exec1, out_avals = list(oav(shape = integer(0))))
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  # A rank-0 f32 buffer, flagged ambiguous, has the aval of the literal `3`.
  arr <- function(v) {
    buf <- pjrt_scalar(v, dtype = "f32")
    structure(
      list(data = buf, ambiguous = TRUE, device = tengen::device(buf), backend = "xla"),
      class = "AnvlArray"
    )
  }
  # The first leaf is an array in every call, so key$device is set throughout:
  # a call of literals alone would carry no device and key apart for that reason.
  expect_equal(out(impl_dispatch_run(d, list(arr(1), arr(3)))), 4) # miss
  expect_equal(out(impl_dispatch_run(d, list(arr(1), 3))), 4) # hit: uploads `3`
  expect_equal(n_miss, 1L)
  expect_equal(impl_dispatcher_size(d), 1L)

  # ...and the entry serves either kind at that position, in either order.
  expect_equal(out(impl_dispatch_run(d, list(arr(2), arr(5)))), 7)
  expect_equal(out(impl_dispatch_run(d, list(arr(2), 5))), 7)
  expect_equal(n_miss, 1L)

  # A differing aval still splits the key: `3L` defaults to i32, not f32.
  invisible(try(impl_dispatch_run(d, list(arr(1), 3L)), silent = TRUE))
  expect_equal(n_miss, 2L)
  expect_equal(impl_dispatcher_size(d), 2L)
})

test_that("closure-array leaves key on their dtype", {
  mk <- function() {
    n <- 0L
    d <- impl_dispatch_create(
      20L,
      function(info) {
        n <<- n + 1L
        list(r_fun = function(flat) list(v = flat[[1]]))
      },
      character(0),
      "closure",
      "quickr",
      FALSE,
      test_quickr_device
    )
    list(d = d, n = function() n)
  }
  qarr <- function(v, dtype) {
    structure(
      list(
        data = v,
        dtype = dtype,
        shape = as.integer(length(v)),
        ambiguous = FALSE,
        device = test_quickr_device(),
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }

  # Every dtype an AnvlDtype names -- which is every dtype tengen can build --
  # is its own cache key.
  dtypes <- c("bool", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "f32", "f64")
  h <- mk()
  for (s in dtypes) {
    invisible(impl_dispatch_run(h$d, list(qarr(c(1, 2), tengen::as_dtype(s)))))
  }
  expect_equal(h$n(), length(dtypes))

  # Same dtype, different values -> cache hit.
  h2 <- mk()
  invisible(impl_dispatch_run(h2$d, list(qarr(c(1, 2), tengen::as_dtype("f64")))))
  invisible(impl_dispatch_run(h2$d, list(qarr(c(7, 7), tengen::as_dtype("f64")))))
  expect_equal(h2$n(), 1L)

  # A dtype object AnvlDtype cannot name is rejected, not keyed approximately:
  # two such dtypes would otherwise share an aval and run each other's program.
  # tengen builds none of these, so this is a guard rather than a path.
  weird <- structure(list(value = 1L), class = c("WeirdType", "DataType"))
  h3 <- mk()
  expect_error(
    impl_dispatch_run(h3$d, list(x = qarr(c(1, 2), weird))),
    "invalid input `x`.*dtype is not one anvl can represent"
  )
  expect_equal(h3$n(), 0L) # rejected before the compile callback
})

test_that("devices are canonicalized: identity first, identical() fallback", {
  # One `const void*` serves every backend: a leaf's token is the address of
  # its `$device`'s canonical representative. An interned device is its own
  # canonical (a pointer compare); an equal-but-distinct one collapses to it.
  n_miss <- 0L
  mk <- function() {
    n_miss <<- 0L
    impl_dispatch_create(
      10L,
      function(info) {
        n_miss <<- n_miss + 1L
        list(r_fun = function(flat) list(v = flat[[1]]))
      },
      character(0),
      "closure",
      "quickr",
      FALSE,
      test_quickr_device
    )
  }
  qarr <- function(device) {
    structure(
      list(
        data = c(1, 2),
        dtype = tengen::as_dtype("f64"),
        shape = 2L,
        ambiguous = FALSE,
        device = device,
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }

  # An interned device is one key across calls...
  d <- mk()
  invisible(impl_dispatch_run(d, list(x = qarr(test_device("cpu")))))
  invisible(impl_dispatch_run(d, list(x = qarr(test_device("cpu")))))
  expect_equal(n_miss, 1L)

  # ...and a different device is a different key.
  invisible(impl_dispatch_run(d, list(x = qarr(test_device("gpu")))))
  expect_equal(n_miss, 2L)
  expect_equal(impl_dispatcher_size(d), 2L)

  # Two arrays on different devices conflict, naming the offender.
  expect_error(
    impl_dispatch_run(mk(), list(x = qarr(test_device("cpu")), y = qarr(test_device("gpu")))),
    "invalid input `y`.*different device"
  )

  # A backend that fails to intern is not punished: equal-but-distinct device
  # objects collapse to one canonical device, within a call and across calls
  # -- and a fresh object equal to an interned one lands on the same token.
  fresh <- function(id) structure(list(device = id), class = "QuickrDevice")
  expect_true(identical(fresh("cpu"), fresh("cpu")))
  d2 <- mk()
  invisible(impl_dispatch_run(d2, list(x = qarr(fresh("cpu")), y = qarr(fresh("cpu")))))
  invisible(impl_dispatch_run(d2, list(x = qarr(fresh("cpu")), y = qarr(test_device("cpu")))))
  expect_equal(n_miss, 1L) # one device, one entry, however it is spelled
  # ...while genuinely different fresh devices still conflict.
  expect_error(
    impl_dispatch_run(d2, list(x = qarr(fresh("cpu")), y = qarr(fresh("gpu")))),
    "invalid input `y`.*different device"
  )

  # A quickr array must carry $device, or a literal-only call (which resolves
  # the default) could never share its entry.
  no_dev <- qarr(test_device("cpu"))
  no_dev$device <- NULL
  expect_error(
    impl_dispatch_run(mk(), list(x = no_dev)),
    "invalid input `x`.*\\$device"
  )
})

test_that("a call with no array input keys on the resolved default device", {
  # Nothing names a device, but the entry the callback compiles is still bound to
  # one. Resolving per call means an entry compiled under one default device is
  # never served after the default changes.
  n_miss <- 0L
  current <- "cpu"
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      list(r_fun = function(flat) list(v = info$default_device))
    },
    character(0),
    "closure",
    "quickr",
    FALSE,
    function() test_device(current)
  )

  r1 <- impl_dispatch_run(d, list(x = 1))
  expect_equal(n_miss, 1L)
  # The resolved device reaches the callback, so it compiles for the same one.
  expect_s3_class(r1$v, "QuickrDevice")
  expect_identical(r1$v$device, "cpu")

  invisible(impl_dispatch_run(d, list(x = 1))) # same default -> hit
  expect_equal(n_miss, 1L)

  current <- "gpu" # the default changes mid-session
  r2 <- impl_dispatch_run(d, list(x = 1))
  expect_equal(n_miss, 2L) # ...so the old entry must not be served
  expect_identical(r2$v$device, "gpu")
  expect_equal(impl_dispatcher_size(d), 2L)
})

test_that("a dispatcher without a default_device rejects a call with no arrays", {
  d <- impl_dispatch_create(
    10L,
    function(info) list(r_fun = function(flat) list(v = 1)),
    character(0),
    "closure",
    "quickr",
    FALSE,
    NULL
  )
  expect_error(impl_dispatch_run(d, list(x = 1)), "without a `default_device` resolver")
})

test_that("the closure engine can pin a device (move_inputs): `r_fun` places its own inputs", {
  # Under move_inputs the entry's device is fixed by `compile`, so the device
  # is not key material and inputs may arrive from any device. The closure
  # engine delegates the placing to `r_fun` -- like the execution and the
  # output wrapping it already delegates -- so pjrt copies nothing here.
  n_miss <- 0L
  placed_on <- NULL
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      # the device this entry is compiled for; `r_fun` closes over it
      target <- test_device("gpu")
      list(
        r_fun = function(flat) {
          placed_on <<- target
          list(v = flat[[1]] + flat[[2]])
        }
      )
    },
    character(0),
    "closure",
    "quickr",
    TRUE,
    NULL # pinned: no `default_device` resolver needed
  )
  qarr <- function(v, device) {
    structure(
      list(
        data = v,
        dtype = tengen::as_dtype("f64"),
        shape = as.integer(length(v)),
        ambiguous = FALSE,
        device = device,
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }

  # inputs spread across devices: an error under the infer policy, fine here
  r1 <- impl_dispatch_run(
    d,
    list(x = qarr(1, test_device("cpu")), y = qarr(2, test_device("gpu")))
  )
  expect_identical(r1, list(v = 3))
  expect_identical(placed_on, test_device("gpu"))

  # the device is not part of the key: the same signature from another device
  # hits the entry that is already there
  r2 <- impl_dispatch_run(
    d,
    list(x = qarr(1, test_device("gpu")), y = qarr(2, test_device("gpu")))
  )
  expect_identical(r2, list(v = 3))
  expect_equal(n_miss, 1L)
  expect_equal(impl_dispatcher_size(d), 1L)
})

test_that("the compile callback receives the tree, leaves, static mask, and avals", {
  seen <- NULL
  d <- impl_dispatch_create(
    10L,
    function(info) {
      seen <<- info
      list(r_fun = function(flat) list(v = 1))
    },
    "flag",
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  qarr <- function(v, dtype = "f64") {
    structure(
      list(
        data = v,
        dtype = tengen::as_dtype(dtype),
        shape = as.integer(length(v)),
        ambiguous = FALSE,
        device = test_quickr_device(),
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }
  invisible(impl_dispatch_run(
    d,
    list(
      x = qarr(c(1, 2)),
      lit = 3L,
      flag = "s",
      nested = list(a = matrix(1:6, 2, 3)),
      b = TRUE
    )
  ))

  expect_named(seen, c("args", "in_tree", "leaves", "is_static", "avals", "default_device"))
  # An array named the device, so no default was resolved.
  expect_null(seen$default_device)
  expect_s3_class(seen$in_tree, "RTree")
  expect_length(seen$leaves, 5L)
  expect_identical(seen$is_static, c(FALSE, FALSE, TRUE, FALSE, FALSE))
  # The tree handed over is the one the mask was computed from.
  expect_identical(tree_leaf_mask(seen$in_tree, "flag"), seen$is_static)

  # The dispatched argument list is still available (device_arg() reads it).
  expect_identical(seen$args$flag, "s")

  # A static leaf has no aval; every dynamic one carries the key's own.
  expect_null(seen$avals[[3L]])
  expect_identical(seen$avals[[1L]], list(dtype = "f64", shape = 2L, ambiguous = FALSE))
  # A bare R literal: pjrt's default dtype, rank-0, and dtype-ambiguous.
  expect_identical(seen$avals[[2L]], list(dtype = "i32", shape = integer(), ambiguous = TRUE))
  # A bare R array keeps its dim.
  expect_identical(seen$avals[[4L]], list(dtype = "i32", shape = c(2L, 3L), ambiguous = TRUE))
  # The boolean dtype crosses into R under tengen's name, not pjrt's C-API
  # "pred": AnvlDtype's canonical vocabulary is the one anvl speaks.
  expect_identical(seen$avals[[5L]], list(dtype = "bool", shape = integer(), ambiguous = TRUE))
})

test_that("an xla leaf's aval comes from its buffer, not its $dtype/$shape", {
  skip_if_not(plugins_downloaded())
  # The generic aval read uses $dtype/$shape (any backend carries them), but
  # the pjrt engine takes a shortcut: the buffer already caches its element
  # type and dimensions. The two agree on any array anvl builds; where they
  # disagree, the buffer wins, because it cannot be falsified by a field that
  # drifted.
  seen <- NULL
  d <- impl_dispatch_create(
    10L,
    function(info) {
      seen <<- info$avals[[1L]]
      list(r_fun = function(flat) list(v = 1))
    },
    character(0),
    "pjrt",
    "xla",
    FALSE,
    test_xla_device
  )
  buf <- pjrt_buffer(matrix(1:6, 2, 3), dtype = "f32")
  lying <- structure(
    list(
      data = buf,
      dtype = tengen::as_dtype("i64"), # a lie
      shape = c(99L, 99L), # also a lie
      ambiguous = FALSE,
      device = tengen::device(buf),
      backend = "xla"
    ),
    class = "AnvlArray"
  )
  # The compile callback errors (no exec), but only after the aval was built.
  invisible(try(impl_dispatch_run(d, list(x = lying)), silent = TRUE))
  expect_identical(seen$dtype, "f32") # the buffer's element type
  expect_identical(seen$shape, c(2L, 3L)) # the buffer's dimensions
})

test_that("an aval's dtype reaches the callback as a name, whatever the backend", {
  seen <- NULL
  d <- impl_dispatch_create(
    10L,
    function(info) {
      seen <<- info
      list(r_fun = function(flat) list(v = 1))
    },
    character(0),
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  arr <- function(dtype) {
    structure(
      list(
        data = c(1, 2),
        dtype = tengen::as_dtype(dtype),
        shape = 2L,
        ambiguous = FALSE,
        device = test_quickr_device(),
        backend = "quickr"
      ),
      class = "AnvlArray"
    )
  }
  invisible(impl_dispatch_run(d, list(x = arr("i64"), y = arr("ui16"))))
  # A quickr leaf's tengen $dtype object becomes an AnvlDtype like any other, so
  # the callback sees the same canonical string an xla leaf's buffer would give.
  expect_identical(seen$avals[[1L]]$dtype, "i64")
  expect_identical(seen$avals[[2L]]$dtype, "ui16")
})

test_that("every input is validated natively, naming the offending argument", {
  arr <- function(backend, data = c(1, 2)) {
    structure(
      list(
        data = data,
        dtype = tengen::as_dtype("f64"),
        shape = as.integer(length(data)),
        ambiguous = FALSE,
        device = test_quickr_device(),
        backend = backend
      ),
      class = "AnvlArray"
    )
  }
  n_miss <- 0L
  mk <- function(engine, static = character(0)) {
    impl_dispatch_create(
      10L,
      function(info) {
        n_miss <<- n_miss + 1L
        list(r_fun = function(flat) list(v = flat[[1]]))
      },
      static,
      engine,
      if (engine == "pjrt") "xla" else "quickr",
      FALSE,
      test_xla_device
    )
  }

  # "quickr" is the closure engine's array leaf: unwrapped to its $data.
  expect_identical(impl_dispatch_run(mk("closure"), list(arr("quickr")))$v, c(1, 2))

  # Rejections happen before the cache is probed, so `compile` is never called.
  n_miss <- 0L

  # An AnvlArray of the wrong backend for the dispatcher.
  expect_error(impl_dispatch_run(mk("closure"), list(x = arr("xla"))), "invalid input `x`.*\"quickr\"")
  expect_error(impl_dispatch_run(mk("pjrt"), list(x = arr("quickr"))), "invalid input `x`.*\"xla\"")

  # anvl's "plain" backend captures trace-time constants; never a call argument.
  expect_error(impl_dispatch_run(mk("closure"), list(x = arr("plain"))), "invalid input `x`.*plain")
  expect_error(impl_dispatch_run(mk("pjrt"), list(x = arr("plain"))), "invalid input `x`.*plain")

  # Values no engine can turn into an array.
  expect_error(impl_dispatch_run(mk("closure"), list(x = c(1, 2, 3))), "invalid input `x`.*<numeric> of length 3")
  expect_error(impl_dispatch_run(mk("closure"), list(x = "hello")), "invalid input `x`.*<character> of length 1")
  expect_error(impl_dispatch_run(mk("closure"), list(x = NA_real_)), "invalid input `x`")
  # A classed numeric is not bare R data: it must not slip through as a leaf the
  # compile callback would trace but execution would never supply.
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = structure(1, class = "myclass"))),
    "invalid input `x`.*<myclass> of length 1"
  )

  # The path names a nested leaf, not just a top-level argument...
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = list(a = "hello"))),
    "invalid input `x\\$a`"
  )
  # ...and positionally indexes an unnamed one.
  expect_error(impl_dispatch_run(mk("closure"), list("hello")), "invalid input `\\[\\[1\\]\\]`")

  # A static argument must not be an AnvlArray: it would key the cache on its
  # contents, and be traced as an input execution never supplies.
  expect_error(
    impl_dispatch_run(mk("closure", static = "s"), list(s = arr("quickr"))),
    "invalid static input `s`.*must not be an AnvlArray"
  )

  expect_equal(n_miss, 0L)

  # Statics are otherwise keyed by value, whatever their type.
  d <- impl_dispatch_create(
    10L,
    function(info) list(r_fun = function(flat) list(v = "ok")),
    "s",
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  expect_identical(impl_dispatch_run(d, list(s = "hello"))$v, "ok")
})

test_that("bitwise number comparison keeps NA_integer64_ apart from 0", {
  skip_if_not_installed("bit64")
  # bit64 stores NA_integer64_ as the int64 minimum, whose double
  # reinterpretation is -0.0. Under R's default identical() (num.eq = TRUE)
  # that compares equal to 0, so the two would share one cache entry and the
  # NA call would run the executable compiled for 0.
  zero <- bit64::as.integer64(0)
  na64 <- bit64::NA_integer64_
  expect_true(identical(zero, na64)) # R's default: the trap

  # ...and the cache keeps them apart.
  n <- 0L
  d <- impl_dispatch_create(
    10L,
    function(info) {
      n <<- n + 1L
      list(r_fun = function(flat) list(v = 1))
    },
    "flag",
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  invisible(impl_dispatch_run(d, list(flag = zero)))
  invisible(impl_dispatch_run(d, list(flag = na64)))
  expect_equal(n, 2L)
  expect_equal(impl_dispatcher_size(d), 2L)
})

test_that("a capacity below 1 is rejected rather than segfaulting", {
  # capacity 0 makes the LRU evict each entry as it is inserted, so the compile
  # path would dereference a null entry.
  expect_error(
    impl_dispatch_create(0L, function(info) list(), character(0), "closure", "quickr", FALSE, test_quickr_device),
    "capacity"
  )
  expect_error(
    impl_dispatch_create(-1L, function(info) list(), character(0), "closure", "quickr", FALSE, test_quickr_device),
    "capacity"
  )
  expect_error(
    dispatcher(0L, function(info) list(), default_device = test_xla_device),
    "Must be >= 1"
  )
})

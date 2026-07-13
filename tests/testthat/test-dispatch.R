# The native eager-dispatch fast path (src/dispatch.cpp, dispatch_engine.cpp).
#
# The dispatcher exists to serve anvl's jit(), so that is how it is tested: the
# first half of this file drives the whole loop through jit() -- anvl traces and
# compiles in the miss callback, the dispatcher caches, executes, and wraps --
# and asserts on what a user of jit() observes, including the cache-key
# semantics, which show up as a recompile or the lack of one.
#
# The second half drives the building blocks directly, and only for what jit()
# cannot express: the compile-callback contract (and its violations), the
# closure engine's contract with a backend, device canonicalization, and the
# guards that exist so a misuse errors instead of segfaulting. anvl reaches none
# of these, because anvl is a correct caller.
#
# The cache key's structure -- avals, device token, the kArray/kRData merge, and
# the hash/equality contract -- is additionally tested in C++, in
# src/test-dispatch.cpp, where a device token can be fabricated and a leaf built
# directly.

# ---------------------------------------------------------------------------
# Through jit(), the way the dispatcher is meant to be used.
# ---------------------------------------------------------------------------

# anvl sits in Suggests, so every jit() test is a no-op where it is absent.
skip_if_no_jit <- function() {
  skip_if_not_installed("anvl")
  skip_if_not(plugins_downloaded())
}

# The dispatcher a jitted function dispatches through (anvl stores it in the
# closure environment of the function's fast entry).
jit_dispatcher <- function(f) {
  environment(attr(f, "jit_run_args"))$dispatcher
}
jit_size <- function(f) dispatcher_size(jit_dispatcher(f))

arr_of <- function(res) as.numeric(tengen::as_array(res))

test_that("jit() dispatches, caches, and returns wrapped arrays", {
  skip_if_no_jit()
  f <- anvl::jit(function(x, y) x + y)
  x <- anvl::nv_array(c(1, 2, 3), dtype = "f32")
  y <- anvl::nv_array(c(10, 20, 30), dtype = "f32")

  r1 <- f(x, y)
  # The result is a fully wrapped array: the dispatcher built it natively.
  expect_s3_class(r1, "AnvlArray")
  expect_identical(as.character(tengen::dtype(r1)), "f32")
  expect_identical(tengen::shape(r1), 3L)
  expect_s3_class(r1$data, "PJRTBuffer")
  expect_s3_class(tengen::device(r1), "PJRTDevice")
  expect_identical(r1$backend, "xla")
  expect_equal(arr_of(r1), c(11, 22, 33))

  # A second call of the same signature is a cache hit...
  expect_equal(arr_of(f(x, y)), c(11, 22, 33))
  d <- jit_dispatcher(f)
  expect_s3_class(d, "Dispatcher")
  expect_equal(dispatcher_size(d), 1L)

  # ...an output feeds straight back in as an input, without re-compiling...
  expect_equal(arr_of(f(r1, y)), c(21, 42, 63))
  expect_equal(dispatcher_size(d), 1L)

  # ...and a new shape is a new cache entry.
  invisible(f(anvl::nv_array(1, dtype = "f32"), anvl::nv_array(2, dtype = "f32")))
  expect_equal(dispatcher_size(d), 2L)

  # GC-correct: many dispatches with periodic gc(), then teardown.
  for (i in 1:300) {
    r <- f(x, y)
    if (i %% 100 == 0) {
      gc()
    }
    expect_equal(arr_of(r), c(11, 22, 33))
  }
  rm(f, d)
  gc()
  expect_true(TRUE) # reached teardown without crashing
})

test_that("jit() preserves nested output structure and names", {
  skip_if_no_jit()
  f <- anvl::jit(function(x) list(sum = x + x, nested = list(sq = x * x)))
  res <- f(anvl::nv_array(c(2, 3), dtype = "f32"))
  expect_named(res, c("sum", "nested"))
  expect_named(res$nested, "sq")
  expect_equal(arr_of(res$sum), c(4, 6))
  expect_equal(arr_of(res$nested$sq), c(4, 9))
})

test_that("the ambiguity of an output survives the native wrap", {
  skip_if_no_jit()
  # x + 1 keeps a committed f32's dtype: the output is unambiguous. A literal
  # alone stays ambiguous. Both bits are stamped by the dispatcher's wrap.
  f <- anvl::jit(function(x) x + 1)
  expect_false(f(anvl::nv_array(1, dtype = "f32"))$ambiguous)
  g <- anvl::jit(function(x) x + 1)
  expect_true(g(2)$ambiguous)
})

test_that("jit() with static args compiles per static value", {
  skip_if_no_jit()
  f <- anvl::jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
  x <- anvl::nv_array(3, dtype = "f32")
  expect_equal(arr_of(f(x, TRUE)), 4)
  expect_equal(arr_of(f(x, FALSE)), 6)
  expect_equal(arr_of(f(x, TRUE)), 4) # hit
  expect_equal(jit_size(f), 2L)
})

test_that("a jitted call with no dynamic input dispatches on its statics alone", {
  skip_if_no_jit()
  # Zero dynamic leaves: the whole call is the static `n`, and the entry's
  # device comes from the compile callback rather than from an input.
  f <- anvl::jit(function(n) anvl::nv_eye(n), static = "n")
  expect_equal(tengen::as_array(f(2L)), diag(2))
  expect_equal(tengen::as_array(f(2L)), diag(2))
  expect_equal(jit_size(f), 1L)
})

test_that("jit() uploads bare R literals and arrays", {
  skip_if_no_jit()
  f <- anvl::jit(function(x, y) x + y)
  x <- anvl::nv_array(c(1, 2), dtype = "f32")

  # A bare double literal is uploaded as a rank-0 f32 buffer per call; the
  # signature does not change, so the second call is a cache hit.
  expect_equal(arr_of(f(x, 5)), c(6, 7))
  expect_equal(arr_of(f(x, 50)), c(51, 52))
  expect_equal(jit_size(f), 1L)

  # kArray and kRData are the same key material: an ambiguous rank-0 f32 array
  # has the literal's aval, so both trace to the same program and share the
  # entry. Only where execution reads the input from differs -- the leaf's
  # buffer, or an upload of the leaf -- and that is decided per call.
  # (nv_scalar(5) would not share it: it commits ambiguous = FALSE.)
  expect_equal(arr_of(f(x, anvl::nv_scalar(5, ambiguous = TRUE))), c(6, 7))
  expect_equal(jit_size(f), 1L)

  # A differing aval still splits the key: `3L` defaults to i32, not f32.
  invisible(try(f(x, 3L), silent = TRUE))
  expect_equal(jit_size(f), 2L)

  # An R array leaf uploads column-major, like pjrt_buffer().
  g <- anvl::jit(function(x) x)
  m <- matrix(c(1, 2, 3, 4), nrow = 2)
  expect_equal(tengen::as_array(g(m)), m)
})

test_that("every dtype is its own cache entry", {
  skip_if_no_jit()
  # Every dtype an AnvlDtype names -- which is every dtype tengen can build.
  dtypes <- c("bool", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "f32", "f64")
  f <- anvl::jit(function(x) x)
  for (dt in dtypes) {
    invisible(f(anvl::nv_array(c(1, 2), dtype = dt)))
  }
  expect_equal(jit_size(f), length(dtypes))

  # Same dtype and shape, different values -> cache hit.
  g <- anvl::jit(function(x) x)
  invisible(g(anvl::nv_array(c(1, 2), dtype = "f64")))
  invisible(g(anvl::nv_array(c(7, 7), dtype = "f64")))
  expect_equal(jit_size(g), 1L)
})

# What follows is how the cache key treats real R values as static arguments.
# Each test drives a jitted function and counts cache entries, because an entry
# per distinct key is the behaviour that matters, not a hash the caller never
# sees. `x + 1` ignores `s` entirely, so the only thing that can split the cache
# is the static's key.

# TRUE iff the two static values are one cache key.
same_key <- function(a, b) {
  f <- anvl::jit(function(x, s) x + 1, static = "s")
  x <- anvl::nv_array(c(1, 2), dtype = "f32")
  invisible(f(x, a))
  invisible(f(x, b))
  jit_size(f) == 1L
}

test_that("static args are keyed with identical(), environment included", {
  skip_if_no_jit()
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
  skip_if_no_jit()
  # The contract: keys the dispatcher calls equal MUST hash alike, or the map
  # stores two entries for one key. Two entries here would mean a hash that
  # disagrees with the equality.
  utf8 <- "é"
  latin1 <- iconv(utf8, "UTF-8", "latin1")
  expect_true(same_key(utf8, latin1)) # same string, different bytes
  expect_true(same_key(1.5, 1.5))
  expect_true(same_key(NaN, NaN))
  expect_true(same_key(1:3, c(1L, 2L, 3L))) # ALTREP compact seq vs materialized
})

test_that("static numbers are keyed bitwise: +0 and -0 are distinct", {
  skip_if_no_jit()
  # A literal `-0` is constant-folded to `+0` by R's byte compiler, so build it
  # from a variable -- otherwise this would quietly compare 0 against 0.
  zero <- 0
  neg_zero <- -1 * zero
  expect_identical(sprintf("%a", neg_zero), "-0x0p+0")
  expect_false(same_key(zero, neg_zero))
})

test_that("bitwise number comparison keeps NA_integer64_ apart from 0", {
  skip_if_no_jit()
  skip_if_not_installed("bit64")
  # bit64 stores NA_integer64_ as the int64 minimum, whose double
  # reinterpretation is -0.0. Under R's default identical() (num.eq = TRUE)
  # that compares equal to 0, so the two would share one cache entry and the
  # NA call would run the executable compiled for 0.
  zero <- bit64::as.integer64(0)
  na64 <- bit64::NA_integer64_
  expect_true(identical(zero, na64)) # R's default: the trap
  expect_false(same_key(zero, na64)) # ...and the cache keeps them apart
})

test_that("distinct static values never merge", {
  skip_if_no_jit()
  expect_false(same_key(1L, 1)) # type is folded before the contents
  expect_false(same_key(TRUE, FALSE))
  expect_false(same_key(NaN, NA_real_))
  expect_false(same_key(c(1, 2), c(2, 1)))
  expect_false(same_key(1 + 2i, 1 + 3i))
  expect_false(same_key(as.raw(1), as.raw(2)))
  expect_false(same_key(NA_character_, "NA"))
})

test_that("invalid jit() inputs are rejected natively, naming the argument", {
  skip_if_no_jit()
  f <- anvl::jit(function(x, y) x + y)
  x <- anvl::nv_array(c(1, 2), dtype = "f32")
  expect_error(f(x, "nope"), "invalid input `y`.*<character> of length 1")
  expect_error(f(x, c(1, 2, 3)), "invalid input `y`.*<numeric> of length 3")
  expect_equal(jit_size(f), 0L) # rejected before any compile

  # A static argument must not be an AnvlArray: it would key the cache on its
  # contents, and be traced as an input execution never supplies.
  g <- anvl::jit(function(x, s) x + 1, static = "s")
  expect_error(g(x, x), "invalid static input `s`.*must not be an AnvlArray")
  expect_equal(jit_size(g), 0L)
})

test_that("jit() rejects inputs spread across devices, naming the input", {
  skip_if_no_jit()
  skip_if(length(devices(pjrt_client("cpu"))) < 2L, "needs a second cpu device")
  f <- anvl::jit(function(x, y) x + y)
  x0 <- anvl::nv_array(c(1, 2), dtype = "f32", device = "cpu:0")
  y1 <- anvl::nv_array(c(3, 4), dtype = "f32", device = "cpu:1")

  # Without a fixed target device the first array's device is the call's, and a
  # conflicting input is an error -- caught natively, before the cache is
  # probed, so nothing is compiled.
  expect_error(f(x0, y1), "invalid input `y`.*different device")
  expect_equal(jit_size(f), 0L)
})

test_that("jit(device = ) fixes the entry's device and moves inputs to it", {
  skip_if_no_jit()
  f <- anvl::jit(function(x) x + 1, device = "cpu:0")
  x0 <- anvl::nv_array(c(1, 2), dtype = "f32", device = "cpu:0")
  res <- f(x0)
  expect_equal(arr_of(res), c(2, 3))
  # Devices are interned, so the wrapped output carries the very object.
  expect_identical(tengen::device(res), pjrt_device("cpu:0"))

  skip_if(length(devices(pjrt_client("cpu"))) < 2L, "needs a second cpu device")
  # An input on another device is copied to the target rather than rejected,
  # and the device is not part of the key: one entry serves both.
  y1 <- anvl::nv_array(c(3, 4), dtype = "f32", device = "cpu:1")
  expect_equal(arr_of(f(y1)), c(4, 5))
  expect_equal(jit_size(f), 1L)
})

test_that("a jitted function with no array inputs keys on the default device", {
  skip_if_no_jit()
  f <- anvl::jit(function(n) n + 1)
  expect_equal(arr_of(f(41)), 42)
  expect_equal(arr_of(f(41)), 42)
  expect_equal(jit_size(f), 1L)
})

test_that("the quickr backend dispatches through the closure engine", {
  skip_if_not_installed("anvl")
  skip_if_not_installed("quickr")
  anvl::with_backend("quickr", {
    f <- anvl::jit(function(x, y) x + y)
    x <- anvl::nv_array(c(1, 2), dtype = "f64")
    y <- anvl::nv_array(c(10, 20), dtype = "f64")
    r1 <- f(x, y)
    expect_s3_class(r1, "AnvlArray")
    expect_identical(r1$backend, "quickr")
    expect_equal(arr_of(r1), c(11, 22))
    invisible(f(x, y))
    expect_equal(jit_size(f), 1L)
  })
})

# ---------------------------------------------------------------------------
# The building blocks directly, for what jit() cannot express.
# ---------------------------------------------------------------------------

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

# An xla array leaf, as anvl builds them: an "AnvlArray" whose $data is a buffer.
xarr <- function(buf) {
  structure(
    list(data = buf, ambiguous = FALSE, device = tengen::device(buf), backend = "xla"),
    class = "AnvlArray"
  )
}

# A closure-engine array leaf, tagged with whichever backend and device.
qarr <- function(v, dtype = "f64", device = test_quickr_device(), backend = "quickr") {
  structure(
    list(
      data = v,
      dtype = if (is.character(dtype)) tengen::as_dtype(dtype) else dtype,
      shape = as.integer(length(v)),
      ambiguous = FALSE,
      device = device,
      backend = backend
    ),
    class = "AnvlArray"
  )
}

# With a leaf out_tree, dispatch() returns the single output as one wrapped
# array: an "AnvlArray" list whose $data is the output buffer.
out <- function(res) as.numeric(tengen::as_array(await(res$data)))

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
  invisible(impl_dispatch_run(d, list(x = qarr(c(1, 2), "i64"), y = qarr(c(1, 2), "ui16"))))
  # A quickr leaf's tengen $dtype object becomes an AnvlDtype like any other, so
  # the callback sees the same canonical string an xla leaf's buffer would give.
  expect_identical(seen$avals[[1L]]$dtype, "i64")
  expect_identical(seen$avals[[2L]]$dtype, "ui16")
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

test_that("out_avals and out_tree are the callback's claim, and are honoured", {
  skip_if_not(plugins_downloaded())
  two_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = "stablehlo.multiply"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0, %1): (tensor<2xf32>, tensor<2xf32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = two_src))
  mk <- function(out_tree, out_avals) {
    impl_dispatch_create(
      10L,
      function(info) xla_entry(exec, out_tree = out_tree, out_avals = out_avals),
      character(0),
      "pjrt",
      "xla",
      FALSE,
      test_xla_device
    )
  }
  x <- xarr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- xarr(pjrt_buffer(c(3, 4), dtype = "f32"))

  # Every wrapped field comes from the declared aval, not from the buffer: the
  # first output is stamped ambiguous although nothing about the buffer is.
  d <- mk(
    build_tree(list(sum = 0, rest = list(prod = 0))),
    list(oav(ambiguous = TRUE), oav())
  )
  res <- impl_dispatch_run(d, list(x, y))
  expect_equal(out(res$sum), c(4, 6))
  expect_equal(out(res$rest$prod), c(3, 8))
  expect_identical(res$sum$ambiguous, TRUE)
  expect_identical(res$rest$prod$ambiguous, FALSE)
  expect_identical(res$sum$shape, 2L)
  expect_identical(res$sum$dtype, tengen::as_dtype("f32"))

  # An out_tree whose leaf count disagrees with the executable's actual output
  # count is the one half of the callback's claim pjrt can still settle, and it
  # does -- on execution, against the real outputs.
  expect_error(
    impl_dispatch_run(mk(build_tree(list(0, 0, 0)), list(oav(), oav(), oav())), list(x, y)),
    "out_tree has 3 leaves but the executable returned 2 outputs"
  )
  # An out_avals that disagrees with out_tree is caught at compile time,
  # before the entry is ever cached.
  expect_error(
    impl_dispatch_run(mk(build_tree(list(0, 0)), list(oav())), list(x, y)),
    "out_avals has length 1 but out_tree has 2 leaves"
  )
})

test_that("the pjrt engine validates the compile callback's entry", {
  skip_if_not(plugins_downloaded())
  id_src <- 'func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    "func.return"(%x): (tensor<2xf32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = id_src))
  mk <- function(entry_fn) {
    impl_dispatch_create(10L, entry_fn, character(0), "pjrt", "xla", FALSE, test_xla_device)
  }
  x <- xarr(pjrt_buffer(c(1, 2), dtype = "f32"))

  # A missing client (needed for uploads, phantoms, and the wrap's device) is a
  # clear error, not a crash at input-assembly time.
  d_bad <- mk(function(info) {
    list(exec = exec, device = pjrt_device("cpu:0"), out_tree = build_tree(0))
  })
  expect_error(impl_dispatch_run(d_bad, list(x)), "must return `client`")

  # So is a const_arrays element that is not a PJRTBuffer: execute would
  # reinterpret the external pointer blindly and segfault, so it must be
  # rejected when the entry is built (here: the exec itself, a plausible slip).
  d_bad2 <- mk(function(info) xla_entry(exec, const_arrays = list(exec)))
  expect_error(
    impl_dispatch_run(d_bad2, list(x)),
    "const_arrays\\[\\[1\\]\\]` must be a PJRTBuffer"
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

test_that("a dispatcher with static names still dispatches a pure-dynamic call", {
  skip_if_not(plugins_downloaded())
  # A static name the dispatcher was built with need not appear in every call:
  # anvl's jit() would reject the name outright unless it is a formal of `f`,
  # so only the raw dispatcher can be handed a call that omits it.
  add_src <- 'func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0): (tensor<2xf32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = add_src))
  d <- dispatcher(
    10L,
    function(info) xla_entry(exec),
    static = "flag",
    default_device = test_xla_device
  )
  x <- xarr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- xarr(pjrt_buffer(c(3, 4), dtype = "f32"))
  expect_equal(out(dispatch(d, list(x = x, y = y))), c(4, 6))
})

test_that("the closure engine passes the dynamic leaves to `r_fun`, and nothing else", {
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
  a <- qarr(c(1, 2))
  b <- qarr(c(10, 20))
  # Only the two arrays are inputs; the static `flag` never reaches `r_fun`.
  r1 <- impl_dispatch_run(d, list(x = a, y = b, flag = TRUE))
  expect_identical(r1, list(n_inputs = 2L, sum = c(11, 22), flag = TRUE))
  r2 <- impl_dispatch_run(d, list(x = a, y = b, flag = TRUE))
  expect_identical(r2, list(n_inputs = 2L, sum = c(11, 22), flag = TRUE))
  expect_equal(n_miss, 1L) # same signature -> hit
  expect_equal(impl_dispatcher_size(d), 1L)

  # a different dtype is a new signature
  invisible(impl_dispatch_run(d, list(x = qarr(c(1L, 2L), "i32"), y = b, flag = TRUE)))
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
  myarr <- qarr(c(1, 2), device = test_device("mydev"), backend = "mybackend")
  expect_identical(impl_dispatch_run(d, list(myarr)), c(2, 4))
  invisible(impl_dispatch_run(d, list(myarr)))
  expect_equal(n_miss, 1L)

  # ...and an array of any other backend is rejected by name.
  expect_error(
    impl_dispatch_run(d, list(x = qarr(c(1, 2), device = test_device("mydev")))),
    'expected an AnvlArray of backend "mybackend"; got "quickr"'
  )
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

  # inputs spread across devices: an error under the infer policy, fine here
  r1 <- impl_dispatch_run(
    d,
    list(x = qarr(1, device = test_device("cpu")), y = qarr(2, device = test_device("gpu")))
  )
  expect_identical(r1, list(v = 3))
  expect_identical(placed_on, test_device("gpu"))

  # the device is not part of the key: the same signature from another device
  # hits the entry that is already there
  r2 <- impl_dispatch_run(
    d,
    list(x = qarr(1, device = test_device("gpu")), y = qarr(2, device = test_device("gpu")))
  )
  expect_identical(r2, list(v = 3))
  expect_equal(n_miss, 1L)
  expect_equal(impl_dispatcher_size(d), 1L)
})

test_that("devices are canonicalized: identity first, identical() fallback", {
  # One `const void*` serves every backend: a leaf's token is the address of
  # its `$device`'s canonical representative. An interned device is its own
  # canonical (a pointer compare); an equal-but-distinct one collapses to it.
  # pjrt interns its own devices, so only a raw dispatcher can present the
  # equal-but-distinct case.
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

  # An interned device is one key across calls...
  d <- mk()
  invisible(impl_dispatch_run(d, list(x = qarr(c(1, 2), device = test_device("cpu")))))
  invisible(impl_dispatch_run(d, list(x = qarr(c(1, 2), device = test_device("cpu")))))
  expect_equal(n_miss, 1L)

  # ...and a different device is a different key.
  invisible(impl_dispatch_run(d, list(x = qarr(c(1, 2), device = test_device("gpu")))))
  expect_equal(n_miss, 2L)
  expect_equal(impl_dispatcher_size(d), 2L)

  # Two arrays on different devices conflict, naming the offender.
  expect_error(
    impl_dispatch_run(
      mk(),
      list(x = qarr(c(1, 2), device = test_device("cpu")), y = qarr(c(1, 2), device = test_device("gpu")))
    ),
    "invalid input `y`.*different device"
  )

  # A backend that fails to intern is not punished: equal-but-distinct device
  # objects collapse to one canonical device, within a call and across calls
  # -- and a fresh object equal to an interned one lands on the same token.
  fresh <- function(id) structure(list(device = id), class = "QuickrDevice")
  expect_true(identical(fresh("cpu"), fresh("cpu")))
  d2 <- mk()
  invisible(impl_dispatch_run(
    d2,
    list(x = qarr(c(1, 2), device = fresh("cpu")), y = qarr(c(1, 2), device = fresh("cpu")))
  ))
  invisible(impl_dispatch_run(
    d2,
    list(x = qarr(c(1, 2), device = fresh("cpu")), y = qarr(c(1, 2), device = test_device("cpu")))
  ))
  expect_equal(n_miss, 1L) # one device, one entry, however it is spelled
  # ...while genuinely different fresh devices still conflict.
  expect_error(
    impl_dispatch_run(
      d2,
      list(x = qarr(c(1, 2), device = fresh("cpu")), y = qarr(c(1, 2), device = fresh("gpu")))
    ),
    "invalid input `y`.*different device"
  )

  # An array must carry $device, or a literal-only call (which resolves the
  # default) could never share its entry.
  no_dev <- qarr(c(1, 2))
  no_dev$device <- NULL
  expect_error(impl_dispatch_run(mk(), list(x = no_dev)), "invalid input `x`.*\\$device")
})

test_that("a call with no array input keys on the resolved default device", {
  # Nothing names a device, but the entry the callback compiles is still bound to
  # one. Resolving per call means an entry compiled under one default device is
  # never served after the default changes -- a default anvl reads off the
  # session, and so cannot change mid-test.
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

test_that("an input pjrt cannot classify is rejected, naming the offending argument", {
  # jit() covers the values a user can plausibly pass; these are the ones only a
  # broken caller produces -- a mistagged array, or a leaf buried in a list.
  n_miss <- 0L
  mk <- function(engine) {
    impl_dispatch_create(
      10L,
      function(info) {
        n_miss <<- n_miss + 1L
        list(r_fun = function(flat) list(v = flat[[1]]))
      },
      character(0),
      engine,
      if (engine == "pjrt") "xla" else "quickr",
      FALSE,
      test_xla_device
    )
  }

  # An AnvlArray of the wrong backend for the dispatcher.
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = qarr(c(1, 2), backend = "xla"))),
    "invalid input `x`.*\"quickr\""
  )
  expect_error(
    impl_dispatch_run(mk("pjrt"), list(x = qarr(c(1, 2)))),
    "invalid input `x`.*\"xla\""
  )

  # anvl's "plain" backend captures trace-time constants; never a call argument.
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = qarr(c(1, 2), backend = "plain"))),
    "invalid input `x`.*plain"
  )
  expect_error(
    impl_dispatch_run(mk("pjrt"), list(x = qarr(c(1, 2), backend = "plain"))),
    "invalid input `x`.*plain"
  )

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

  # A dtype object AnvlDtype cannot name is rejected, not keyed approximately:
  # two such dtypes would otherwise share an aval and run each other's program.
  # tengen builds none of these, so this is a guard rather than a path.
  weird <- structure(list(value = 1L), class = c("WeirdType", "DataType"))
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = qarr(c(1, 2), dtype = weird))),
    "invalid input `x`.*dtype is not one anvl can represent"
  )

  expect_equal(n_miss, 0L) # every rejection happened before the cache was probed
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

# The `default_device` resolvers a dispatcher needs when a call has no array
# input to read a device from. `test_device()` returns a fresh object each call
# (like anvl's quickr backend): the dispatcher canonicalizes devices with
# identical() as a fallback to object identity, so equal-but-distinct devices
# still collapse to one. The identity fast path is exercised where a test reuses
# one object (see "devices are canonicalized").
test_pjrt_device <- function() pjrt_device("cpu:0")
test_device <- function(id = "cpu") structure(list(device = id), class = "QuickrDevice")
test_quickr_device <- function() test_device("cpu")

# One output aval, as the compile callback declares it: the dtype and shape
# pjrt stamps on that output's wrapper. Same shape as the input avals the
# callback receives in `info$avals`.
oav <- function(dtype = "f32", shape = 2L) {
  list(dtype = dtype, shape = as.integer(shape))
}

# The pjrt engine's full compile-callback contract for a single-output
# executable: exec, client + device (uploads, moves, phantoms, and the wrapped
# outputs' $device), a leaf out_tree (a single un-nested output), and the
# out_avals the outputs are wrapped from. The default is the tests' most common
# program: one f32 output of shape 2.
pjrt_entry <- function(
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

# A pjrt array leaf, as anvl builds them: an "AnvlArray" whose $data is a buffer.
parr <- function(buf) {
  structure(
    list(data = buf, device = tengen::device(buf), backend = "pjrt"),
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
      device = device,
      backend = backend
    ),
    class = "AnvlArray"
  )
}

# With a leaf out_tree, dispatch() returns the single output as one wrapped
# array: an "AnvlArray" list whose $data is the output buffer.
out <- function(res) as.numeric(tengen::as_array(await(res$data)))

# The closure engine reads a leaf's metadata through a backend-supplied
# extractor rather than by reaching for fields. The test arrays (qarr) do store
# fields, so the test extractor simply reads them back -- standing in for a
# backend whose accessors happen to be `$` reads.
test_extractor <- function(leaf) {
  list(
    aval = list(dtype = leaf$dtype, shape = leaf$shape),
    device = leaf$device,
    backend = leaf$backend
  )
}

# impl_dispatcher_create with the extractor wired up the way each engine needs it:
# the closure engine requires one (here the field-reading test extractor); the
# pjrt engine ignores it and reads the PJRTBuffer directly.
new_dispatcher <- function(capacity, miss, static, engine, backend, move, default_device) {
  extractor <- if (engine == "pjrt") NULL else test_extractor
  impl_dispatcher_create(capacity, miss, static, engine, backend, move, default_device, extractor)
}
# ---------------------------------------------------------------------------
# Programs. `dispatcher()` needs something real to execute, so the tests compile
# tiny stablehlo functions rather than tracing them.
# ---------------------------------------------------------------------------

# Elementwise `x <op> y` over two tensors of one type.
binop_exec <- function(ty = "tensor<2xf32>", op = "stablehlo.add") {
  pjrt_compile(pjrt_program(
    src = sprintf(
      'func.func @main(%%x: %s, %%y: %s) -> %s {
       %%0 = "%s"(%%x, %%y) : (%s, %s) -> %s
       "func.return"(%%0): (%s) -> ()
     }',
      ty,
      ty,
      ty,
      op,
      ty,
      ty,
      ty,
      ty
    )
  ))
}

# Identity over one tensor, for tests that only care about the input's aval.
id_exec <- function(ty = "tensor<2xf32>") {
  pjrt_compile(pjrt_program(
    src = sprintf(
      'func.func @main(%%x: %s) -> %s {
       "func.return"(%%x): (%s) -> ()
     }',
      ty,
      ty,
      ty
    )
  ))
}

# The MLIR spelling of each dtype the engine can represent.
mlir_ty <- c(
  bool = "i1",
  i8 = "i8",
  i16 = "i16",
  i32 = "i32",
  i64 = "i64",
  ui8 = "ui8",
  ui16 = "ui16",
  ui32 = "ui32",
  ui64 = "ui64",
  f32 = "f32",
  f64 = "f64"
)

# An "rdata" aval names the leaf's R storage type, not a dtype -- the program
# decides what it is uploaded at. These are the choices the tests make, which
# happen to be pjrt_scalar()'s defaults.
rdata_upload_dtype <- c(double = "f32", integer = "i32", logical = "bool")


# ---------------------------------------------------------------------------
# The cache key: static arguments.
# ---------------------------------------------------------------------------
# What matters is an entry per distinct key, not a hash the caller never sees,
# so each test counts cache entries. `r_fun` ignores the static entirely, which
# leaves the static's key as the only thing that can split the cache.
#
# These run on the closure engine: static-ness is resolved in the dispatcher
# core before any engine is consulted, so the keying is identical, and this way
# the tests need no compiled program and no plugin.

# TRUE iff the two static values are one cache key.
same_key <- function(a, b) {
  d <- new_dispatcher(
    10L,
    function(info) list(r_fun = function(flat) flat[[1L]]),
    "s",
    "closure",
    "quickr",
    FALSE,
    test_quickr_device
  )
  x <- qarr(c(1, 2))
  invisible(dispatch(d, list(x = x, s = a)))
  invisible(dispatch(d, list(x = x, s = b)))
  dispatcher_size(d) == 1L
}

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
  # A literal `-0` is constant-folded to `+0` by R's byte compiler, so build it
  # from a variable -- otherwise this would quietly compare 0 against 0.
  # this is important for bit64 which uses -0 for NA
  zero <- 0
  neg_zero <- -1 * zero
  expect_false(same_key(zero, neg_zero))
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
  expect_false(same_key(zero, na64)) # ...and the cache keeps them apart
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

# ---------------------------------------------------------------------------
# The pjrt engine.
# ---------------------------------------------------------------------------

test_that("the pjrt engine wraps its outputs and caches one entry per signature", {
  skip_if_not(plugins_downloaded())
  n_compile <- 0L
  d <- dispatcher(
    10L,
    function(info) {
      n_compile <<- n_compile + 1L
      n <- info$avals[[1L]]$shape
      pjrt_entry(
        binop_exec(sprintf("tensor<%dxf32>", n)),
        out_avals = list(oav(shape = n))
      )
    },
    default_device = test_pjrt_device
  )
  x <- parr(pjrt_buffer(c(1, 2, 3), dtype = "f32"))
  y <- parr(pjrt_buffer(c(10, 20, 30), dtype = "f32"))

  r1 <- dispatch(d, list(x = x, y = y))
  # A fully wrapped array: the engine built it natively from `out_avals`.
  expect_s3_class(r1, "AnvlArray")
  expect_identical(as.character(r1$dtype), "f32")
  expect_identical(r1$shape, 3L)
  expect_s3_class(r1$data, "PJRTBuffer")
  expect_s3_class(r1$device, "PJRTDevice")
  expect_identical(r1$backend, "pjrt")
  expect_equal(out(r1), c(11, 22, 33))

  # A second call of the same signature is a cache hit...
  expect_equal(out(dispatch(d, list(x = x, y = y))), c(11, 22, 33))
  expect_equal(dispatcher_size(d), 1L)
  expect_equal(n_compile, 1L)

  # ...an output feeds straight back in as an input, without re-compiling...
  expect_equal(out(dispatch(d, list(x = r1, y = y))), c(21, 42, 63))
  expect_equal(dispatcher_size(d), 1L)

  # ...and a new shape is a new cache entry.
  s <- parr(pjrt_buffer(c(1, 2), dtype = "f32"))
  invisible(dispatch(d, list(x = s, y = s)))
  expect_equal(dispatcher_size(d), 2L)

  # GC-correct: many dispatches with periodic gc(), then teardown.
  for (i in 1:300) {
    r <- dispatch(d, list(x = x, y = y))
    if (i %% 100 == 0) {
      gc()
    }
    expect_equal(out(r), c(11, 22, 33))
  }
})

test_that("a static argument compiles one entry per distinct value", {
  skip_if_not(plugins_downloaded())
  d <- dispatcher(
    10L,
    # The static is what the callback compiles against: it is a constant of the
    # entry, not an execute-time input.
    function(info) {
      op <- if (isTRUE(info$args$flag)) "stablehlo.add" else "stablehlo.multiply"
      pjrt_entry(binop_exec(op = op))
    },
    static = "flag",
    default_device = test_pjrt_device
  )
  x <- parr(pjrt_buffer(c(2, 3), dtype = "f32"))
  y <- parr(pjrt_buffer(c(10, 10), dtype = "f32"))
  expect_equal(out(dispatch(d, list(x = x, y = y, flag = TRUE))), c(12, 13))
  expect_equal(out(dispatch(d, list(x = x, y = y, flag = FALSE))), c(20, 30))
  expect_equal(out(dispatch(d, list(x = x, y = y, flag = TRUE))), c(12, 13)) # hit
  expect_equal(dispatcher_size(d), 2L)
})

test_that("bare R data is keyed by its R storage type and uploaded column-major", {
  skip_if_not(plugins_downloaded())
  # A bare R leaf's aval names its R storage type -- "double", "integer",
  # "logical" -- and that is what the cache key is built from. It is not a dtype:
  # the callback picks the one the program takes the value at and declares it.
  d <- dispatcher(
    10L,
    function(info) {
      aval <- info$avals[[1L]]
      is_rdata <- aval$kind == "rdata"
      dt <- if (is_rdata) rdata_upload_dtype[[aval$dtype]] else aval$dtype
      shp <- aval$shape
      ty <- if (length(shp) == 0L) {
        sprintf("tensor<%s>", mlir_ty[[dt]])
      } else {
        sprintf("tensor<%sx%s>", paste(shp, collapse = "x"), mlir_ty[[dt]])
      }
      pjrt_entry(
        id_exec(ty),
        out_avals = list(oav(dtype = dt, shape = shp)),
        input_dtypes = if (is_rdata) dt else NA_character_
      )
    },
    default_device = test_pjrt_device
  )

  # The aval of a bare R value says what it is, not what a buffer holds.
  expect_identical(
    vapply(
      list(5, 3L, TRUE),
      function(x) {
        av <- NULL
        dd <- dispatcher(
          1L,
          function(info) {
            av <<- info$avals[[1L]]$dtype
            stop("not compiled")
          },
          default_device = test_pjrt_device
        )
        try(dispatch(dd, list(x = x)), silent = TRUE)
        av
      },
      character(1L)
    ),
    c("double", "integer", "logical")
  )

  # A rank-0 double literal, uploaded per call; the signature does not change,
  # so the second call is a cache hit.
  expect_equal(out(dispatch(d, list(x = 5))), 5)
  expect_equal(out(dispatch(d, list(x = 50))), 50)
  expect_equal(dispatcher_size(d), 1L)

  # An array leaf is not the same key material as bare R data, even at the same
  # aval: the R value has no dtype of its own until the program says what it is
  # used as, so the two compile to different programs and take separate entries.
  expect_equal(out(dispatch(d, list(x = parr(pjrt_scalar(5, dtype = "f32"))))), 5)
  expect_equal(dispatcher_size(d), 2L)

  # An integer literal is "integer", a different aval and so a new entry.
  invisible(dispatch(d, list(x = 3L)))
  expect_equal(dispatcher_size(d), 3L)

  # An R array uploads column-major, like pjrt_buffer().
  m <- matrix(c(1, 2, 3, 4), nrow = 2)
  expect_equal(tengen::as_array(await(dispatch(d, list(x = m))$data)), m)
})

test_that("every dtype the engine can represent is its own cache entry", {
  skip_if_not(plugins_downloaded())
  d <- dispatcher(
    50L,
    function(info) {
      dt <- info$avals[[1L]]$dtype
      pjrt_entry(
        id_exec(sprintf("tensor<2x%s>", mlir_ty[[dt]])),
        out_avals = list(oav(dtype = dt))
      )
    },
    default_device = test_pjrt_device
  )
  for (dt in names(mlir_ty)) {
    buf <- pjrt_empty(2L, dtype = if (dt == "bool") "pred" else dt)
    invisible(dispatch(d, list(x = parr(buf))))
  }
  expect_equal(dispatcher_size(d), length(mlir_ty))

  # Same dtype and shape, different values -> cache hit.
  expect_equal(
    dispatcher_size(d),
    {
      invisible(dispatch(d, list(x = parr(pjrt_buffer(c(7, 7), dtype = "f64")))))
      length(mlir_ty)
    }
  )
})

test_that("a static argument must not be an AnvlArray", {
  skip_if_not(plugins_downloaded())
  # It would key the cache on its contents, and the callback would trace it as
  # a real input that execution then never supplies.
  d <- dispatcher(
    10L,
    function(info) stop("must not reach the compile callback"),
    static = "s",
    default_device = test_pjrt_device
  )
  x <- parr(pjrt_buffer(c(1, 2), dtype = "f32"))
  expect_error(dispatch(d, list(x = x, s = x)), "must not be an AnvlArray")
  expect_equal(dispatcher_size(d), 0L)
})

test_that("inputs spread across devices are rejected, naming the input", {
  skip_if_not(plugins_downloaded())
  skip_if(length(devices(pjrt_client("cpu"))) < 2L, "needs a second cpu device")
  # Without a fixed target device the first array's device is the call's, and a
  # conflicting input is an error -- caught before the cache is probed, so
  # nothing is compiled.
  d <- dispatcher(
    10L,
    function(info) stop("must not reach the compile callback"),
    default_device = test_pjrt_device
  )
  x0 <- parr(pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0"))
  y1 <- parr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:1"))
  expect_error(dispatch(d, list(x = x0, y = y1)), "invalid input `y`.*different device")
  expect_equal(dispatcher_size(d), 0L)
})

test_that("move_inputs copies a pjrt input to the entry's device", {
  skip_if_not(plugins_downloaded())
  skip_if(length(devices(pjrt_client("cpu"))) < 2L, "needs a second cpu device")
  # The closure engine's move_inputs test above proves the *policy*; this one
  # proves the pjrt engine's copy, which is the only place pjrt itself moves a
  # buffer between devices.
  dev0 <- pjrt_device("cpu:0")
  d <- dispatcher(
    10L,
    function(info) pjrt_entry(binop_exec(), device = dev0),
    move_inputs = TRUE
  )
  x0 <- parr(pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0"))
  y0 <- parr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:0"))
  r <- dispatch(d, list(x = x0, y = y0))
  expect_equal(out(r), c(4, 6))
  # Devices are interned, so the wrapped output carries the very object.
  expect_identical(r$device, dev0)

  # An input on another device is copied to the target rather than rejected,
  # and the device is not part of the key: one entry serves both.
  y1 <- parr(pjrt_buffer(c(3, 4), dtype = "f32", device = "cpu:1"))
  expect_equal(out(dispatch(d, list(x = x0, y = y1))), c(4, 6))
  expect_equal(dispatcher_size(d), 1L)
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
    d <- new_dispatcher(
      4L,
      function(info) {
        pjrt_entry(
          pjrt_compile(pjrt_program(src = src)),
          out_avals = list(oav(dtype = spec_dtype)),
          phantom_specs = list(list(dtype = spec_dtype, shape = 2L))
        )
      },
      "flag",
      "pjrt",
      "pjrt",
      FALSE,
      test_pjrt_device
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
    function(info) pjrt_entry(exec),
    static = "flag",
    default_device = test_pjrt_device
  )
  x <- parr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- parr(pjrt_buffer(c(3, 4), dtype = "f32"))
  expect_equal(out(dispatch(d, list(x = x, y = y))), c(4, 6))
})

test_that("`dispatcher()` picks the engine from the backend", {
  cb <- function(info) stop("must not reach the compile callback")
  # The default backend is the native PJRT fast path, which reads a leaf's
  # metadata off its buffer and so needs no extractor.
  expect_s3_class(
    dispatcher(10L, cb, default_device = test_pjrt_device),
    "Dispatcher"
  )
  # Any other backend runs through the closure engine, which has nothing to read
  # a leaf's metadata with unless it is given an extractor.
  expect_error(
    dispatcher(10L, cb, backend = "quickr", default_device = test_quickr_device),
    "is required for a non-.*pjrt.* backend"
  )
  expect_s3_class(
    dispatcher(
      10L,
      cb,
      backend = "quickr",
      default_device = test_quickr_device,
      extractor = test_extractor
    ),
    "Dispatcher"
  )
})

test_that("the closure engine passes the dynamic leaves to `r_fun`, and nothing else", {
  n_miss <- 0L
  d <- new_dispatcher(
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
  d2 <- new_dispatcher(
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
  d <- new_dispatcher(
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
  d <- new_dispatcher(
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
    NULL # move_inputs: no `default_device` resolver needed
  )

  # inputs spread across devices: an error without move_inputs, fine here
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
  # its `$device`'s canonical representative. The same object is its own
  # canonical (a pointer compare); an equal-but-distinct one collapses to it via
  # identical(). The closure engine uses this base canonicalization; pjrt's own
  # engine interns by PJRT_Device* instead.
  n_miss <- 0L
  mk <- function() {
    n_miss <<- 0L
    new_dispatcher(
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

  # The same object across calls -> the identity fast path -> one key.
  dev <- test_device("cpu")
  d <- mk()
  invisible(impl_dispatch_run(d, list(x = qarr(c(1, 2), device = dev))))
  invisible(impl_dispatch_run(d, list(x = qarr(c(1, 2), device = dev))))
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

  # Equal-but-distinct device objects (a backend that hands out fresh ones, as
  # quickr does) collapse to one canonical device via identical(), within a call
  # and across calls.
  expect_true(identical(test_device("cpu"), test_device("cpu")))
  d2 <- mk()
  invisible(impl_dispatch_run(
    d2,
    list(x = qarr(c(1, 2), device = test_device("cpu")), y = qarr(c(1, 2), device = test_device("cpu")))
  ))
  invisible(impl_dispatch_run(
    d2,
    list(x = qarr(c(1, 2), device = test_device("cpu")), y = qarr(c(1, 2), device = test_device("cpu")))
  ))
  expect_equal(n_miss, 1L) # one device, one entry, however it is spelled
  # ...while genuinely different devices still conflict.
  expect_error(
    impl_dispatch_run(
      d2,
      list(x = qarr(c(1, 2), device = test_device("cpu")), y = qarr(c(1, 2), device = test_device("gpu")))
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
  d <- new_dispatcher(
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
  d <- new_dispatcher(
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
    new_dispatcher(
      10L,
      function(info) {
        n_miss <<- n_miss + 1L
        list(r_fun = function(flat) list(v = flat[[1]]))
      },
      character(0),
      engine,
      if (engine == "pjrt") "pjrt" else "quickr",
      FALSE,
      test_pjrt_device
    )
  }

  # An AnvlArray of the wrong backend for the dispatcher.
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = qarr(c(1, 2), backend = "pjrt"))),
    "invalid input `x`.*\"quickr\""
  )
  expect_error(
    impl_dispatch_run(mk("pjrt"), list(x = qarr(c(1, 2)))),
    "invalid input `x`.*\"pjrt\""
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
  # tengen names more dtypes than the dispatcher can represent; this is a real
  # one it cannot key.
  weird <- tengen::as_dtype("f16")
  expect_error(
    impl_dispatch_run(mk("closure"), list(x = qarr(c(1, 2), dtype = weird))),
    "invalid input `x`.*dtype is not one anvl can represent"
  )

  expect_equal(n_miss, 0L) # every rejection happened before the cache was probed
})

test_that("a closure backend can compute metadata via accessors, storing no fields", {
  # anvl's AnvlBackend contract guarantees only $data on a leaf; dtype/shape/
  # device/backend may be computed by the backend's accessors rather
  # than stored as fields. The dispatcher must read them through the extractor,
  # never by reaching for fields -- this array carries $data and nothing else.
  n_miss <- 0L
  dev <- test_device("cpu")
  extractor <- function(leaf) {
    list(
      aval = list(dtype = tengen::as_dtype("f64"), shape = length(leaf$data)),
      device = dev,
      backend = "quickr"
    )
  }
  d <- impl_dispatcher_create(
    10L,
    function(info) {
      n_miss <<- n_miss + 1L
      list(r_fun = function(flat) list(v = flat[[1]] * 2))
    },
    character(0),
    "closure",
    "quickr",
    FALSE,
    test_quickr_device,
    extractor
  )
  bare <- function(v) structure(list(data = v), class = "AnvlArray")
  expect_identical(impl_dispatch_run(d, list(bare(c(1, 2, 3))))$v, c(2, 4, 6))
  # A second call of the same accessor-derived aval is a cache hit.
  invisible(impl_dispatch_run(d, list(bare(c(5, 6, 7)))))
  expect_equal(n_miss, 1L)
})

test_that("a capacity below 1 is rejected rather than segfaulting", {
  expect_error(
    new_dispatcher(0L, function(info) list(), character(0), "closure", "quickr", FALSE, test_quickr_device),
    "capacity"
  )
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
    new_dispatcher(
      10L,
      function(info) pjrt_entry(exec, out_tree = out_tree, out_avals = out_avals),
      character(0),
      "pjrt",
      "pjrt",
      FALSE,
      test_pjrt_device
    )
  }
  x <- parr(pjrt_buffer(c(1, 2), dtype = "f32"))
  y <- parr(pjrt_buffer(c(3, 4), dtype = "f32"))

  # Every wrapped field comes from the declared aval, not from the buffer: the
  # first output is stamped f64 although the buffer it holds is f32.
  d <- mk(
    build_tree(list(sum = 0, rest = list(prod = 0))),
    list(oav("f64"), oav())
  )
  res <- impl_dispatch_run(d, list(x, y))
  expect_equal(out(res$sum), c(4, 6))
  expect_equal(out(res$rest$prod), c(3, 8))
  expect_identical(res$sum$dtype, tengen::as_dtype("f64"))
  expect_identical(res$rest$prod$dtype, tengen::as_dtype("f32"))
  expect_identical(res$sum$shape, 2L)

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
    new_dispatcher(10L, entry_fn, character(0), "pjrt", "pjrt", FALSE, test_pjrt_device)
  }
  x <- parr(pjrt_buffer(c(1, 2), dtype = "f32"))

  # A missing client (needed for uploads, phantoms, and the wrap's device) is a
  # clear error, not a crash at input-assembly time.
  d_bad <- mk(function(info) {
    list(exec = exec, device = pjrt_device("cpu:0"), out_tree = build_tree(0))
  })
  expect_error(impl_dispatch_run(d_bad, list(x)), "must return `client`")

  # So is a const_arrays element that is not a PJRTBuffer: execute would
  # reinterpret the external pointer blindly and segfault, so it must be
  # rejected when the entry is built (here: the exec itself, a plausible slip).
  d_bad2 <- mk(function(info) pjrt_entry(exec, const_arrays = list(exec)))
  expect_error(
    impl_dispatch_run(d_bad2, list(x)),
    "const_arrays\\[\\[1\\]\\]` must be a PJRTBuffer"
  )
})

test_that("a handle that is not a Dispatcher is rejected, not reinterpreted", {
  skip_if_not(plugins_downloaded())
  # Rcpp's XPtr conversion checks only that the SEXP is an external pointer, so
  # without an explicit class check these would reinterpret a PJRTBuffer as a
  # Dispatcher* and read arbitrary memory -- returning a plausible wrong answer
  # rather than erroring.
  buf <- pjrt_buffer(c(1, 2), dtype = "f32")
  expect_error(dispatcher_size(buf), "expected a `Dispatcher`")
  expect_error(dispatch(buf, list(1)), "expected a `Dispatcher`")
  expect_error(dispatcher_size(build_tree(0)), "expected a `Dispatcher`")
})

test_that("a default_device resolver must return a PJRTDevice", {
  skip_if_not(plugins_downloaded())
  # The resolver is the backend's own R code, and its result is dereferenced as
  # a PJRTDevice. A foreign external pointer would otherwise have its garbage
  # PJRT_Device* cached and handed to uploads and execution.
  d <- new_dispatcher(
    4L,
    function(info) stop("must not reach the compile callback"),
    character(0),
    "pjrt",
    "pjrt",
    FALSE,
    function() pjrt_client("cpu") # a PJRTClient, not a PJRTDevice
  )
  expect_error(
    impl_dispatch_run(d, list(1)),
    "`default_device` resolver must return a PJRTDevice"
  )
})

test_that("an AnvlArray that is not a list is rejected, naming the argument", {
  skip_if_not(plugins_downloaded())
  # anvl_field() reads a leaf with VECTOR_ELT, so a value carrying the class but
  # not the type must fall through to the core's documented rejection rather
  # than reaching VECTOR_ELT and raising R's low-level type error.
  d <- new_dispatcher(
    4L,
    function(info) stop("must not reach the compile callback"),
    character(0),
    "pjrt",
    "pjrt",
    FALSE,
    test_pjrt_device
  )
  bad <- structure(c(data = 1), class = "AnvlArray")
  expect_error(impl_dispatch_run(d, list(x = bad)), "invalid input `x`")
})

# ---------------------------------------------------------------------------
# `input_dtypes`: the dtype an execute-time input is supplied at.
# ---------------------------------------------------------------------------

test_that("`input_dtypes` decides the dtype a bare R leaf is uploaded at", {
  skip_if_not(plugins_downloaded())
  # An f64 program: without `input_dtypes` the bare R double would arrive as
  # f32 and the executable would refuse it, and -- the point of the mechanism
  # -- the value would already have been rounded to f32 on the way in.
  src <- 'func.func @main(%x: tensor<f64>) -> tensor<f64> {
    "func.return"(%x): (tensor<f64>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = src))
  entry <- function(dtypes) {
    function(info) {
      pjrt_entry(
        exec,
        out_tree = build_tree(0),
        out_avals = list(oav("f64", integer())),
        input_dtypes = dtypes
      )
    }
  }
  d <- dispatcher(10L, entry("f64"), default_device = test_pjrt_device)
  res <- dispatch(d, list(x = sqrt(2)))
  # Exact to the last bit: the R double was uploaded as f64, not widened from f32.
  expect_identical(as.numeric(tengen::as_array(await(res$data))), sqrt(2))

  # The same call keys the same entry whatever the value, so a second value
  # is served by the entry compiled for the first one.
  expect_identical(as.numeric(tengen::as_array(await(dispatch(d, list(x = pi))$data))), pi)
  expect_equal(dispatcher_size(d), 1L)

  # There is no default to fall back on: an entry that declares nothing for a
  # bare R input is a malformed result, not a licence to guess f32.
  d2 <- dispatcher(10L, entry(NULL), default_device = test_pjrt_device)
  expect_error(
    dispatch(d2, list(x = sqrt(2))),
    "`input_dtypes` is required, because input 1 is bare R data"
  )
  d3 <- dispatcher(10L, entry(NA_character_), default_device = test_pjrt_device)
  expect_error(
    dispatch(d3, list(x = sqrt(2))),
    "`input_dtypes\\[\\[1\\]\\]` is NA for a bare R input"
  )
})

test_that("a malformed `input_dtypes` is rejected, not silently ignored", {
  skip_if_not(plugins_downloaded())
  src <- 'func.func @main(%x: tensor<f64>) -> tensor<f64> {
    "func.return"(%x): (tensor<f64>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = src))
  cb <- function(dtypes) {
    function(info) {
      pjrt_entry(exec, out_avals = list(oav("f64", integer())), input_dtypes = dtypes)
    }
  }
  expect_error(
    dispatch(dispatcher(10L, cb(c("f64", "f64")), default_device = test_pjrt_device), list(x = 1)),
    "2 entries but the call supplies 1"
  )
  expect_error(
    dispatch(dispatcher(10L, cb("f16"), default_device = test_pjrt_device), list(x = 1)),
    "not a dtype anvl can represent"
  )
  expect_error(
    dispatch(dispatcher(10L, cb(64), default_device = test_pjrt_device), list(x = 1)),
    "must be a character vector"
  )
})

test_that("a bare R input needs its dtype declared, whatever else the call passes", {
  skip_if_not(plugins_downloaded())
  # The requirement is per input: an array alongside it still takes NA, and the
  # rejection names the bare R one by its position among the call's inputs.
  src <- 'func.func @main(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.add"(%a, %b): (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%0): (tensor<f32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = src))
  cb <- function(dtypes) {
    function(info) {
      pjrt_entry(
        exec,
        out_avals = list(oav("f32", integer())),
        input_dtypes = dtypes
      )
    }
  }
  args <- list(a = parr(pjrt_scalar(1, dtype = "f32")), b = 2)
  expect_error(
    dispatch(dispatcher(10L, cb(NULL), default_device = test_pjrt_device), args),
    "`input_dtypes` is required, because input 2 is bare R data"
  )
  expect_error(
    dispatch(dispatcher(10L, cb(rep(NA_character_, 2L)), default_device = test_pjrt_device), args),
    "`input_dtypes\\[\\[2\\]\\]` is NA for a bare R input"
  )
  d <- dispatcher(10L, cb(c(NA, "f32")), default_device = test_pjrt_device)
  expect_equal(out(dispatch(d, args)), 3)
})

test_that("`input_dtypes` may not name a dtype the R value cannot upload at", {
  skip_if_not(plugins_downloaded())
  # Each R storage type is uploaded through the buffer entry point for its
  # SEXPTYPE, and those do not all reach every dtype. The pair is settled on
  # the compile path, against the `input_dtypes` entry the callback must fix,
  # rather than surfacing from the buffer layer mid-execution.
  cb <- function(dtypes, ty) {
    src <- sprintf(
      'func.func @main(%%x: tensor<%s>) -> tensor<%s> {
        "func.return"(%%x): (tensor<%s>) -> ()
      }',
      ty,
      ty,
      ty
    )
    exec <- pjrt_compile(pjrt_program(src = src))
    function(info) {
      pjrt_entry(exec, out_avals = list(oav(ty, integer())), input_dtypes = dtypes)
    }
  }
  reject <- function(x, dtypes, ty) {
    d <- dispatcher(10L, cb(dtypes, ty), default_device = test_pjrt_device)
    expect_error(dispatch(d, list(x = x)), "cannot be uploaded at: it is")
  }
  # An R integer has no path to bool, and an R logical no path to anything else.
  reject(1L, "bool", "i1")
  reject(TRUE, "i32", "i32")
  reject(TRUE, "f32", "f32")

  # The pairs that do work still do, including the ones an R double reaches
  # only by conversion.
  accept <- function(x, dtype, ty, expected) {
    d <- dispatcher(10L, cb(dtype, ty), default_device = test_pjrt_device)
    expect_equal(
      as.vector(tengen::as_array(await(dispatch(d, list(x = x))$data))),
      expected
    )
  }
  accept(TRUE, "bool", "i1", TRUE)
  accept(3L, "f64", "f64", 3)
  accept(2.5, "f32", "f32", 2.5)
  accept(1, "bool", "i1", TRUE)
  accept(7, "i32", "i32", 7L)
})

test_that("`input_dtypes` may not declare a dtype for an array input", {
  skip_if_not(plugins_downloaded())
  # An array is supplied as it is -- nothing uploads it -- so a declared dtype
  # could not take effect, and is rejected rather than silently ignored.
  src <- 'func.func @main(%x: tensor<f32>) -> tensor<f32> {
    "func.return"(%x): (tensor<f32>) -> ()
  }'
  exec <- pjrt_compile(pjrt_program(src = src))
  cb <- function(dtypes) {
    function(info) {
      pjrt_entry(
        exec,
        out_avals = list(oav("f32", integer())),
        input_dtypes = dtypes
      )
    }
  }
  arr <- parr(pjrt_scalar(1, dtype = "f32"))
  expect_error(
    dispatch(dispatcher(10L, cb("f32"), default_device = test_pjrt_device), list(x = arr)),
    "declares dtype \"f32\" for an array input"
  )
  # NA is the entry an array takes, and leaves the buffer alone.
  d <- dispatcher(10L, cb(NA_character_), default_device = test_pjrt_device)
  expect_identical(
    as.numeric(tengen::as_array(await(dispatch(d, list(x = arr))$data))),
    1
  )
})

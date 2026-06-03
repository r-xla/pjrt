# Invariants for the CPU zero-copy buffer + donation keepalive model.
#
# Every CPU buffer is backed by a RAWSXP held in the XPtr's protected
# slot. When an executable's input is donated to an output, pjrt_execute
# must migrate that RAWSXP from the donated input XPtr to the aliased
# output XPtr so the underlying bytes stay alive for the output's
# lifetime. These tests pin down both invariants.

# Helper: read the protected SEXP off an external pointer. Defined here
# rather than exporting from the package, since it's only useful for
# poking at the keepalive invariant in tests.
xptr_prot <- local({
  fn <- NULL
  function(x) {
    if (is.null(fn)) {
      fn <<- Rcpp::cppFunction(
        'SEXP xptr_prot_impl(SEXP x) { return R_ExternalPtrProtected(x); }'
      )
    }
    fn(x)
  }
})

vcells_mb <- function() {
  g <- gc(full = TRUE, verbose = FALSE)
  g["Vcells", "(Mb)"]
}

test_that("pjrt_buffer keeps its backing RAWSXP in the XPtr's prot slot", {
  skip_if(!is_cpu())
  nfloats <- 1024L * 256L
  b <- pjrt_buffer(matrix(1.25, nfloats, 1), dtype = "f32")
  p <- xptr_prot(b)
  expect_true(is.raw(p))
  expect_equal(length(p), nfloats * 4L)
})

test_that("pjrt_empty keeps its backing RAWSXP in the XPtr's prot slot", {
  skip_if(!is_cpu())
  e <- pjrt_empty("f32", c(256L, 256L))
  p <- xptr_prot(e)
  expect_true(is.raw(p))
  expect_equal(length(p), 256L * 256L * 4L)
})

test_that("RAWSXPs survive GC pressure while the XPtr is reachable", {
  skip_if(!is_cpu())
  bufs <- lapply(1:8, function(i) {
    pjrt_buffer(matrix(0.5, 1024L * 1024L, 1), dtype = "f32")
  })
  for (i in 1:5) {
    invisible(rnorm(1e5))
    gc(full = TRUE)
  }
  expect_true(all(as.numeric(as_array(bufs[[1]])) == 0.5))
})

test_that("RAWSXPs are reclaimed when XPtrs go out of scope", {
  skip_if(!is_cpu())
  gc(full = TRUE); gc(full = TRUE)
  before <- vcells_mb()

  bufs <- lapply(1:20, function(i) {
    pjrt_buffer(matrix(0.5, 1024L * 1024L, 1), dtype = "f32")
  })
  during <- vcells_mb()
  # 20 buffers x 4 MB f32 = ~80 MB. Allow some slack for accounting noise.
  expect_gt(during - before, 70)

  rm(bufs)
  gc(full = TRUE); gc(full = TRUE)
  after <- vcells_mb()
  expect_lt(abs(after - before), 5)
})

test_that("compiled executable exposes input_output_alias entries", {
  skip_if(!is_cpu())
  mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
  prog <- pjrt_program(src = mlir, format = "mlir")
  exec <- pjrt_compile(prog, device = "cpu")
  aliases <- impl_loaded_executable_aliases(exec)
  expect_equal(aliases$input, 0L)
  expect_equal(aliases$output, 0L)
})

test_that("pjrt_execute transfers keepalive from donated input to aliased output", {
  skip_if(!is_cpu())
  mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
  prog <- pjrt_program(src = mlir, format = "mlir")
  exec <- pjrt_compile(prog, device = "cpu")
  x <- pjrt_buffer(c(10, 20, 30, 40), dtype = "f32")
  x_prot <- xptr_prot(x)
  expect_true(is.raw(x_prot))

  out <- pjrt_execute(exec, x)

  # Output's prot slot now holds the input's RAWSXP; input's is cleared.
  expect_identical(xptr_prot(out), x_prot)
  expect_null(xptr_prot(x))

  # Drop the input and stress the GC. The output's bytes must remain
  # readable — they live in the RAWSXP that the output's prot now pins.
  rm(x)
  for (i in 1:5) {
    invisible(rnorm(1e5))
    gc(full = TRUE)
  }
  expect_equal(as.numeric(as_array(out)), c(20, 40, 60, 80), tolerance = 1e-6)
})

test_that("operations on a donated input raise a clean R-level error", {
  skip_if(!is_cpu())
  mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
  prog <- pjrt_program(src = mlir, format = "mlir")
  exec <- pjrt_compile(prog, device = "cpu")
  x <- pjrt_buffer(c(1, 2, 3, 4), dtype = "f32")
  pjrt_execute(exec, x)
  expect_error(as_array(x), "called on deleted or donated buffer")
})

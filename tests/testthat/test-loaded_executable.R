test_that("arguments must be unnamed", {
  skip_if_metal("only supports MLIR programs")
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")
  executable <- pjrt_compile(program)
  buf <- pjrt_buffer(1)
  expect_error(pjrt_execute(executable, a = buf, "Expected unnamed arguments"))
})

test_that("execute program without arguments", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_equal(as_array(result), 3)
})

test_that("can return two values", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_list(result, types = "PJRTBuffer", len = 2L)
  expect_equal(as_array(result[[1]]), 3)
  expect_equal(as_array(result[[2]]), 7)
})

test_that("single-output returns list when simplify=FALSE", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable, simplify = FALSE)
  expect_list(result, types = "PJRTBuffer", len = 1L)
  expect_equal(as_array(result[[1]]), 3)

  result <- pjrt_execute(executable, simplify = TRUE)
  expect_class(result, "PJRTBuffer")
  expect_equal(as_array(result), 3)
})

test_that("can execute empty constant", {
  path <- system.file("programs/stablehlo-empty-constant.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_equal(as_array(result), array(integer(), 0L))
})

test_that("print works", {
  path <- system.file("programs/stablehlo-empty-constant.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  expect_snapshot(executable)
})

# Async execution tests

test_that("pjrt_execute returns buffer promise", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  expect_class(result, "PJRTBuffer")
})

test_that("pjrt_execute returns a buffer promise", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  expect_s3_class(result, "PJRTBuffer")
})

test_that("is_ready works for async values", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  # is_ready should return logical
  ready <- is_ready(result)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("await() returns correct result for async execution", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  output <- await(result)
  expect_class(output, "PJRTBuffer")
  expect_equal(as_array(output), 3)
})

test_that("async execution with multiple outputs", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # With multiple outputs, returns list of buffer promises
  result <- pjrt_execute(executable)
  expect_list(result, types = "PJRTBuffer", len = 2L)

  # Each buffer can be awaited individually
  buf1 <- await(result[[1]])
  buf2 <- await(result[[2]])
  expect_class(buf1, "PJRTBuffer")
  expect_class(buf2, "PJRTBuffer")
  expect_equal(as_array(buf1), 3)
  expect_equal(as_array(buf2), 7)
})

test_that("async execution with simplify=FALSE", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # With simplify=FALSE, returns list even for single output
  result <- pjrt_execute(executable, simplify = FALSE)
  expect_list(result, types = "PJRTBuffer", len = 1L)

  # The buffer contains a single buffer
  buf <- await(result[[1]])
  expect_class(buf, "PJRTBuffer")
})

test_that("print.PJRTBuffer works", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  expect_output(print(result), "PJRTBuffer")
})

test_that("as_array works for async values (single output)", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  arr <- as_array(result)
  expect_equal(arr, 3)
})

test_that("as_array works for async values (multiple outputs)", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # With multiple outputs, get list of async values
  result <- pjrt_execute(executable)
  expect_list(result, len = 2L)

  # as_array on each async value
  arr1 <- as_array(result[[1]])
  arr2 <- as_array(result[[2]])
  expect_equal(arr1, 3)
  expect_equal(arr2, 7)
})

test_that("async execution chained with async buffer-to-host", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # Start async execution
  async_result <- pjrt_execute(executable)
  expect_class(async_result, "PJRTBuffer")

  # Chain with async buffer-to-host transfer (auto-waits for execution)
  async_array <- as_array_async(async_result)
  expect_class(async_array, "PJRTArrayPromise")

  # Check is_ready returns logical
  ready <- is_ready(async_array)
  expect_true(is.logical(ready))

  # Get the final R value (this waits for transfer)
  arr <- value(async_array)
  expect_equal(arr, 3)
})

test_that("async execution with inputs chained to async buffer-to-host", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  path <- system.file("programs/jax-stablehlo-subset-2d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # Create input buffers
  x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  x_buf <- pjrt_buffer(x)
  i1_buf <- pjrt_scalar(0L, "i32")
  i2_buf <- pjrt_scalar(1L, "i32")

  # Execute asynchronously
  async_result <- pjrt_execute(executable, x_buf, i1_buf, i2_buf)
  expect_class(async_result, "PJRTBuffer")

  # Chain with async buffer-to-host transfer
  async_array <- as_array_async(async_result)
  expect_class(async_array, "PJRTArrayPromise")

  # Get final value
  result <- value(async_array)
  expect_equal(result, x[1, 2]) # 0-indexed: x[0+1, 1+1] = x[1, 2] = 3
})


# Async chain tests ---------------------------------------------------------

test_that("async chain: buffer -> execute -> as_array", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("longer async chains produce correct results", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  exec1 <- pjrt_compile(pjrt_program(src))
  exec2 <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result1 <- pjrt_execute(exec1, input)
  result2 <- pjrt_execute(exec2, result1)
  arr_promise <- as_array_async(result2)

  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("is_ready works on buffer and array promises", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  ready <- is_ready(arr_promise)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)

  value(arr_promise)
  expect_true(is_ready(arr_promise))
})

test_that("multiple inputs work correctly", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  src <- r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  x <- pjrt_buffer(matrix(1:4, 2, 2), dtype = "f32")
  y <- pjrt_buffer(matrix(5:8, 2, 2), dtype = "f32")

  result <- pjrt_execute(executable, x, y)

  arr <- as_array(result)
  expect_equal(as.vector(arr), as.vector(matrix(1:4, 2, 2) + matrix(5:8, 2, 2)), tolerance = 1e-6)
})

# Error propagation tests ---------------------------------------------------

test_that("error in execute_async is caught when calling value on buffer_promise", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Wrong shape input - should fail
  wrong_input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # The error should be caught during execute_async (input validation)
  # CPU says "size", Metal says "shape"
  expect_error(
    pjrt_execute(executable, wrong_input),
    "size|shape"
  )
})

test_that("error in execute_async is caught when calling value on array_promise", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Create valid input first, then test with wrong shape
  wrong_input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # Error should be caught at execute time
  # CPU says "size", Metal says "shape"
  expect_error(
    pjrt_execute(executable, wrong_input),
    "size|shape"
  )
})

test_that("async chain with buffer_promise produces correct results", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("value() caches result on repeated calls", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)

  # Call value again - should return cached result
  arr2 <- value(arr_promise)
  expect_identical(arr, arr2)
})

# Tests documenting error behavior in async operations ----------------------

test_that("async errors: input validation errors appear at execute time", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  wrong_shape_buffer <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # CPU says "size", Metal says "shape"
  expect_error(
    pjrt_execute(executable, wrong_shape_buffer),
    "size|shape"
  )
})

test_that("async errors: chained operations produce correct results", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%0): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(2.0, 4.0, 6.0), tolerance = 1e-6)
})

test_that("async errors: is_ready works on array promises", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  ready <- is_ready(arr_promise)
  expect_true(is.logical(ready))

  value(arr_promise)
  expect_true(is_ready(arr_promise))
})

test_that("async errors: error messages are descriptive", {
  src <- r"(
func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
  "func.return"(%x): (tensor<4xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  wrong_buffer <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  expect_error(
    pjrt_execute(executable, wrong_buffer),
    regexp = "size|shape",
    ignore.case = TRUE
  )
})

test_that("async errors: OOM during execution is caught at pjrt_execute() time", {
  skip_if(!is_cuda(), "OOM test only meaningful on GPU")

  # Program that broadcasts a scalar to an enormous tensor (~40GB for f32).
  # The input is tiny, so buffer creation succeeds. The OOM happens when
  # the CUDA backend tries to allocate the output buffer during execution.
  # On GPU, this is caught synchronously by PJRT_LoadedExecutable_Execute_.
  src <- r"(
func.func @main(%x: tensor<f32>) -> tensor<100000x100000xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%x) {
    broadcast_dimensions = array<i64>
  } : (tensor<f32>) -> tensor<100000x100000xf32>
  "func.return"(%0): (tensor<100000x100000xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))
  input <- pjrt_scalar(1.0, dtype = "f32")

  expect_error(pjrt_execute(executable, input), "Out of memory")
})

test_that("async errors: CPU backend clamps out-of-bounds indices (no runtime error)", {
  # This test documents that the CPU backend is robust and doesn't error
  # on out-of-bounds access - it clamps indices instead. This is XLA behavior.
  # Metal/GPU backends may behave differently (return zeros or different values).
  skip_if_metal("Metal handles out-of-bounds differently")

  src <- r"(
func.func @main(%x: tensor<3xf32>, %idx: tensor<i32>) -> tensor<2xf32> {
  %0 = "stablehlo.dynamic_slice"(%x, %idx) {
    slice_sizes = array<i64: 2>
  } : (tensor<3xf32>, tensor<i32>) -> tensor<2xf32>
  "func.return"(%0): (tensor<2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  x <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  # Index 5 would be out of bounds, but XLA clamps it
  idx <- pjrt_scalar(5L, "i32")

  result <- pjrt_execute(executable, x, idx)
  arr_promise <- as_array_async(result)

  # No error - XLA clamps the index to valid range
  arr <- value(arr_promise)
  # Result is the last 2 elements (index clamped to 1)
  expect_equal(as.vector(arr), c(2.0, 3.0), tolerance = 1e-6)
})

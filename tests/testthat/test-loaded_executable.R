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

test_that("pjrt_execute_async returns async value", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  expect_class(result, "pjrt_async_value")
})

test_that("is_ready works for async values", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  # is_ready should return logical
  ready <- is_ready(result)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("value() returns correct result for async execution", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  output <- value(result)
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

  # With multiple outputs, returns list of async values
  result <- pjrt_execute_async(executable)
  expect_list(result, types = "pjrt_async_value", len = 2L)

  # Each async value can be awaited individually
  buf1 <- value(result[[1]])
  buf2 <- value(result[[2]])
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
  result <- pjrt_execute_async(executable, simplify = FALSE)
  expect_list(result, types = "pjrt_async_value", len = 1L)

  # The async value contains a single buffer
  buf <- value(result[[1]])
  expect_class(buf, "PJRTBuffer")
})

test_that("print.pjrt_async_value works", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  expect_output(print(result), "pjrt_async_value")
})

test_that("as_array works for async values (single output)", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
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
  result <- pjrt_execute_async(executable)
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
  async_result <- pjrt_execute_async(executable)
  expect_class(async_result, "pjrt_async_value")

  # Chain with async buffer-to-host transfer (auto-waits for execution)
  async_array <- as_array_async(async_result)
  expect_class(async_array, "pjrt_async_buffer")

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
  async_result <- pjrt_execute_async(executable, x_buf, i1_buf, i2_buf)
  expect_class(async_result, "pjrt_async_value")

  # Chain with async buffer-to-host transfer
  async_array <- as_array_async(async_result)
  expect_class(async_array, "pjrt_async_buffer")

  # Get final value
  result <- value(async_array)
  expect_equal(result, x[1, 2])  # 0-indexed: x[0+1, 1+1] = x[1, 2] = 3
})

test_that("async execution with multiple outputs chained to async transfer", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # Execute asynchronously - returns list of async values
  result <- pjrt_execute_async(executable)
  expect_list(result, len = 2L)

  # Chain each output with async buffer-to-host transfer
  async_arr1 <- as_array_async(result[[1]])
  async_arr2 <- as_array_async(result[[2]])

  expect_class(async_arr1, "pjrt_async_buffer")
  expect_class(async_arr2, "pjrt_async_buffer")

  # Get final values
  arr1 <- value(async_arr1)
  arr2 <- value(async_arr2)
  expect_equal(arr1, 3)
  expect_equal(arr2, 7)
})

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

  result <- pjrt_execute_async(executable)
  outputs <- value(result)
  expect_list(outputs, types = "PJRTBuffer", len = 2L)
  expect_equal(as_array(outputs[[1]]), 3)
  expect_equal(as_array(outputs[[2]]), 7)
})

test_that("async execution with simplify=FALSE", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable, simplify = FALSE)
  outputs <- value(result)
  expect_list(outputs, types = "PJRTBuffer", len = 1L)
})

test_that("print.pjrt_async_value works", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  expect_output(print(result), "pjrt_async_value")
})

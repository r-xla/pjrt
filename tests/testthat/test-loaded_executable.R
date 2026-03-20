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
  expect_class(result, "PJRTBuffer")
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

test_that("wrong shape input raises error", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  wrong_input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # CPU says "size", Metal says "shape"
  expect_error(
    pjrt_execute(executable, wrong_input),
    "size|shape"
  )
})

# is_ready / await / value -------------------------------------------------

test_that("is_ready works on buffers and array promises", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)

  ready <- is_ready(result)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)

  arr_promise <- as_array_async(result)
  ready <- is_ready(arr_promise)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("await blocks until buffer is ready", {
  src <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(matrix(runif(1e6), 1000, 1000), dtype = "f32")
  result <- pjrt_execute(executable, input)
  buf <- await(result)
  expect_class(buf, "PJRTBuffer")
  expect_true(is_ready(buf))
})

test_that("value() materializes R array from promise", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  arr <- value(as_array_async(result))
  expect_equal(arr, 3)
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

# Chaining ------------------------------------------------------------------

test_that("chained execution produces correct results", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%0): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr <- value(as_array_async(result))
  expect_equal(as.vector(arr), c(2.0, 4.0, 6.0), tolerance = 1e-6)
})

test_that("longer chains produce correct results", {
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
  arr <- value(as_array_async(result2))
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("execution with inputs chained to buffer-to-host", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  path <- system.file("programs/jax-stablehlo-subset-2d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  x_buf <- pjrt_buffer(x)
  i1_buf <- pjrt_scalar(0L, "i32")
  i2_buf <- pjrt_scalar(1L, "i32")

  async_result <- pjrt_execute(executable, x_buf, i1_buf, i2_buf)
  result <- value(as_array_async(async_result))
  expect_equal(result, x[1, 2]) # 0-indexed: x[0+1, 1+1] = x[1, 2] = 3
})

test_that("print works on not-yet-ready buffer", {
  src <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(matrix(runif(1e6), 1000, 1000), dtype = "f32")
  result <- pjrt_execute(executable, input)
  expect_output(print(result), "PJRTBuffer")
})

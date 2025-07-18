test_that("execution options can be created and configured", {
  options <- pjrt_execution_options()
  expect_true(inherits(options, "PJRTExecuteOptions"))

  options <- pjrt_execution_options(
    non_donatable_input_indices = c(0L, 2L),
    launch_id = 42L
  )
  expect_true(inherits(options, "PJRTExecuteOptions"))
})

test_that("execution with options works", {
  skip_if_metal()

  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")
  executable <- pjrt_compile(program)

  data <- 3.0
  scalar_buffer <- pjrt_scalar(data)

  result1 <- pjrt_execute(executable, scalar_buffer)
  r_res1 <- as_array(result1)
  expect_equal(r_res1, 9)

  options <- pjrt_execution_options()
  result2 <- pjrt_execute(
    executable,
    scalar_buffer,
    execution_options = options
  )
  r_res2 <- as_array(result2)
  expect_equal(r_res2, 9)

  options <- pjrt_execution_options(non_donatable_input_indices = c(0L))
  result3 <- pjrt_execute(
    executable,
    scalar_buffer,
    execution_options = options
  )
  r_res3 <- as_array(result3)
  expect_equal(r_res3, 9)

  options <- pjrt_execution_options(launch_id = 42L)
  result4 <- pjrt_execute(
    executable,
    scalar_buffer,
    execution_options = options
  )
  r_res4 <- as_array(result4)
  expect_equal(r_res4, 9)
})

test_that("can donate input", {
  skip_if_metal()

  program <- program_load(
    system.file("programs/jax-stablehlo-update-param.mlir", package = "pjrt"),
    format = "mlir"
  )
  executable <- pjrt_compile(program)

  exec_options <- pjrt_execution_options(non_donatable_input_indices = 1L)

  x <- pjrt_buffer(as.double(1:1000000))
  grad <- pjrt_buffer(as.double(1:1000000))
  y <- pjrt_execute(executable, x, grad, execution_options = exec_options)
  expect_equal(as_array(y)[1L], 0)
  expect_error(as_array(x), regexp = "called on deleted or donated buffer")
})

test_that("compile program works", {
  skip_if_metal()

  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")

  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  expect_equal(
    client_platform_name(client),
    "cpu"
  )

  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  data <- 3.0
  scalar_buffer <- client_scalar_buffer_from_host(data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- client_buffer_to_host(result)

  expect_equal(r_res, 9)
})

test_that("compile program works", {
  # this won't work on CI currently because it runs on a Mac VM which doesn't support GPU access.
  skip_if_metal()

  path <- system.file("programs/stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  plugin <- plugin_load()
  client <- plugin_client_create(plugin)
  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  image <- array(0, dim = c(28, 28))
  weights <- array(0, dim = c(784, 10))
  bias <- array(0, dim = c(1, 10))

  image_buffer <- pjrt_buffer(image)
  weights_buffer <- pjrt_buffer(weights)
  bias_buffer <- pjrt_buffer(bias)

  result <- loaded_executable_execute(
    executable,
    list(
      image_buffer,
      weights_buffer,
      bias_buffer
    )
  )
  r_res <- client_buffer_to_host(result)

  expect_equal(r_res, matrix(0, ncol = 10, nrow = 1))
})

test_that("can execute mlir program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  client <- default_client()

  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  if (is_metal()) {
    expect_equal(
      client_platform_name(client),
      "metal"
    )
  } else {
    expect_equal(
      client_platform_name(client),
      "cpu"
    )
  }

  data <- 3.0
  scalar_buffer <- pjrt_scalar(data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- client_buffer_to_host(result)

  expect_equal(r_res, 6)
})

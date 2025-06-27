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

  executable <- client_program_compile(client, program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  data <- 3.0
  scalar_buffer <- client_scalar_buffer_from_host(client, data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- client_buffer_to_host(client, result)

  expect_equal(r_res, 9)
})

test_that("compile program works", {
  # this won't work on CI currently because it runs on a Mac VM which doesn't support GPU access.
  skip_if_metal()

  path <- system.file("programs/stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  plugin <- plugin_load()
  client <- plugin_client_create(plugin)
  executable <- client_program_compile(client, program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  image <- array(0, dim = c(28, 28))
  weights <- array(0, dim = c(784, 10))
  bias <- array(0, dim = c(1, 10))

  image_buffer <- client_buffer_from_host(client, image)
  weights_buffer <- client_buffer_from_host(client, weights)
  bias_buffer <- client_buffer_from_host(client, bias)

  result <- loaded_executable_execute(
    executable,
    list(
      image_buffer,
      weights_buffer,
      bias_buffer
    )
  )
  r_res <- client_buffer_to_host(client, result)

  expect_equal(r_res, matrix(0, ncol = 10, nrow = 1))
})

test_that("can create a scalar buffer from host", {
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  data <- 3.0
  scalar_buffer <- client_scalar_buffer_from_host(client, data)

  expect_true(inherits(scalar_buffer, "PJRTBuffer"))

  x <- client_buffer_to_host(client, scalar_buffer)
  expect_equal(x, data)
})

test_that("can create a buffer from host data", {
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  x <- runif(10)
  buffer <- client_buffer_from_host(client, x)
  expect_true(inherits(buffer, "PJRTBuffer"))

  x_res <- client_buffer_to_host(client, buffer)
  expect_equal(x_res, x, tolerance = 1e-5)
})

test_that("can create a buffer from an array", {
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  x <- array(runif(30), dim = c(2, 5, 3))
  buffer <- client_buffer_from_host(client, x)
  expect_true(inherits(buffer, "PJRTBuffer"))

  x_res <- client_buffer_to_host(client, buffer)
  expect_equal(x_res, x, tolerance = 1e-5)
})

test_that("can execute mlir program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  plugin <- plugin_load()
  client <- plugin_client_create(plugin)
  executable <- client_program_compile(client, program)

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
  scalar_buffer <- client_scalar_buffer_from_host(client, data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- client_buffer_to_host(client, result)

  expect_equal(r_res, 6)
})

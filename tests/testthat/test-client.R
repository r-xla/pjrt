test_that("compile program works", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")

  plugin <- plugin_load()
  client <- plugin_client_create(plugin)
  executable <- client_program_compile(client, program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  data <- 3.0
  scalar_buffer <- client_scalar_buffer_from_host(client, data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- client_buffer_to_host(client, result)

  expect_equal(r_res, 9)
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

test_that("can execute mlir program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  plugin <- plugin_load()
  client <- plugin_client_create(plugin)
  executable <- client_program_compile(client, program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  data <- 3.0
  scalar_buffer <- client_scalar_buffer_from_host(client, data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- client_buffer_to_host(client, result)

  expect_equal(r_res, 6)
})

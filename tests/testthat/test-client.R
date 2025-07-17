test_that("compile program with one input", {
  skip_if_metal()

  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")

  plugin <- plugin_load("cpu")
  client <- plugin_client_create(plugin, "cpu")

  check_client_device(client)

  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  data <- 3.0
  scalar_buffer <- pjrt_scalar(data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- as_array(result)

  expect_equal(r_res, 9)
})

test_that("compile program with multiple inputs", {
  # this won't work on CI currently because it runs on a Mac VM which doesn't support GPU access.
  skip_if_metal()

  path <- system.file("programs/stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  plugin <- plugin_load("cpu")
  client <- plugin_client_create(plugin, "cpu")
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
  r_res <- as_array(result)

  expect_equal(r_res, matrix(0, ncol = 10, nrow = 1))
})

test_that("can execute mlir program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  client <- default_client()
  if (!is_metal()) {
    check_client_device(client)
  }

  data <- 3.0
  scalar_buffer <- pjrt_scalar(data)

  result <- loaded_executable_execute(executable, scalar_buffer)
  r_res <- as_array(result)

  expect_equal(r_res, 6)
})

test_that("can use more than one client", {
  skip_if(!is_metal() || !is_cuda())
  is_metal() && skip_on_ci()

  device <- if (is_metal()) "metal" else "cpu"

  pjrt_buffer(1, pjrt_client(device))

  expect_permutation(c(device, "cpu"), names(the$clients))
  expect_permutation(c(device, "cpu"), names(the$plugins))

  # not they are loaded and global env 'the' is not changed
  pjrt_buffer(1, pjrt_client("cpu"))
  pjrt_buffer(1, pjrt_client(device))

  expect_permutation(c(device, "cpu"), names(the$clients))
  expect_permutation(c(device, "cpu"), names(the$plugins))
})

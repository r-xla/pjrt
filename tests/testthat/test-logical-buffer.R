test_that("logical scalar buffer creation works", {
  # Create a plugin and client
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  # Test logical scalar
  logical_scalar <- TRUE
  buffer <- client_scalar_buffer_from_host(client, logical_scalar)

  expect_true(inherits(buffer, "PJRTBuffer"))

  # Test conversion back to host
  result <- client_buffer_to_host(client, buffer)
  expect_equal(result, logical_scalar)
})

test_that("logical vector buffer creation works", {
  # Create a plugin and client
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  # Test logical vector
  logical_vec <- c(TRUE, FALSE, TRUE, FALSE)
  buffer <- client_buffer_from_host(client, logical_vec)

  expect_true(inherits(buffer, "PJRTBuffer"))

  # Test conversion back to host
  result <- client_buffer_to_host(client, buffer)
  expect_equal(result, logical_vec)
})

test_that("logical matrix buffer creation works", {
  # Create a plugin and client
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  # Test logical matrix
  logical_matrix <- matrix(
    c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    nrow = 2,
    ncol = 3
  )
  buffer <- client_buffer_from_host(client, logical_matrix)

  expect_true(inherits(buffer, "PJRTBuffer"))

  # Test conversion back to host
  result <- client_buffer_to_host(client, buffer)
  expect_equal(result, logical_matrix)
})

test_that("mixed numeric and logical buffers work", {
  skip_if_metal()

  # Create a plugin and client
  plugin <- plugin_load()
  client <- plugin_client_create(plugin)

  # Test numeric scalar
  numeric_scalar <- 42.0
  numeric_buffer <- client_scalar_buffer_from_host(client, numeric_scalar)
  numeric_result <- client_buffer_to_host(client, numeric_buffer)
  expect_equal(numeric_result, numeric_scalar)

  # Test logical scalar
  logical_scalar <- TRUE
  logical_buffer <- client_scalar_buffer_from_host(client, logical_scalar)
  logical_result <- client_buffer_to_host(client, logical_buffer)
  expect_equal(logical_result, logical_scalar)

  # Test numeric vector
  numeric_vec <- c(1.0, 2.0, 3.0, 4.0)
  numeric_vec_buffer <- client_buffer_from_host(client, numeric_vec)
  numeric_vec_result <- client_buffer_to_host(client, numeric_vec_buffer)
  expect_equal(numeric_vec_result, numeric_vec)

  # Test logical vector
  logical_vec <- c(TRUE, FALSE, TRUE, FALSE)
  logical_vec_buffer <- client_buffer_from_host(client, logical_vec)
  logical_vec_result <- client_buffer_to_host(client, logical_vec_buffer)
  expect_equal(logical_vec_result, logical_vec)
})

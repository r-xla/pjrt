test_that("can get client by platform name", {
  # Test getting CPU client
  cpu_client <- get_client("cpu")
  expect_true(inherits(cpu_client, "PJRTClient"))
  expect_equal(client_platform_name(cpu_client), "cpu")

  # Test that getting the same platform again returns the same client
  cpu_client2 <- get_client("cpu")
  expect_identical(cpu_client, cpu_client2)
})

test_that("can list platforms", {
  # Clear any existing clients
  clear_plugins_and_clients()

  # Initially no platforms
  expect_equal(list_platforms(), character(0))

  # Get a client
  get_client("cpu")

  # Now should have cpu platform
  expect_equal(list_platforms(), "cpu")
})

test_that("can use platform name in pjrt_compile", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")

  # Test using platform name instead of client object
  executable <- pjrt_compile(program, client = "cpu")
  expect_true(inherits(executable, "PJRTLoadedExecutable"))
})

test_that("can use platform name in pjrt_buffer", {
  # Test using platform name instead of client object
  buffer <- pjrt_buffer(1:10, client = "cpu")
  expect_true(inherits(buffer, "PJRTBuffer"))

  # Test scalar
  scalar <- pjrt_scalar(5, client = "cpu")
  expect_true(inherits(scalar, "PJRTBuffer"))
})

test_that("can use platform name in as_array", {
  buffer <- pjrt_buffer(1:5, client = "cpu")
  result <- as_array(buffer, client = "cpu")
  expect_equal(result, 1:5)
})

test_that("can use platform name in client_platform_name", {
  platform <- client_platform_name("cpu")
  expect_equal(platform, "cpu")
})

test_that("default_client works with platform name", {
  client <- default_client("cpu")
  expect_true(inherits(client, "PJRTClient"))
  expect_equal(client_platform_name(client), "cpu")
})

test_that("clear_plugins_and_clients works", {
  # Get a client first
  get_client("cpu")
  expect_equal(list_platforms(), "cpu")

  # Clear everything
  clear_plugins_and_clients()
  expect_equal(list_platforms(), character(0))

  # Should be able to get client again
  get_client("cpu")
  expect_equal(list_platforms(), "cpu")
})

test_that("plugin_load works with device specification", {
  # Test loading CPU plugin explicitly
  plugin <- plugin_load("cpu")
  expect_true(inherits(plugin, "PJRTPlugin"))

  # Test that loading the same device again returns the same plugin
  plugin2 <- plugin_load("cpu")
  expect_identical(plugin, plugin2)
})

test_that("plugin_client_create works with platform name", {
  plugin <- plugin_load("cpu")

  # Create client with explicit platform name
  client <- plugin_client_create(plugin, "cpu")
  expect_true(inherits(client, "PJRTClient"))
  expect_equal(client_platform_name(client), "cpu")

  # Create client without platform name (should auto-detect)
  client2 <- plugin_client_create(plugin)
  expect_true(inherits(client2, "PJRTClient"))
  expect_equal(client_platform_name(client2), "cpu")
})

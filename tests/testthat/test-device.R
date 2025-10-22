test_that("device specification with index", {
  # In setup.R, we specify 2 CPU devices.
  d1 <- pjrt_device("cpu:0")
  d2 <- pjrt_device("cpu:1")
  expect_true(inherits(d1, "PJRTDevice"))
  expect_true(inherits(d2, "PJRTDevice"))
  expect_equal(as.character(d1), "CpuDevice(id=0)")
  expect_equal(as.character(d2), "CpuDevice(id=1)")
  expect_false(d1 == d2)
})

test_that("first device is the default", {
  expect_true(pjrt_device("cpu") == pjrt_device("cpu:0"))
})

test_that("Out of range index error", {
  expect_error(
    pjrt_device("cpu:99"),
    "Device index 99 out of range"
  )
})

test_that("device identity", {
  x <- pjrt_device("cpu:1")
  expect_equal(x, pjrt_device(x))
})

test_that("devices", {
  expect_list(devices("cpu"), types = "PJRTDevice")
})

test_that("can set cpu_device_count", {
  plugin <- impl_plugin_load(plugin_path("cpu"))
  client <- impl_plugin_client_create(plugin, list(cpu_device_count = 2))
  expect_list(devices(client), types = "PJRTDevice", len = 2L)
})

test_that("device count is like it was setup in setup.R", {
  expect_list(devices("cpu"), types = "PJRTDevice", len = 2L)
})

test_that("== for device", {
  skip_if(!(is_metal() || is_cuda()))
  device_name <- if (is_metal()) "metal" else "cuda"
  expect_false(pjrt_device("cpu") == pjrt_device(device_name))
  expect_true(pjrt_device("cpu") == pjrt_device("cpu"))
  expect_true(pjrt_device(device_name) == pjrt_device(device_name))
})

skip_if_metal <- function() {
  if (is_metal()) {
    testthat::skip("Skipping test on Metal device")
  }
}

is_metal <- function() {
  Sys.getenv("PJRT_DEVICE") == "metal"
}

is_cuda <- function() {
  Sys.getenv("PJRT_DEVICE") == "cuda"
}

check_client_device <- function() {
  device <- Sys.getenv("PJRT_DEVICE", "cpu")
  expect_equal(
    client_platform_name(client),
    device
  )
}

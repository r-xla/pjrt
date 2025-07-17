skip_if_metal <- function() {
  if (is_metal()) {
    testthat::skip("Skipping test on Metal device")
  }
}

is_metal <- function() {
  Sys.getenv("PJRT_PLATFORM") == "metal"
}

is_cuda <- function() {
  Sys.getenv("PJRT_PLATFORM") == "cuda"
}

check_client_device <- function(client) {
  device <- Sys.getenv("PJRT_PLATFORM", "cpu")
  expect_equal(
    tolower(pjrt_platform_name(client)),
    tolower(device)
  )
}

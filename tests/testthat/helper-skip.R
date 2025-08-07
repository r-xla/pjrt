skip_if_metal <- function() {
  if (is_metal()) {
    testthat::skip("Skipping test on Metal device")
  }
}

# GHA does not have access to metal hardware
skip_if_gha_metal <- function() {
  if (is_metal() && is_gha()) {
    testthat::skip("Skipping test on (fake) Metal GHA runner")
  }
}

is_gha <- function() {
  Sys.getenv("GITHUB_ACTIONS") == "true"
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
    tolower(platform_name(client)),
    tolower(device)
  )
}

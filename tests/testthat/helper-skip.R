skip_if_mps <- function(msg = "") {
  if (is_mps()) {
    testthat::skip(sprintf("Skipping test on MPS device: %s", msg))
  }
}

is_cpu <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu") == "cpu"
}

is_mps <- function() {
  Sys.getenv("PJRT_PLATFORM") == "mps"
}

is_cuda <- function() {
  Sys.getenv("PJRT_PLATFORM") == "cuda"
}

check_client_device <- function(client) {
  device <- Sys.getenv("PJRT_PLATFORM", "cpu")
  testthat::expect_equal(
    tolower(platform(client)),
    tolower(device)
  )
}

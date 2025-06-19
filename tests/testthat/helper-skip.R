skip_if_metal <- function() {
  if (is_metal()) {
    testthat::skip("Skipping test on Metal device")
  }
}

is_metal <- function() {
  Sys.getenv("PJRT_DEVICE") == "metal"
}

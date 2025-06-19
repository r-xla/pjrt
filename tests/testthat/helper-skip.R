skip_if_metal <- function() {
  if (Sys.getenv("PJRT_DEVICE") == "metal") {
    testthat::skip("Skipping test on Metal device")
  }
}

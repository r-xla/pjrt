skip_if_metal <- function(msg = "") {
  if (is_metal()) {
    testthat::skip(sprintf("Skipping test on Metal device: %s", msg))
  }
}

# The stablehlo-opt binary is a large (hundreds of MB) separate download, so
# tests that need it only run when it is already cached locally, or when
# PJRT_TEST_STABLEHLO_OPT=1 explicitly opts into downloading it.
skip_if_no_stablehlo_opt <- function() {
  if (Sys.getenv("PJRT_TEST_STABLEHLO_OPT") == "1") {
    return(invisible(NULL))
  }
  if (!stablehlo_opt_available()) {
    testthat::skip("stablehlo-opt is not downloaded")
  }
  invisible(NULL)
}

is_cpu <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu") == "cpu"
}

is_metal <- function() {
  Sys.getenv("PJRT_PLATFORM") == "metal"
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

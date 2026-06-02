#' @title Create a PJRT Device
#' @description
#' Create a PJRT Device from an R object.
#' @section Extractors:
#' * [`platform()`] for a `character(1)` representation of the platform.
#' @param device (any)\cr
#'   The device.
#' @return `PJRTDevice`
#' @examplesIf plugins_downloaded("cpu")
#' # Show available devices for CPU client
#' devices(pjrt_client("cpu"))
#' # Create device 0 for CPU client
#' dev <- pjrt_device("cpu:0")
#' dev
#' @export
pjrt_device <- function(device) {
  if (inherits(device, "PJRTDevice")) {
    return(device)
  }
  if (is.character(device) && length(device) == 1 && nchar(device) > 0) {
    return(as_pjrt_device(device))
  }
  cli_abort("Must be a PJRTDevice or a platform name")
}


#' @include client.R
#' @export
platform.PJRTDevice <- function(x, ...) {
  platform_from_device_string(as.character(x))
}

# Extract the platform name (e.g. "cuda", "cpu") from a PJRT device's
# `to_string` representation (e.g. "CudaDevice(id=0)", "CpuDevice(id=0)"):
# strip the leading [A-Za-z]+ run, drop a trailing "Device", and lowercase.
# Used for both PJRTDevice and PJRTBuffer so they agree with platform.PJRTClient.
platform_from_device_string <- function(desc) {
  letters_only <- regmatches(desc, regexpr("^[A-Za-z]+", desc, perl = TRUE))
  tolower(sub("Device$", "", letters_only))
}

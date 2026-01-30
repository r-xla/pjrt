#' @title Create a PJRT Device
#' @description
#' Create a PJRT Device from an R object.
#' @section Extractors:
#' * [`platform()`] for a `character(1)` representation of the platform.
#' @param device (any)\cr
#'   The device.
#' @return `PJRTDevice`
#' @examplesIf plugin_is_downloaded("cpu")
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
  impl_device_platform(x)
}

#' @title Create a PJRT Device
#' @description
#' Create a PJRT Device from an R object.
#' @param device (any)\cr
#'   The device.
#' @return `PJRTDevice`
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

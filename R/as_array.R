#' Convert a PJRT Buffer to an R Array
#'
#' @description
#' Copy a [`PJRTBuffer`] to an R array.
#' @details
#' Currently, two copies are being done:
#' 1. Copy the data from the PJRT device to the CPU.
#' 2. Copy the data from the CPU to the R array.
#'
#' @template param_buffer
#' @template param_client
#' @export
as_array <- function(buffer, client = default_client()) {
  impl_client_buffer_to_host(buffer, client = client)
}

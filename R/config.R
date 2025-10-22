#' @title Configure PJRT
#' @description
#' Configure global PJRT options.
#' This needs to set before initializing the PJRT client, usually after loading the package.
#' @param cpu_device_count (`integer(1)`)\cr
#'   The number of CPU devices to use.
#' @return (`list()`)\cr
#'   The current configuration.
#' @export
pjrt_config <- function(cpu_device_count = NULL) {
  if (!is.null(cpu_device_count)) {
    assert_int(cpu_device_count, lower = 1L)
    the[["config"]][["cpu_device_count"]] <- cpu_device_count
  }
  the[["config"]]
}

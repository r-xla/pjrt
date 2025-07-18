get_dims <- function(data) {
  if (is.null(dim(data))) {
    if (length(data) == 1) {
      return(integer())
    }
    return(length(data))
  }
  dim(data)
}

#' @title Default Client
#' @description
#' Creates a client for the [`default_platform`].
#' @return `PJRTClient`
#' @export
default_client <- function() {
  platform <- default_platform()
  plugin_client_create(plugin_load(platform), platform)
}

#' @title Default Platform
#' @description
#' Respects environment variable `PJRT_PLATFORM` and otherwise defaults to "cpu".
#'
#' @return `character(1)`
#' @export
default_platform <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu")
}

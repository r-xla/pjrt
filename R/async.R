# Async Value Classes for PJRT

#' @title Check if an async operation is ready
#' @description
#' Non-blocking check to see if an async operation has completed.
#'
#' For a [`PJRTBuffer`][pjrt_buffer], whether its computation or transfer has
#' finished. For an array returned by [as_array_async()], whether its transfer
#' to the host has finished. Any other R object is always ready.
#'
#' @param x An async value object.
#' @param ... Additional arguments (unused).
#' @return `TRUE` if the operation has completed, `FALSE` otherwise.
#' @export
is_ready <- function(x, ...) {
  UseMethod("is_ready")
}

#' @title Await an async operation
#' @description
#' Block until the async operation is complete and return the object.
#'
#' For a [`PJRTBuffer`][pjrt_buffer], waits for its computation or transfer.
#' For an array returned by [as_array_async()], waits for its transfer to the
#' host and materializes the array. Any other R object is returned as is.
#'
#' @param x An async value object.
#' @param ... Additional arguments (unused).
#' @return The awaited object (invisibly).
#' @export
await <- function(x, ...) {
  UseMethod("await")
}

#' @export
await.PJRTBuffer <- function(x, ...) {
  impl_buffer_await(x)
  invisible(x)
}

#' @export
await.default <- function(x, ...) {
  impl_host_array_materialize(x)
  invisible(x)
}

#' @export
is_ready.PJRTBuffer <- function(x, ...) {
  impl_buffer_is_ready(x)
}

#' @export
is_ready.default <- function(x, ...) {
  impl_host_array_is_ready(x)
}

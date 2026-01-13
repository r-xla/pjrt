# Async Value Classes for PJRT
# Opaque wrappers for async execution results

#' @title Get the value of an async operation
#' @description
#' Materialize and return the result of an async operation.
#' Blocks until the operation is complete if it hasn't finished yet.
#'
#' Returns `PJRTBuffer` or list of `PJRTBuffer`s (same as `pjrt_execute()`).
#'
#' @param x An async value object.
#' @param ... Additional arguments (unused).
#' @return The materialized value.
#' @export
value <- function(x, ...) {
  UseMethod("value")
}

#' @title Check if an async operation is ready
#' @description
#' Non-blocking check to see if an async operation has completed.
#'
#' @param x An async value object.
#' @param ... Additional arguments (unused).
#' @return `TRUE` if the operation has completed, `FALSE` otherwise.
#' @export
is_ready <- function(x, ...) {
  UseMethod("is_ready")
}

# pjrt_async_value Class --------------------------------------------------

#' @title Create a PJRT Async Value (internal)
#' @description
#' Internal constructor for async execution results.
#' Users should not call this directly - use `pjrt_execute_async()` instead.
#' @param buffers List of PJRTBuffer external pointers.
#' @param event PJRTEvent external pointer.
#' @param simplify Whether to simplify single outputs.
#' @return A `pjrt_async_value` object.
#' @keywords internal
pjrt_async_value <- function(buffers, event, simplify = TRUE) {
  env <- new.env(parent = emptyenv())
  env$buffers <- buffers
  env$event <- event
  env$simplify <- simplify
  env$awaited <- FALSE

  structure(env, class = "pjrt_async_value")
}

#' @export
value.pjrt_async_value <- function(x, ...) {
  if (!x$awaited) {
    # If event is NULL, backend doesn't support async events - already complete
    if (!is.null(x$event)) {
      impl_event_await(x$event)
    }
    x$awaited <- TRUE
  }
  if (x$simplify && length(x$buffers) == 1L) {
    x$buffers[[1L]]
  } else {
    x$buffers
  }
}

#' @export
is_ready.pjrt_async_value <- function(x, ...) {
  # If event is NULL, backend doesn't support async events - already complete
  if (is.null(x$event)) {
    return(TRUE)
  }
  impl_event_is_ready(x$event)
}

#' @export
print.pjrt_async_value <- function(x, ...) {
  cat("<pjrt_async_value>\n")
  cat("Value:\n")
  print(value(x))
  invisible(x)
}

#' @keywords internal
is_async_value <- function(x) {
  inherits(x, "pjrt_async_value")
}

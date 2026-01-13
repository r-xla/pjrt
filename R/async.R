# Async Value Classes for PJRT
# Opaque wrappers for async execution results

#' @title Get the value of an async operation
#' @description
#' Materialize and return the result of an async operation.
#' Blocks until the operation is complete if it hasn't finished yet.
#'
#' Returns `PJRTBuffer` for `pjrt_async_value` or an R array for `pjrt_async_buffer`.
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
#'
#' Each `pjrt_async_value` wraps a single buffer and an event. Multiple async
#' values from the same execution share the same event.
#' @param buffer A single PJRTBuffer external pointer.
#' @param event PJRTEvent external pointer (may be shared across values).
#' @return A `pjrt_async_value` object.
#' @keywords internal
pjrt_async_value <- function(buffer, event) {
  env <- new.env(parent = emptyenv())
  env$buffer <- buffer
  env$event <- event
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
  x$buffer
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

#' @export
as_array.pjrt_async_value <- function(x, ...) {
  # Get buffer - this blocks if not ready
  buf <- value(x)
  as_array(buf)
}

#' @keywords internal
is_async_value <- function(x) {
  inherits(x, "pjrt_async_value")
}

# pjrt_async_buffer Class -------------------------------------------------
# Represents the result of an async buffer-to-host transfer

#' @title Create a PJRT Async Buffer (internal)
#' @description
#' Internal constructor for async buffer-to-host transfer results.
#' Users should not call this directly - use `as_array_async()` instead.
#' @param data XPtr to std::vector<uint8_t> holding raw bytes (row-major).
#' @param event PJRTEvent external pointer (or NULL).
#' @param dtype Element type string (e.g., "f32", "i32").
#' @param dims Integer vector of dimensions.
#' @return A `pjrt_async_buffer` object.
#' @keywords internal
pjrt_async_buffer <- function(data, event, dtype, dims) {
  env <- new.env(parent = emptyenv())
  env$data <- data
  env$event <- event
  env$dtype <- dtype
  env$dims <- dims
  env$materialized <- NULL

  structure(env, class = "pjrt_async_buffer")
}

#' @export
value.pjrt_async_buffer <- function(x, ...) {
  if (is.null(x$materialized)) {
    # Wait for transfer to complete (if event exists)
    if (!is.null(x$event)) {
      impl_event_await(x$event)
    }
    # Convert raw bytes to R array
    x$materialized <- impl_raw_to_array(x$data, x$dtype, x$dims)
  }
  x$materialized
}

#' @export
is_ready.pjrt_async_buffer <- function(x, ...) {
  if (is.null(x$event)) {
    return(TRUE)
  }
  impl_event_is_ready(x$event)
}

#' @export
as_array.pjrt_async_buffer <- function(x, ...) {
  value(x)
}

#' @export
print.pjrt_async_buffer <- function(x, ...) {
  cat("<pjrt_async_buffer>\n")
  cat("Value:\n")
  print(value(x))
  invisible(x)
}

#' @keywords internal
is_async_buffer <- function(x) {
  inherits(x, "pjrt_async_buffer")
}

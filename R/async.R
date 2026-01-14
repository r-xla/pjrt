# Async Value Classes for PJRT
# Promise-style wrappers for async operation results

#' @title Get the value of an async operation
#' @description
#' Materialize and return the result of an async operation.
#' Blocks until the operation is complete if it hasn't finished yet.
#'
#' Returns `PJRTBuffer` for `pjrt_buffer_promise` or an R array for `pjrt_array_promise`.
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

# pjrt_buffer_promise Class ------------------------------------------------
# Represents a promise of a PJRTBuffer (from execution or host-to-device transfer)

#' @title Create a PJRT Buffer Promise (internal)
#' @description
#' Internal constructor for async buffer results.
#' Users should not call this directly - use `pjrt_execute_async()` or
#' `pjrt_buffer_async()` instead.
#'
#' The buffer is valid immediately and can be used in subsequent operations
#' (PJRT handles dependencies internally). Call `value()` to block until
#' the operation is complete.
#'
#' @param buffer A PJRTBuffer external pointer (valid immediately).
#' @param event PJRTEvent external pointer (or NULL if already complete).
#' @param data_holder Optional XPtr keeping host data alive until transfer completes.
#' @return A `pjrt_buffer_promise` object.
#' @keywords internal
pjrt_buffer_promise <- function(buffer, event, data_holder = NULL) {
  env <- new.env(parent = emptyenv())
  env$buffer <- buffer
  env$event <- event
  env$data_holder <- data_holder
  env$awaited <- FALSE

  structure(env, class = "pjrt_buffer_promise")
}

#' @export
value.pjrt_buffer_promise <- function(x, ...) {
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
is_ready.pjrt_buffer_promise <- function(x, ...) {
  # If event is NULL, backend doesn't support async events - already complete
  if (is.null(x$event)) {
    return(TRUE)
  }
  impl_event_is_ready(x$event)
}

#' @export
print.pjrt_buffer_promise <- function(x, ...) {
  cat("<pjrt_buffer_promise>\n")
  if (is_ready(x)) {
    cat("Status: Ready\n")
    cat("Buffer:\n")
    print(value(x))
  } else {
    cat("Status: Pending\n")
  }
  invisible(x)
}

#' @export
as_array.pjrt_buffer_promise <- function(x, ...) {
  # Get buffer - this blocks if not ready
  buf <- value(x)
  as_array(buf)
}

#' @keywords internal
is_buffer_promise <- function(x) {
  inherits(x, "pjrt_buffer_promise")
}

# pjrt_array_promise Class -------------------------------------------------
# Represents a promise of an R array (from device-to-host transfer)

#' @title Create a PJRT Array Promise (internal)
#' @description
#' Internal constructor for async device-to-host transfer results.
#' Users should not call this directly - use `as_array_async()` instead.
#' @param data XPtr to std::vector<uint8_t> holding raw bytes (row-major).
#' @param event PJRTEvent external pointer (or NULL).
#' @param dtype Element type string (e.g., "f32", "i32").
#' @param dims Integer vector of dimensions.
#' @return A `pjrt_array_promise` object.
#' @keywords internal
pjrt_array_promise <- function(data, event, dtype, dims) {
  env <- new.env(parent = emptyenv())
  env$data <- data
  env$event <- event
  env$dtype <- dtype
  env$dims <- dims
  env$materialized <- NULL

  structure(env, class = "pjrt_array_promise")
}

#' @export
value.pjrt_array_promise <- function(x, ...) {
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
is_ready.pjrt_array_promise <- function(x, ...) {
  if (is.null(x$event)) {
    return(TRUE)
  }
  impl_event_is_ready(x$event)
}

#' @export
as_array.pjrt_array_promise <- function(x, ...) {
  value(x)
}

#' @export
print.pjrt_array_promise <- function(x, ...) {
  cat("<pjrt_array_promise>\n")
  if (is_ready(x)) {
    cat("Status: Ready\n")
    cat("Value:\n")
    print(value(x))
  } else {
    cat("Status: Pending\n")
  }
  invisible(x)
}

#' @keywords internal
is_array_promise <- function(x) {
  inherits(x, "pjrt_array_promise")
}

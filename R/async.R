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
#' @param events List of ancestor events to check for errors (for chained operations).
#' @return A `pjrt_buffer_promise` object.
#' @keywords internal
pjrt_buffer_promise <- function(buffer, event, data_holder = NULL, events = list()) {
  env <- new.env(parent = emptyenv())
  env$buffer <- buffer
  env$event <- event
  env$data_holder <- data_holder
  env$awaited <- FALSE
  # Accumulate all events in the chain (ancestors + this event)
  env$events <- if (!is.null(event)) c(events, list(event)) else events

  structure(env, class = "pjrt_buffer_promise")
}

#' @export
value.pjrt_buffer_promise <- function(x, ...) {
  if (!x$awaited) {
    # Await ALL events in the chain to ensure errors are propagated
    for (evt in x$events) {
      impl_event_await(evt)
    }
    x$awaited <- TRUE
  }
  x$buffer
}

#' @export
is_ready.pjrt_buffer_promise <- function(x, ...) {
  # Check if ALL events in the chain are ready
  for (evt in x$events) {
    if (!impl_event_is_ready(evt)) {
      return(FALSE)
    }
  }
  TRUE
}

#' @export
print.pjrt_buffer_promise <- function(x, ...) {

  cat("<pjrt_buffer_promise>\n")
  if (x$awaited) {
    # Already awaited - safe to show buffer without side effects
    cat("Status: Awaited\n")
    cat("Buffer:\n")
    print(x$buffer)
  } else {
    # Not yet awaited - show promise info without materializing
    cat("Status: Not awaited\n")
    cat("Events:", length(x$events), "\n")
    cat("(Call value() to await and retrieve the buffer)\n")
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

#' @keywords internal
#' @description Extract all events from a buffer promise for chaining
get_events <- function(x) {
  if (inherits(x, "pjrt_buffer_promise")) {
    x$events
  } else {
    list()
  }
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
#' @param events List of ancestor events to check for errors (for chained operations).
#' @return A `pjrt_array_promise` object.
#' @keywords internal
pjrt_array_promise <- function(data, event, dtype, dims, events = list()) {
  env <- new.env(parent = emptyenv())
  env$data <- data
  env$event <- event
  env$dtype <- dtype
  env$dims <- dims
  env$materialized <- NULL
  # Accumulate all events in the chain (ancestors + this event)
  env$events <- if (!is.null(event)) c(events, list(event)) else events

  structure(env, class = "pjrt_array_promise")
}

#' @export
value.pjrt_array_promise <- function(x, ...) {
  if (is.null(x$materialized)) {
    # Await ALL events in the chain to ensure errors are propagated
    for (evt in x$events) {
      impl_event_await(evt)
    }
    # Convert raw bytes to R array
    x$materialized <- impl_raw_to_array(x$data, x$dtype, x$dims)
  }
  x$materialized
}

#' @export
is_ready.pjrt_array_promise <- function(x, ...) {
  # Check if ALL events in the chain are ready
  for (evt in x$events) {
    if (!impl_event_is_ready(evt)) {
      return(FALSE)
    }
  }
  TRUE
}

#' @export
as_array.pjrt_array_promise <- function(x, ...) {
  value(x)
}

#' @export
print.pjrt_array_promise <- function(x, ...) {
  cat("<pjrt_array_promise>\n")
  if (!is.null(x$materialized)) {
    # Already materialized - safe to show value without side effects
    cat("Status: Materialized\n")
    cat("Value:\n")
    print(x$materialized)
  } else {
    # Not yet materialized - show promise info without triggering transfer
    cat("Status: Not materialized\n")
    cat("Events:", length(x$events), "\n")
    cat("dtype:", x$dtype, "\n")
    cat("dims:", paste(x$dims, collapse = " x "), "\n")
    cat("(Call value() to materialize the array)\n")
  }
  invisible(x)
}

#' @keywords internal
is_array_promise <- function(x) {
  inherits(x, "pjrt_array_promise")
}

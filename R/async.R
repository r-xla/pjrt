# Async Value Classes for PJRT
# Promise-style wrappers for async operation results

#' @title Get the value of an async operation
#' @description
#' Materialize and return the result of an async operation.
#' Blocks until the operation is complete if it hasn't finished yet.
#'
#' Returns `PJRTBuffer` for `PJRTBufferPromise` or an R array for `PJRTArrayPromise`.
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

# PJRTBufferPromise Class ------------------------------------------------
# Represents a promise of a PJRTBuffer (from execution or host-to-device transfer)

#' @title Create a PJRT Buffer Promise (internal)
#' @description
#' Internal constructor for async buffer results.
#' Users should not call this directly - use `pjrt_execute()` or
#' `pjrt_buffer()` instead.
#'
#' The buffer is valid immediately and can be used in subsequent operations
#' (PJRT handles dependencies internally). Call `value()` to block until
#' the operation is complete.
#'
#' @param buffer A PJRTBuffer external pointer (valid immediately).
#' @return A `PJRTBufferPromise` object.
#' @keywords internal
pjrt_buffer_promise <- function(buffer) {
  env <- new.env(parent = emptyenv())
  env$buffer <- buffer
  env$awaited <- FALSE
  structure(env, class = "PJRTBufferPromise")
}

#' @export
value.PJRTBufferPromise <- function(x, ...) {
  if (!x$awaited) {
    impl_buffer_await(x$buffer)
    x$awaited <- TRUE
  }
  x$buffer
}

#' @export
is_ready.PJRTBufferPromise <- function(x, ...) {
  impl_buffer_is_ready(x$buffer)
}

#' @export
print.PJRTBufferPromise <- function(x, ...) {
  cat("<PJRTBufferPromise>\n")
  if (x$awaited) {
    cat("Status: Awaited\n")
    cat("Buffer:\n")
    print(x$buffer)
  } else {
    cat("Status: Not awaited\n")
    cat("(Call value() to await and retrieve the buffer)\n")
  }
  invisible(x)
}

#' @export
as_array.PJRTBufferPromise <- function(x, ...) {
  value(as_array_async(x))
}

#' @keywords internal
is_buffer_promise <- function(x) {
  inherits(x, "PJRTBufferPromise")
}

# PJRTArrayPromise Class -------------------------------------------------
# Represents a promise of an R array (from device-to-host transfer)

#' @title Create a PJRT Array Promise (internal)
#' @description
#' Internal constructor for async device-to-host transfer results.
#' Users should not call this directly - use `as_array_async()` instead.
#' @param data XPtr to PJRTHostData holding raw bytes and completion event.
#' @param dtype Element type string (e.g., "f32", "i32").
#' @param shape Integer vector of dimensions.
#' @return A `PJRTArrayPromise` object.
#' @keywords internal
pjrt_array_promise <- function(data, dtype, shape) {
  env <- new.env(parent = emptyenv())
  env$data <- data
  env$dtype <- dtype
  env$shape <- shape
  env$materialized <- NULL
  structure(env, class = "PJRTArrayPromise")
}

#' @export
value.PJRTArrayPromise <- function(x, ...) {
  if (is.null(x$materialized)) {
    impl_host_data_await(x$data)
    x$materialized <- impl_raw_to_array(x$data, x$dtype, x$shape)
  }
  x$materialized
}

#' @export
is_ready.PJRTArrayPromise <- function(x, ...) {
  impl_host_data_is_ready(x$data)
}

#' @export
as_array.PJRTArrayPromise <- function(x, ...) {
  value(x)
}

#' @export
print.PJRTArrayPromise <- function(x, ...) {
  cat("<PJRTArrayPromise>\n")
  if (!is.null(x$materialized)) {
    cat("Status: Materialized\n")
    cat("Value:\n")
    print(x$materialized)
  } else {
    cat("Status: Not materialized\n")
    cat("dtype:", x$dtype, "\n")
    cat("shape:", paste(x$shape, collapse = " x "), "\n")
    cat("(Call value() to materialize the array)\n")
  }
  invisible(x)
}

#' @keywords internal
is_array_promise <- function(x) {
  inherits(x, "PJRTArrayPromise")
}

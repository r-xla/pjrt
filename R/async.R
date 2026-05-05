# Async Value Classes for PJRT

#' @title Get the value of an async operation
#' @description
#' Materialize and return the result of an async operation.
#' Blocks until the operation is complete if it hasn't finished yet.
#'
#' For `PJRTArrayPromise`, returns the materialized R array.
#' For `PJRTBuffer`, use `await()` to block until ready.
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

#' @title Await an async operation
#' @description
#' Block until the async operation is complete and return the object.
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
is_ready.PJRTBuffer <- function(x, ...) {
  impl_buffer_is_ready(x)
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
#' @param as_integer64 If `TRUE` and `dtype == "i64"`, materialize as
#'   `bit64::integer64` instead of `integer`.
#' @return A `PJRTArrayPromise` object.
#' @keywords internal
pjrt_array_promise <- function(data, dtype, shape, as_integer64 = FALSE) {
  env <- new.env(parent = emptyenv())
  env$data <- data
  env$dtype <- dtype
  env$shape <- shape
  env$as_integer64 <- isTRUE(as_integer64)
  env$materialized <- NULL
  structure(env, class = "PJRTArrayPromise")
}

#' @export
value.PJRTArrayPromise <- function(x, ...) {
  if (is.null(x$materialized)) {
    impl_host_data_await(x$data)
    if (x$as_integer64 && identical(x$dtype, "i64")) {
      out <- impl_raw_to_integer64_array(x$data, x$shape)
      class(out) <- "integer64"
      x$materialized <- out
    } else {
      x$materialized <- impl_raw_to_array(x$data, x$dtype, x$shape)
    }
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

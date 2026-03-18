# Extract the raw PJRTBuffer XPtr from a buffer promise or buffer.
# For buffer promises, the buffer is valid immediately - PJRT handles
# dependencies internally. If there's a data_holder, register callback
# to keep it alive until the transfer completes.
resolve_buffer_input <- function(x) {
  if (is_buffer_promise(x)) {
    if (!is.null(x$event) && !is.null(x$data_holder)) {
      impl_event_release_on_ready(x$event, x$data_holder)
    }
    x$buffer
  } else if (is_buffer(x)) {
    x
  } else {
    cli_abort("Expected PJRTBuffer or PJRTBufferPromise")
  }
}

#' @title Execute a PJRT program
#' @description
#' Execute a PJRT program with the given inputs and execution options.
#' Returns immediately with buffer promise(s) that can be awaited later.
#'
#' **Important:**
#' Arguments are passed by position and names are ignored.
#'
#' Inputs can be `PJRTBuffer` objects or buffer promises (`PJRTBufferPromise`).
#' Buffer promises are resolved automatically before execution.
#'
#' Use `value()` to get the result (blocks if not ready).
#' Use `is_ready()` to check if execution has completed (non-blocking).
#' Use `as_array_async()` to chain async buffer-to-host transfer.
#'
#' @param executable (`PJRTLoadedExecutable`)\cr
#' A PJRT program.
#' @param ... (`PJRTBuffer` | `PJRTBufferPromise`)\cr
#'   Inputs to the program.
#'   Named are ignored and arguments are passed in order.
#' @param execution_options (`PJRTExecuteOptions`)\cr
#'   Optional execution options for configuring buffer donation and other settings.
#' @param simplify (`logical(1)`)\cr
#'   If `TRUE` (default), a single output is returned as a `PJRTBufferPromise`.
#'   If `FALSE`, a single output is returned as a `list` of length 1 containing a `PJRTBufferPromise`.
#' @return `PJRTBufferPromise` | `list` of `PJRTBufferPromise`s
#' @seealso [value()], [is_ready()], [as_array_async()]
#' @examplesIf plugin_is_downloaded()
#' # Create and compile a simple identity program
#' src <- r"(
#' func.func @main(
#'   %x: tensor<2x2xf32>,
#'   %y: tensor<2x2xf32>
#' ) -> tensor<2x2xf32> {
#'   %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
#'   "func.return"(%0): (tensor<2x2xf32>) -> ()
#' }
#' )"
#' prog <- pjrt_program(src = src)
#' exec <- pjrt_compile(prog)
#'
#' # Execute with input
#' x <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
#' y <- pjrt_buffer(c(5, 6, 7, 8), shape = c(2, 2), dtype = "f32")
#' pjrt_execute(exec, x, y)
#' @export
pjrt_execute <- function(executable, ..., execution_options = NULL, simplify = TRUE) {
  if (!is.null(...names())) {
    cli_abort("Expected unnamed arguments")
  }

  check_loaded_executable(executable)
  input_raw <- list(...)

  # Resolve any async inputs (extract buffers)
  input <- lapply(input_raw, resolve_buffer_input)

  if (is.null(execution_options)) {
    execution_options <- pjrt_execution_options()
  } else {
    check_execution_options(execution_options)
  }

  assert_flag(simplify)

  result <- impl_loaded_executable_execute_async(executable, input, execution_options)

  # Create a list of buffer promises, one per buffer, all sharing the same event
  promises <- lapply(result$buffers, function(buf) {
    pjrt_buffer_promise(buf, result$event)
  })

  if (simplify && length(promises) == 1L) {
    return(promises[[1L]])
  }

  promises
}

#' @export
print.PJRTLoadedExecutable <- function(x, ...) {
  cat("<PJRTLoadedExecutable>\n")
  invisible(x)
}

check_loaded_executable <- function(x) {
  stopifnot(inherits(x, "PJRTLoadedExecutable"))
  invisible(NULL)
}

is_loaded_executable <- function(x) {
  inherits(x, "PJRTLoadedExecutable")
}

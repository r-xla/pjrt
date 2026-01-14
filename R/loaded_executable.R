# Helper to resolve async inputs to buffers
# For buffer promises, the buffer is valid immediately - PJRT handles
# dependencies internally. If there's a data_holder, register callback
# to keep it alive until the transfer completes.
resolve_buffer_input <- function(x) {
  if (inherits(x, "pjrt_buffer_promise")) {
    # Buffer is valid immediately - PJRT handles dependencies internally.
    # If there's a data_holder (from host-to-device transfer), register
    # callback to keep it alive until transfer completes.
    if (!is.null(x$event) && !is.null(x$data_holder)) {
      impl_event_release_on_ready(x$event, x$data_holder)
    }
    x$buffer
  } else if (is_buffer(x)) {
    x
  } else {
    cli_abort("Expected PJRTBuffer or pjrt_buffer_promise")
  }
}

#' @title Execute a PJRT program
#' @description
#' Execute a PJRT program with the given inputs and execution options.
#'
#' **Important:**
#' Arguments are passed by position and names are ignored.
#'
#' Inputs can be `PJRTBuffer` objects or buffer promises (`pjrt_buffer_promise`).
#' Buffer promises are resolved automatically before execution.
#'
#' @param executable (`PJRTLoadedExecutable`)\cr
#' A PJRT program.
#' @param ... (`PJRTBuffer` | `pjrt_buffer_promise`)\cr
#'   Inputs to the program.
#'   Named are ignored and arguments are passed in order.
#' @param execution_options (`PJRTExecuteOptions`)\cr
#'   Optional execution options for configuring buffer donation and other settings.
#' @param simplify (`logical(1)`)\cr
#'   If `TRUE` (default), a single output is returned as a `PJRTBuffer`.
#'   If `FALSE`, a single output is returned as a `list` of length 1 containing a `PJRTBuffer`.
#' @return `PJRTBuffer` | `list` of `PJRTBuffer`s
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
  input <- list(...)
  # Resolve any async inputs (auto-wait)
  input <- lapply(input, resolve_buffer_input)

  if (is.null(execution_options)) {
    execution_options <- pjrt_execution_options()
  } else {
    check_execution_options(execution_options)
  }

  assert_flag(simplify)

  result <- impl_loaded_executable_execute(executable, input, execution_options)

  if (simplify && length(result) == 1L) {
    return(result[[1]])
  }

  result
}

#' @title Execute a PJRT program asynchronously
#' @description
#' Execute a PJRT program asynchronously with the given inputs.
#' Returns immediately with buffer promise(s) that can be awaited later.
#'
#' Use `value()` to get the result (blocks if not ready).
#' Use `is_ready()` to check if execution has completed (non-blocking).
#' Use `as_array_async()` to chain async buffer-to-host transfer.
#'
#' Inputs can be `PJRTBuffer` objects or buffer promises (`pjrt_buffer_promise`).
#' Buffer promises are resolved automatically before execution.
#'
#' @inheritParams pjrt_execute
#' @return A `pjrt_buffer_promise` object (or list of them if multiple outputs).
#'   Call `value()` to get the `PJRTBuffer`.
#' @seealso [pjrt_execute()], [value()], [is_ready()], [as_array_async()], [pjrt_buffer_async()]
#' @examplesIf plugin_is_downloaded()
#' # Create and compile a simple program
#' src <- r"(
#' func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
#'   "func.return"(%x): (tensor<2x2xf32>) -> ()
#' }
#' )"
#' prog <- pjrt_program(src = src)
#' exec <- pjrt_compile(prog)
#'
#' # Execute asynchronously
#' x <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
#' result <- pjrt_execute_async(exec, x)
#'
#' # Check if ready (non-blocking)
#' is_ready(result)
#'
#' # Get the result (blocks if not ready)
#' value(result)
#'
#' # Chain with async buffer-to-host transfer
#' arr <- as_array_async(result)
#' value(arr)
#' @export
pjrt_execute_async <- function(executable, ..., execution_options = NULL, simplify = TRUE) {
  if (!is.null(...names())) {
    cli_abort("Expected unnamed arguments")
  }

  check_loaded_executable(executable)
  input_raw <- list(...)

  # Collect all events from input buffer promises for error propagation
  parent_events <- unlist(lapply(input_raw, get_events), recursive = FALSE)

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
  # Pass parent events for error propagation through the chain
  promises <- lapply(result$buffers, function(buf) {
    pjrt_buffer_promise(buf, result$event, events = parent_events)
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

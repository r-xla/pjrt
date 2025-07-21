loaded_executable_execute <- function(
  executable,
  input,
  execution_options = NULL
) {
  check_loaded_executable(executable)
  if (is_buffer(input)) {
    input <- list(input)
  }
  lapply(input, check_buffer)

  if (is.null(execution_options)) {
    execution_options <- pjrt_execution_options()
  } else {
    check_execution_options(execution_options)
  }

  impl_loaded_executable_execute(executable, input, execution_options)
}

#' @title Execute a PJRT program
#' @description
#' Execute a PJRT program with the given inputs and execution options.
#'
#' **Important:**
#' Arguments are passed by position and names are ignored.
#'
#' @param executable (`PJRTLoadedExecutable`)\cr
#' A PJRT program.
#' @param ... (`PJRTBuffer)`\cr
#'   Inputs to the program.
#'   Named are ignored and arguments are passed in order.
#' @param execution_options (`PJRTExecuteOptions`)\cr
#'   Optional execution options for configuring buffer donation and other settings.
#' @return `PJRTBuffer` | list of `PJRTBuffers`
#' @export
pjrt_execute <- function(executable, ..., execution_options = NULL) {
  if (!is.null(...names())) {
    stop("Expected unnamed arguments")
  }
  loaded_executable_execute(executable, list(...), execution_options)
}

check_loaded_executable <- function(x) {
  stopifnot(inherits(x, "PJRTLoadedExecutable"))
  invisible(NULL)
}

is_loaded_executable <- function(x) {
  inherits(x, "PJRTLoadedExecutable")
}

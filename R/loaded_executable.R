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
#' @param simplify (`logical(1)`)\cr
#'   If `TRUE` (default), a single output is returned as a `PJRTBuffer`.
#'   If `FALSE`, a single output is returned as a `list` of length 1 containing a `PJRTBuffer`.
#' @return `PJRTBuffer` | `list` of `PJRTBuffer`s
#' @export
pjrt_execute <- function(executable, ..., execution_options = NULL, simplify = TRUE) {
  if (!is.null(...names())) {
    stop("Expected unnamed arguments")
  }
  check_loaded_executable(executable)
  input <- list(...)
  lapply(input, check_buffer)

  if (is.null(execution_options)) {
    execution_options <- pjrt_execution_options()
  } else {
    check_execution_options(execution_options)
  }

  assert_flag(simplify)

  result <- impl_loaded_executable_execute(executable, input, execution_options)

  if (is.list(result)) {
    return(result)
  }

  if (simplify) {
    return(result)
  }

  list(result)
}

check_loaded_executable <- function(x) {
  stopifnot(inherits(x, "PJRTLoadedExecutable"))
  invisible(NULL)
}

is_loaded_executable <- function(x) {
  inherits(x, "PJRTLoadedExecutable")
}

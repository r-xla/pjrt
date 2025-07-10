loaded_executable_execute <- function(executable, input) {
  check_loaded_executable(executable)
  if (is_buffer(input)) {
    input <- list(input)
  }
  lapply(input, check_buffer)
  impl_loaded_executable_execute(executable, input)
}

#' @title Execute a PJRT program
#' @description Execute a PJRT program with the given inputs.
#' @param executable (`PJRTLoadedExecutable`)\cr
#' A PJRT program.
#' @param ... Inputs to the program.
#' @return `PJRTBuffer` | list of `PJRTBuffers`
#' @export
pjrt_execute <- function(executable, ...) {
  loaded_executable_execute(executable, list(...))
}

check_loaded_executable <- function(x) {
  stopifnot(inherits(x, "PJRTLoadedExecutable"))
  invisible(NULL)
}

is_loaded_executable <- function(x) {
  inherits(x, "PJRTLoadedExecutable")
}

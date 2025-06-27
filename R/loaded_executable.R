loaded_executable_execute <- function(executable, input) {
  check_loaded_executable(executable)
  if (is_buffer(input)) {
    input <- list(input)
  }
  lapply(input, check_buffer)
  impl_loaded_executable_execute(executable, input)
}

check_loaded_executable <- function(x) {
  stopifnot(inherits(x, "PJRTLoadedExecutable"))
  invisible(NULL)
}

is_loaded_executable <- function(x) {
  inherits(x, "PJRTLoadedExecutable")
}

loaded_executable_execute <- function(executable, input) {
  check_loaded_executable(executable)
  impl_loaded_executable_execute(executable, input)
}

check_loaded_executable <- function(x) {
  stopifnot(inherits(x, "PJRTLoadedExecutable"))
  invisible(NULL)
}

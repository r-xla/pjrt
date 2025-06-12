program_load <- function(path, format = c("mlir", "hlo")) {
  path <- normalizePath(path, mustWork = TRUE)
  format <- rlang::arg_match(format)
  impl_program_load(path, format)
}

#' @export
format.PJRTProgram <- function(x, ..., n = 10) {
  check_program(x)
  impl_program_repr(x, n)
}

#' @export
print.PJRTProgram <- function(x, ..., n = 5) {
  check_program(x)
  cat(format(x, ..., n = n), "\n")
  invisible(x)
}

check_program <- function(program) {
  stopifnot(inherits(program, "PJRTProgram"))
  invisible(NULL)
}

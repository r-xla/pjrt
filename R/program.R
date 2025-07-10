program_load <- function(path, format = c("mlir", "hlo")) {
  path <- normalizePath(path, mustWork = TRUE)
  format <- rlang::arg_match(format)
  impl_program_load(path, format)
}

#' @title Create a `PJRTProgram`
#' @description
#' Create a program from a string.
#' @param code (`character(1)`)\cr
#'   Source code.
#' @param format (`character(1)`)\cr
#'   The format of the program.
#'   One of "mlir" or "hlo".
#' @return `PJRTProgram`
#' @export
pjrt_program <- function(code, format = c("mlir", "hlo")) {
  if (!is.character(code) || length(code) != 1) {
    stop("code must be a character(1)")
  }
  format <- rlang::arg_match(format)
  on.exit(unlink(file))
  file <- tempfile()
  writeLines(code, file)
  program_load(file, format)
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

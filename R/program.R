#' @title Create a `PJRTProgram`
#' @description
#' Create a program from a string or file path.
#' @param src (`character(1)`)
#'   Source code.
#' @param path (`character(1)`)
#'   Path to the program file.
#' @param format (`character(1)`)
#'   One of "mlir" or "hlo".
#' @return `PJRTProgram`
#' @export
pjrt_program <- function(src = NULL, path = NULL, format = c("mlir", "hlo")) {
  if (!xor(is.null(src), is.null(path))) {
    cli_abort("Either src or path must be provided")
  }
  checkmate::assert(
    checkmate::check_string(src, null.ok = TRUE),
    checkmate::check_string(path, null.ok = TRUE),
    combine = "or"
  )
  temp_file <- NULL
  if (!is.null(src)) {
    temp_file <- tempfile()
    writeLines(src, temp_file)
    path <- temp_file
    on.exit(unlink(temp_file))
  }
  format <- rlang::arg_match(format)
  path <- normalizePath(path, mustWork = TRUE)
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

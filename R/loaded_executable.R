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
  lapply(input, check_buffer)

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

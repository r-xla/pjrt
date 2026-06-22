#' @title Format Buffer Data
#'
#' @description
#' Formats buffer data into a character vector of string representations of individual elements suitable for stableHLO.
#'
#' @template param_buffer
#'
#' @return `character()` A character vector containing the formatted elements.
#' @examplesIf plugins_downloaded()
#' buf <- pjrt_buffer(c(1.5, 2.5, 3.5))
#' format_buffer(buf)
#' @export
#' @title Format Array Lines
#'
#' @description
#' Formats an R array (or vector/scalar) into display lines using the same
#' printer as [print.PJRTBuffer()].
#'
#' @param data An R numeric, integer, or logical array/vector/scalar.
#' @param max_rows Maximum total rows to print (`-1` for unlimited).
#' @param max_width Maximum line width in characters.
#' @param max_rows_slice Maximum rows per 2D slice.
#'
#' @return `character()` Vector of formatted lines.
#' @examples
#' format_array(matrix(1:6, nrow = 2))
#' @export
format_array <- function(
  data,
  max_rows = getOption("pjrt.print_max_rows", 30L),
  max_width = getOption("pjrt.print_max_width", 85L),
  max_rows_slice = getOption("pjrt.print_max_rows_slice", max_rows)
) {
  impl_format_array(data, max_rows, max_width, max_rows_slice)
}

format_buffer <- function(buffer) {
  if (!is_buffer(buffer)) {
    cli_abort("`buffer` must be a `PJRTBuffer`")
  }
  out <- format_raw_buffer_cpp(
    as_raw(buffer, row_major = FALSE),
    as.character(elt_type(buffer)),
    shape(buffer)
  )
  if (!identical(shape(buffer), integer())) {
    out <- array(out, shape(buffer))
  }
  out
}

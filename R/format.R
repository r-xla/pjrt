#' @title Format Buffer Data
#'
#' @description
#' Formats buffer data into a character vector of string representations of individual elements suitable for stableHLO.
#'
#' @template param_buffer
#'
#' @return `character()` A character vector containing the formatted elements.
#' @export
format_buffer <- function(buffer) {
  if (!inherits(buffer, "PJRTBuffer")) {
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

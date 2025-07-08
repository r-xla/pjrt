check_buffer <- function(x) {
  stopifnot(is_buffer(x))
  invisible(NULL)
}

is_buffer <- function(x) {
  inherits(x, "PJRTBuffer")
}

#' Create a PJRT buffer
#'
#' @param data A vector of data to convert to a PJRT buffer.
#' @param ... Additional arguments.
#' @param client A PJRT client object.
#'
#' @return A PJRT buffer object.
pjrt_buffer <- function(data, ..., client = default_client()) {
  switch(
    typeof(data),
    logical = client_buffer_from_logical(data, ..., client = client),
    integer = client_buffer_from_integer(data, ..., client = client),
    double = client_buffer_from_double(data, ..., client = client),
    stop("Unsupported data type: ", typeof(data))
  )
}

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
pjrt_buffer <- function(data, client = default_client(), ...) {
  UseMethod("pjrt_buffer")
}


#' @export
pjrt_buffer.logical <- function(data, client = default_client()) {
  dim = get_dims(data)
  client_buffer_from_logical(data, dim = dim, client = client)
}

#' @export
pjrt_buffer.integer <- function(data, client = default_client(), precision = 32) {
  dim = get_dims(data)
  client_buffer_from_integer(data, dim = dim, client = client, precision = precision)
}


#' @export
pjrt_buffer.double <- function(data, client = default_client(), precision = 32) {
  dim = get_dims(data)
  client_buffer_from_double(data, dim = dim, client = client, precision = precision)
}

#' @export
pjrt_scalar <- function(data, client = default_client(), ...) {
  UseMethod("pjrt_scalar")
}

#' @export
pjrt_scalar.logical <- function(data, client = default_client()) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  client_buffer_from_logical(data, dim = integer(), client = client)
}

#' @export
pjrt_scalar.integer <- function(data, client = default_client(), precision = 32) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  client_buffer_from_integer(data, dim = integer(), client = client, precision = precision)
}

#' @export
pjrt_scalar.double <- function(data, client = default_client(), precision = 32) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  client_buffer_from_double(data, dim = integer(), client = client, precision = precision)
}

#' Get the data type of a PJRT buffer
#'
#' @param buffer A PJRT buffer object.
#'
#' @return A PJRT element type object.
#' @export
pjrt_elt_type <- function(buffer) {
  check_buffer(buffer)
  impl_buffer_element_type(buffer)
}

#' Check if an object is a PJRT element type
#'
#' @param x An object to check.
#'
#' @return TRUE if the object is a PJRT element type, FALSE otherwise.
is_element_type <- function(x) {
  inherits(x, "PJRTElementType")
}

#' Get a string representation of a PJRT element type
#'
#' @param element_type A PJRT element type object.
#'
#' @return A string representation of the element type.
#' @export
as.character.PJRTElementType <- function(x, ...) {
  impl_element_type_as_string(x)
}

#' Print a PJRT element type
#'
#' @param x A PJRT element type object.
#' @param ... Additional arguments passed to print.
#' @export
print.PJRTElementType <- function(x, ...) {
  cat("PJRT Element Type: ", as.character(x), "\n")
}

#' Get the dimensions of a PJRT buffer
#'
#' @param x A PJRT buffer object.
#'
#' @return An integer vector of dimensions.
#' @export
dim.PJRTBuffer <- function(x) {
  check_buffer(x)
  impl_buffer_dimensions(x)
}

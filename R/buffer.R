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
#' @param ... Additional arguments.
#' @return A PJRT buffer object.
#' @export
pjrt_buffer <- function(data, client, ...) {
  UseMethod("pjrt_buffer")
}


#' @export
pjrt_buffer.logical <- function(data, client = default_client(), ...) {
  dims = get_dims(data)
  client_buffer_from_logical(data, dims = dims, client = client)
}

#' @export
pjrt_buffer.integer <- function(data, client = default_client(), precision = 32, signed = TRUE, ...) {
  dims = get_dims(data)
  client_buffer_from_integer(data, dims = dims, client = client, precision = precision, signed = signed)
}

#' @export
pjrt_buffer.double <- function(data, client = default_client(), precision = 32, ...) {
  dims = get_dims(data)
  client_buffer_from_double(data, dims = dims, client = client, precision = precision)
}

#' Create a PJRT scalar buffer
#'
#' @param data A scalar value to convert to a PJRT buffer.
#' @param client A PJRT client object.
#' @param ... Additional arguments.
#' @return A PJRT buffer object representing a scalar.
#' @export
pjrt_scalar <- function(data, client, ...) {
  UseMethod("pjrt_scalar")
}

#' @export
pjrt_scalar.logical <- function(data, client = default_client(), ...) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  client_buffer_from_logical(data, dims = integer(), client = client)
}

#' @export
pjrt_scalar.integer <- function(data, client = default_client(), precision = NULL, ...) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (is.null(precision)) {
    precision = 32
  }
  client_buffer_from_integer(data, dims = integer(), client = client, precision = precision, ...)
}

#' @export
pjrt_scalar.double <- function(data, client = default_client(), precision = NULL, ...) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (is.null(precision)) {
    precision = 32
  }
  client_buffer_from_double(data, dims = integer(), client = client, precision = precision)
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

#' @export
print.PJRTElementType <- function(x, ...) {
  cat(sprintf("<ElementType: %s>\n"))
}

#' Get the platform name of a PJRT client
#'
#' @param client A PJRT client object.
#'
#' @return A string representing the platform name.
#' @export
client_platform_name <- function(client = default_client()) {
  check_client(client)
  impl_client_platform_name(client)
}

check_buffer <- function(x) {
  stopifnot(is_buffer(x))
  invisible(NULL)
}

is_buffer <- function(x) {
  inherits(x, "PJRTBuffer")
}

#' @title Create a PJRT Buffer
#' @rdname pjrt_buffer
#' @description
#' Create a PJRT Buffer from an R object.
#' Any numeric PJRT buffer is an array and 0-dimensional arrays are used as scalars.
#' [`pjrt_buffer`] will create a array with dimensions `(1)` for a vector of length 1, while
#' Therefore, [`pjrt_scalar`] will create a 0-dimensional array for an R vector of length 1.
#'
#' @details
#' R does not have 0-dimensional arrays, hence we need the extra `pjrt_scalar` function.
#'
#' @param data (any)\cr
#'  Data to convert to a `PJRTBuffer`.
#' @param type (`character(1)`)\cr
#'   The type of the buffer.
#'   Currently supported types are:
#'   - `"pred"`: predicate (i.e. a boolean)
#'   - `"{s,u}{8,16,32,64}"`: (Un)signed integer (for `integer` data).
#'   - `"f{32,64}"`: Floating point (for `double` data).
#' @param ... (any)\cr
#'   Additional arguments.
#' @template param_client
#' @return `PJRTBuffer`
#' @export
pjrt_buffer <- function(data, type, client = default_client(), ...) {
  check_client(client)
  UseMethod("pjrt_buffer")
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar <- function(data, type, client = default_client(), ...) {
  check_client(client)
  UseMethod("pjrt_scalar")
}

#' @export
pjrt_buffer.logical <- function(
  data,
  type = "pred",
  client = default_client(),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  dims = get_dims(data)
  client_buffer_from_logical(data, dims = dims, client = client, type = type)
}

#' @export
pjrt_buffer.integer <- function(
  data,
  type = "s32",
  client = default_client(),
  ...
) {
  dims = get_dims(data)
  if (...length()) {
    stop("Unused arguments")
  }
  client_buffer_from_integer(data, dims = dims, client = client, type = type)
}

#' @export
pjrt_buffer.double <- function(
  data,
  type = "f32",
  client = default_client(),
  ...
) {
  dims = get_dims(data)
  if (...length()) {
    stop("Unused arguments")
  }
  client_buffer_from_double(data, dims = dims, client = client, type = type)
}

#' @export
pjrt_scalar.logical <- function(
  data,
  type = "pred",
  client = default_client(),
  ...
) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (...length()) {
    stop("Unused arguments")
  }
  client_buffer_from_logical(
    data,
    dims = integer(),
    client = client,
    type = type
  )
}

#' @export
pjrt_scalar.integer <- function(
  data,
  type = "s32",
  client = default_client(),
  ...
) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (...length()) {
    stop("Unused arguments")
  }
  client_buffer_from_integer(
    data,
    dims = integer(),
    client = client,
    type = type
  )
}

#' @export
pjrt_scalar.double <- function(
  data,
  type = "f32",
  client = default_client(),
  ...
) {
  if (!is.atomic(data) || length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (...length()) {
    stop("Unused arguments")
  }
  client_buffer_from_double(
    data,
    dims = integer(),
    client = client,
    type = type
  )
}

#' Get the Element Type of a `PJRTBuffer`
#'
#' @param buffer A PJRT buffer object.
#'
#' @return A PJRT element type object.
#' @export
pjrt_element_type <- function(buffer) {
  check_buffer(buffer)
  impl_buffer_element_type(buffer)
}

is_element_type <- function(x) {
  inherits(x, "PJRTElementType")
}

#' Convert `PJRTElementType` to string
#' Get a (lowecase) string representation of a PJRT element type
#'
#' @param x A PJRT element type object.
#' @param ... Additional arguments (unused).
#'
#' @return A string representation of the element type.
#' @export
as.character.PJRTElementType <- function(x, ...) {
  tolower(impl_element_type_as_string(x))
}

#' Dimenson of `PJRTBuffer`
#' Get the dimensions of a PJRT buffer
#'
#' @param x A PJRT buffer object.
#'
#' @return An integer vector of dimensions.
#' @export
dim.PJRTBuffer <- function(x) {
  impl_buffer_dimensions(x)
}

#' @export
print.PJRTElementType <- function(x, ...) {
  cat(sprintf("<%s>\n", as.character(x)))
}

#' Get the platform name of a PJRT client
#'
#' @param client A PJRT client object.
#'
#' @return A string representing the platform name.
#' @export
pjrt_platform_name <- function(client) {
  check_client(client)
  impl_client_platform_name(client)
}

client_buffer_from_integer <- function(
  data,
  type = "f32",
  dims,
  client
) {
  check_client(client)
  impl_client_buffer_from_integer(
    client,
    data,
    dims,
    type
  )
}

client_buffer_from_logical <- function(
  data,
  dims,
  type = "pred",
  client
) {
  check_client(client)
  impl_client_buffer_from_logical(client, data, dims, type)
}

client_buffer_from_double <- function(
  data,
  type = "f32",
  dims,
  client
) {
  check_client(client)
  impl_client_buffer_from_double(
    client,
    data,
    dims,
    type
  )
}

#' Convert a PJRT Buffer to an R object.
#'
#' @description
#' Copy a [`PJRTBuffer`][pjrt_buffer] to an R object.
#' For 0-dimensional PJRT buffers, the R object will be a vector of length 1 and otherwise an array.
#'
#' @details
#' Moving the buffer to the host requires to:
#' 1. Copy the data from the PJRT device to the CPU.
#' 2. Transpose the data, because PJRT returns it in row-major order but R uses column-major order.
#'
#' @template param_buffer
#' @template param_client
#' @export
as_array <- function(buffer, client = default_client()) {
  check_client(client)
  impl_client_buffer_to_host(buffer, client = client)
}

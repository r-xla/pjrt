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
#' @section Extractors:
#' * [`device()`] for the device of the buffer.
#' * [`element_type()`] for the element type of the buffer.
#'
#' @section Converters:
#' * [`as_array()`] for an array.
#' * [`as_raw()`] for a raw vector.
#'
#' @details
#' R does not have 0-dimensional arrays, hence we need the extra `pjrt_scalar` function.
#'
#' @param data (any)\cr
#'  Data to convert to a `PJRTBuffer`.
#' @param elt_type (`character(1)`)\cr
#'   The type of the buffer.
#'   Currently supported types are:
#'   - `"pred"`: predicate (i.e. a boolean)
#'   - `"{s,u}{8,16,32,64}"`: (Un)signed integer (for `integer` data).
#'   - `"f{32,64}"`: Floating point (for `double` or `integer` data).
#' @param ... (any)\cr
#'   Additional arguments.
#' @template param_client
#' @return `PJRTBuffer`
#' @export
pjrt_buffer <- function(data, elt_type, client = pjrt_client(), ...) {
  UseMethod("pjrt_buffer")
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar <- function(data, elt_type, client = pjrt_client(), ...) {
  UseMethod("pjrt_scalar")
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.logical <- function(
  data,
  elt_type = "pred",
  client = pjrt_client(),
  dims = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  if (!is.array(data)) {
    data <- array(data, dim = dims)
  }
  impl_client_buffer_from_logical(
    client = as_pjrt_client(client),
    data = data,
    dims = dims,
    elt_type = elt_type
  )
}

#' @rdname pjrt_buffer
#' @param dims (integer)\cr
#'   The dimensions of the buffer.
#'   If `NULL`, the dimensions will be inferred from the data.
#' @export
pjrt_buffer.integer <- function(
  data,
  elt_type = "i32",
  client = pjrt_client(),
  dims = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  if (!is.array(data)) {
    data <- array(data, dim = dims)
  }
  impl_client_buffer_from_integer(
    client = as_pjrt_client(client),
    data = data,
    dims = get_dims(data),
    elt_type = elt_type
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.double <- function(
  data,
  elt_type = "f32",
  client = pjrt_client(),
  dims = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  if (!is.array(data)) {
    data <- array(data, dim = dims)
  }
  impl_client_buffer_from_double(
    client = as_pjrt_client(client),
    data = data,
    dims = dims,
    elt_type = elt_type
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.raw <- function(
  data,
  ...,
  elt_type,
  client = pjrt_client(),
  shape,
  row_major
) {
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_raw(
    data = data,
    dims = shape,
    client = as_pjrt_client(client),
    elt_type = elt_type,
    row_major = row_major
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.logical <- function(
  data,
  elt_type = "pred",
  client = pjrt_client(),
  ...
) {
  if (length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_logical(
    data,
    dims = integer(),
    client = as_pjrt_client(client),
    elt_type = elt_type
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.integer <- function(
  data,
  elt_type = "i32",
  client = pjrt_client(),
  ...
) {
  if (length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_integer(
    data,
    dims = integer(),
    client = as_pjrt_client(client),
    elt_type = elt_type
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.double <- function(
  data,
  elt_type = "f32",
  client = pjrt_client(),
  ...
) {
  if (length(data) != 1) {
    stop("data must be an atomic vector of length 1")
  }
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_double(
    data,
    dims = integer(),
    client = as_pjrt_client(client),
    elt_type = elt_type
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.raw <- function(
  data,
  ...,
  elt_type,
  client = pjrt_client()
) {
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_raw(
    data,
    dims = integer(),
    client = as_pjrt_client(client),
    elt_type = elt_type,
    row_major = FALSE
  )
}

#' Get the Element Type of a `PJRTBuffer`
#'
#' @param buffer A PJRT buffer object.
#'
#' @return A PJRT element type object.
#' @export
element_type <- function(buffer) {
  check_buffer(buffer)
  impl_buffer_element_type(buffer)
}

#' Gets the memory of a Pjrt buffer
#' @noRd
pjrt_memory <- function(buffer) {
  check_buffer(buffer)
  impl_buffer_memory(buffer)
}

#' @export
print.PJRTMemory <- function(x, ...) {
  cat(sprintf("<PJRTMemory %s>\n", impl_memory_debug_string(x)))
}

is_element_type <- function(x) {
  inherits(x, "PJRTElementType")
}

#' @title Convert `PJRTElementType` to string
#'
#' @description
#' Get a (lowercase) string representation of a PJRT element type
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
platform_name <- function(client = pjrt_client()) {
  client <- as_pjrt_client(client)
  impl_client_platform_name(client)
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
as_array <- function(buffer, client = pjrt_client()) {
  client <- as_pjrt_client(client)
  impl_client_buffer_to_array(client, buffer)
}

#' Convert a PJRT Buffer to a raw R vector.
#'
#' @description
#' Copy a [`PJRTBuffer`][pjrt_buffer] to a raw R vector containing the buffer data as bytes.
#' Any shape information is lost.
#'
#' @template param_buffer
#' @template param_client
#' @param row_major (`logical(1)`)\cr
#'   Whether to return the data in row-major format (TRUE) or column-major format (FALSE).
#'   R uses column-major format.
#' @return `raw()`
#' @export
as_raw <- function(buffer, client = pjrt_client(), row_major) {
  check_buffer(buffer)
  check_client(client)
  impl_client_buffer_to_raw(client, buffer, row_major = row_major)
}

#' Device of a PJRTBuffer
#'
#' @description
#' Get the device of a [`PJRTBuffer`][pjrt_buffer].
#'
#' @param x (any)\cr
#'   Object for which to get the `PJRTDevice`.
#' @return `PJRTDevice`
#' @export
device <- function(x) {
  UseMethod("device")
}

#' @export
device.PJRTBuffer <- function(x) {
  impl_buffer_device(x)
}

#' @export
as.character.PJRTDevice <- function(x, ...) {
  impl_device_to_string(x)
}

#' @export
format.PJRTDevice <- function(x, ...) {
  as.character(x)
}

#' @export
print.PJRTDevice <- function(x, ...) {
  cat(sprintf("<%s>\n", as.character(x)))
}

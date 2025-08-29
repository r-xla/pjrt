assert_buffer <- function(x) {
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
#' [`pjrt_scalar`] will create a 0-dimensional array for an R vector of length 1.
#'
#' @section Extractors:
#' * [`device()`] for the device of the buffer.
#' * [`etype()`] for the element type of the buffer.
#'
#' @section Converters:
#' * [`as_array()`] for an array.
#' * [`as_raw()`] for a raw vector.
#'
#' @section Buffer Overflow:
#' No checks are performed when converting an R object to a PJRT buffer.
#' It is in the caller's responsibility to ensure that the data fits the selected element type.
#'
#' @details
#' R does not have 0-dimensional arrays, hence we need the extra `pjrt_scalar` function.
#'
#' @param data (any)\cr
#'  Data to convert to a `PJRTBuffer`.
#' @param etype (`character(1)`)\cr
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
pjrt_buffer <- function(data, etype, client = pjrt_client(), ...) {
  UseMethod("pjrt_buffer")
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar <- function(data, etype, client = pjrt_client(), ...) {
  UseMethod("pjrt_scalar")
}

#' @rdname pjrt_buffer
#' @param shape (`integer()`)\cr
#'   The dimensions of the buffer.
#'   The default is to infer them from the data.
#' @export
pjrt_buffer.logical <- function(
  data,
  etype = "pred",
  client = pjrt_client(),
  shape = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  if (!is.array(data)) {
    data <- array(data, dim = shape)
  }
  impl_client_buffer_from_logical(
    client = as_pjrt_client(client),
    data = data,
    dims = shape,
    etype = etype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.integer <- function(
  data,
  etype = "i32",
  client = pjrt_client(),
  shape = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  if (!is.array(data)) {
    data <- array(data, dim = shape)
  }
  impl_client_buffer_from_integer(
    client = as_pjrt_client(client),
    data = data,
    dims = get_dims(data),
    etype = etype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.double <- function(
  data,
  etype = "f32",
  client = pjrt_client(),
  shape = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  if (!is.array(data)) {
    data <- array(data, dim = shape)
  }
  impl_client_buffer_from_double(
    client = as_pjrt_client(client),
    data = data,
    dims = shape,
    etype = etype
  )
}

#' @rdname pjrt_buffer
#' @param row_major (logical(1))\cr
#'   Whether to read the data in row-major format or column-major format.
#'   R uses column-major format.
#' @export
pjrt_buffer.raw <- function(
  data,
  ...,
  etype,
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
    etype = etype,
    row_major = row_major
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.logical <- function(
  data,
  etype = "pred",
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
    etype = etype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.integer <- function(
  data,
  etype = "i32",
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
    etype = etype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.double <- function(
  data,
  etype = "f32",
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
    etype = etype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.raw <- function(
  data,
  ...,
  etype,
  client = pjrt_client()
) {
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_raw(
    data,
    dims = integer(),
    client = as_pjrt_client(client),
    etype = etype,
    row_major = FALSE
  )
}

#' Get the Element Type of a `PJRTBuffer`
#'
#' @param buffer A PJRT buffer object.
#'
#' @return A PJRT element type object.
#' @export
etype <- function(buffer) {
  assert_buffer(buffer)
  impl_buffer_etype(buffer)
}

#' Gets the memory of a Pjrt buffer
#' @noRd
pjrt_memory <- function(buffer) {
  assert_buffer(buffer)
  impl_buffer_memory(buffer)
}

#' @export
print.PJRTMemory <- function(x, ...) {
  cat(sprintf("<PJRTMemory %s>\n", impl_memory_debug_string(x)))
}

is_etype <- function(x) {
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
  tolower(impl_etype_as_string(x))
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
  assert_buffer(buffer)
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

#' @export
print.PJRTBuffer <- function(
  x,
  n = getOption("pjrt.print_max", 30L),
  max_width = getOption("pjrt.print_width", 85L),
  max_rows = getOption("pjrt.print_rows", 30L),
  print_dtype = TRUE,
  ...
) {
  if (print_dtype) {
    cat(sprintf("PJRTBuffer%s", dtype(x)), "\n")
  }
  impl_buffer_print(
    x,
    n = as.integer(n),
    max_width = as.integer(max_width),
    max_rows = as.integer(max_rows)
  )
  invisible(x)
}


#' @export
as.character.PJRTDtype <- function(x, ...) {
  sprintf("<%s: %s>", x$etype, paste(x$shape, collapse = ","))
}

#' @title Get the data type of a PJRTBuffer
#' @description
#' Get the data type of a PJRTBuffer.
#'
#' @param x [`PJRTBuffer`][pjrt_buffer]
#'
#' @return A `PJRTDtype` object.
#' @export
dtype <- function(x) {
  assert_buffer(x)
  structure(
    list(
      etype = etype(x),
      shape = shape(x)
    ),
    class = "PJRTDtype"
  )
}


#' @export
print.PJRTDtype <- function(x, ...) {
  cat(as.character(x), "\n")
}


#' @title Get the shape of a PJRTBuffer
#' @description
#' Get the shape of a PJRTBuffer.
#'
#' @param x (any)\cr
#'
#' @return `integer()`
#' @export
shape <- function(x) {
  UseMethod("shape")
}

#' @export
shape.PJRTBuffer <- function(x) {
  impl_buffer_dimensions(x)
}

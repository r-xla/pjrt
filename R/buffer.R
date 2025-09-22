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
#' [`pjrt_scalar`] will create a 0-dimensional array for an R vector of length 1.
#'
#' **Important**:
#' No checks are performed when creating the buffer, so you need to ensure that the data fits
#' the selected element type (e.g., to prevent buffer overflow) and that no NA values are present.
#'
#' @section Extractors:
#' * [`device()`] for the device of the buffer.
#' * [`dtype()`] for the element type of the buffer.
#' * [`shape()`] for the shape of the buffer.
#'
#' @section Converters:
#' * [`as_array()`] for an array.
#' * [`as_raw()`] for a raw vector.
#'
#' @section Scalars:
#' When calling this function on a vector of length 1, the resulting shape is `1L`.
#' To create a 0-dimensional buffer, use `pjrt_scalar` where the resulting shape is `integer()`.
#'
#' @param data (any)\cr
#'  Data to convert to a `PJRTBuffer`.
#' @param dtype (`character(1)`)\cr
#'   The type of the buffer.
#'   Currently supported types are:
#'   - `"pred"`: predicate (i.e. a boolean)
#'   - `"{s,u}{8,16,32,64}"`: Signed and unsigned integer (for `integer` data).
#'   - `"f{32,64}"`: Floating point (for `double` or `integer` data).
#' @param ... (any)\cr
#'   Additional arguments.
#' @template param_client
#' @return `PJRTBuffer`
#' @export
pjrt_buffer <- function(data, dtype, client = pjrt_client(), ...) {
  UseMethod("pjrt_buffer")
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar <- function(data, dtype, client = pjrt_client(), ...) {
  UseMethod("pjrt_scalar")
}

assert_data_shape <- function(data, shape) {
  data_len <- length(data)
  numel <- prod(shape)

  if (data_len == numel) {
    return(NULL)
  } else if ((data_len == 1) && (numel != 0)) {
    return(NULL)
  } else {
    stop(
      "Data has length ",
      data_len,
      ", but shape is ",
      paste0(shape, collapse = "x")
    )
  }
}

#' @rdname pjrt_buffer
#' @param shape (`integer()`)\cr
#'   The dimensions of the buffer.
#'   The default is to infer them from the data.
#' @export
pjrt_buffer.logical <- function(
  data,
  dtype = "pred",
  client = pjrt_client(),
  shape = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  assert_data_shape(data, shape)
  if (!is.array(data)) {
    data <- array(data, dim = shape)
  }
  impl_client_buffer_from_logical(
    client = as_pjrt_client(client),
    data = data,
    dims = shape,
    dtype = dtype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.integer <- function(
  data,
  dtype = "i32",
  client = pjrt_client(),
  shape = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  assert_data_shape(data, shape)
  if (!is.array(data)) {
    data <- array(data, dim = shape)
  }
  impl_client_buffer_from_integer(
    client = as_pjrt_client(client),
    data = data,
    dims = get_dims(data),
    dtype = dtype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_buffer.double <- function(
  data,
  dtype = "f32",
  client = pjrt_client(),
  shape = get_dims(data),
  ...
) {
  if (...length()) {
    stop("Unused arguments")
  }
  assert_data_shape(data, shape)
  if (!is.array(data)) {
    data <- array(data, dim = shape)
  }
  impl_client_buffer_from_double(
    client = as_pjrt_client(client),
    data = data,
    dims = shape,
    dtype = dtype
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
  dtype,
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
    dtype = dtype,
    row_major = row_major
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.logical <- function(
  data,
  dtype = "pred",
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
    dtype = dtype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.integer <- function(
  data,
  dtype = "i32",
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
    dtype = dtype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.double <- function(
  data,
  dtype = "f32",
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
    dtype = dtype
  )
}

#' @rdname pjrt_buffer
#' @export
pjrt_scalar.raw <- function(
  data,
  ...,
  dtype,
  client = pjrt_client()
) {
  if (...length()) {
    stop("Unused arguments")
  }
  impl_client_buffer_from_raw(
    data,
    dims = integer(),
    client = as_pjrt_client(client),
    dtype = dtype,
    row_major = FALSE
  )
}

method(dtype, S7::new_S3_class("PJRTBuffer")) <- function(x) {
  impl_buffer_dtype(x)
}

method(as_array, S7::new_S3_class("PJRTBuffer")) <- function(x, client = pjrt_client(), ...) {
  client <- as_pjrt_client(client)
  impl_client_buffer_to_array(client, x)
}

method(as_raw, S7::new_S3_class("PJRTBuffer")) <- function(x, client = pjrt_client(), row_major, ...) {
  client <- as_pjrt_client(client)
  assert_flag(row_major)
  impl_client_buffer_to_raw(client, x, row_major = row_major)
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

is_dtype <- function(x) {
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
  tolower(impl_dtype_as_string(x))
}

#' @export
print.PJRTElementType <- function(x, ...) {
  cat(sprintf("<%s>\n", as.character(x)))
}

method(device, S7::new_S3_class("PJRTBuffer")) <- function(x) {
  impl_buffer_device(x)
}


#' @include client.R
S7::method(platform, S7::new_S3_class("PJRTBuffer")) <- function(x) {
  desc <- as.character(device(x))
  # Known prefixes map to platform names
  # e.g., "CpuDevice(id=0)", "CudaDevice(id=0)", "MetalDevice(id=0)"
  prefix <- tolower(regmatches(desc, regexpr("^[A-Za-z]+(?=Device)", desc, perl = TRUE)))
  tolower(prefix)
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

#' @title Print a PJRT Buffer
#' @description
#' Print a [`PJRTBuffer`][pjrt_buffer].
#' @param x (`PJRTBuffer`)\cr
#'   The buffer.
#' @param max_rows (`integer(1)`)\cr
#'   The maximum number of rows to print, excluding header and footer.
#' @param max_width (`integer(1)`)\cr
#'   The maximum width (in characters) of the printed buffer.
#'   Set to negative values for no limit.
#'   Note that for very small values, the actual printed width might be slightly smaller
#'   as at least one column will be printed.
#'   Also, this limit only affects the printed rows containing the actual data,
#'   other rows might exceed the width.
#' @param max_rows_slice (`integer(1)`)\cr
#'   The maximum number of rows to print for each slice.
#' @param header (`logical(1)`)\cr
#'   Whether to print the header.
#' @param ... Additional arguments (unused).
#' @export
print.PJRTBuffer <- function(
  x,
  max_rows = getOption("pjrt.print_max_rows", 30L),
  max_width = getOption("pjrt.print_max_width", 85L),
  max_rows_slice = getOption("pjrt.print_max_rows_slice", max_rows),
  header = TRUE,
  ...
) {
  assert_flag(header)
  max_rows <- assert_int(max_rows, coerce = TRUE, lower = 1L)
  max_width <- assert_int(max_width, coerce = TRUE)
  if (max_width %in% c(0, 1L)) {
    # we disallow 1, because every data line starts with ' '
    stop("Either provide a negative value for max_width or a value > 1")
  }
  max_rows_slice <- assert_int(max_rows_slice, coerce = TRUE, lower = 1L)

  if (header) {
    shp <- shape(x)
    shape_str <- if (length(shp)) {
      paste0(": ", paste0(shp, collapse = "x"))
    } else {
      ""
    }
    cat(sprintf("PJRTBuffer<%s%s>", dtype(x), shape_str), "\n")
  }
  impl_buffer_print(
    x,
    max_rows = max_rows,
    max_width = max_width,
    max_rows_slice = max_rows_slice
  )
  invisible(x)
}


S7::method(shape, S7::new_S3_class("PJRTBuffer")) <- function(x) {
  impl_buffer_dimensions(x)
}

#' @export
`==.PJRTElementType` <- function(e1, e2) {
  identical(as.character(e1), as.character(e2))
}

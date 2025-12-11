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
#' To create an empty buffer (at least one dimension must be 0), use [`pjrt_empty`].
#'
#' **Important**:
#' No checks are performed when creating the buffer, so you need to ensure that the data fits
#' the selected element type (e.g., to prevent buffer overflow) and that no NA values are present.
#'
#' @section Extractors:
#' * [`platform()`] -> `character(1)`: for the platform name of the buffer (`"cpu"`, `"cuda"`, ...).
#' * [`device()`] -> `PJRTDevice`: for the device of the buffer (also includes device number)
#' * [`elt_type()`] -> `PJRTElementType`: for the element type of the buffer.
#' * [`shape()`] -> `integer()`: for the shape of the buffer.
#'
#' @section Converters:
#' * [`as_array()`] -> `array` | `vector`: for converting back to R (`vector` is only used for shape `integer()`).
#' * [`as_raw()`] -> `raw` for a raw vector.
#'
#' @section Reading and Writing:
#' * [`safetensors::safe_save_file`] for writing to a safetensors file.
#' * [`safetensors::safe_load_file`] for reading from a safetensors file.
#'
#' @section Scalars:
#' When calling this function on a vector of length 1, the resulting shape is `1L`.
#' To create a 0-dimensional buffer, use `pjrt_scalar` where the resulting shape is `integer()`.
#'
#' @param data (any)\cr
#'  Data to convert to a `PJRTBuffer`.
#' @param dtype (`NULL` | `character(1)`)\cr
#'   The type of the buffer.
#'   Currently supported types are:
#'   - `"pred"`: predicate (i.e. a boolean)
#'   - `"{s,u}{8,16,32,64}"`: Signed and unsigned integer (for `integer` data).
#'   - `"f{32,64}"`: Floating point (for `double` or `integer` data).
#'   The default (`NULL`) depends on the method:
#'   - `logical` -> `"pred"`
#'   - `integer` -> `"i32"`
#'   - `double` -> `"f32"`
#'   - `raw` -> must be supplied
#' @param shape (`NULL` | `integer()`)\cr
#'   The dimensions of the buffer.
#'   The default (`NULL`) is to infer them from the data if possible.
#'   The default (`NULL`) depends on the method.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr
#'   A `PJRTDevice` object or the name of the platform to use ("cpu", "cuda", ...), in which
#'   case the first device for that platform is used.
#'   The default is to use the CPU platform, but this can be configured via the `PJRT_PLATFORM`
#'   environment variable.
#' @param ... (any)\cr
#'   Additional arguments.
#'   For `raw` types, this includes:
#'   - `row_major`: Whether to read the data in row-major format or column-major format.
#'     R uses column-major format.
#' @return `PJRTBuffer`
#' @examplesIf plugin_is_downloaded()
#' # Create a buffer from a numeric vector
#' buf <- pjrt_buffer(c(1, 2, 3, 4))
#' buf
#'
#' # Create a buffer from a matrix
#' mat <- matrix(1:6, nrow = 2)
#' buf <- pjrt_buffer(mat)
#' buf
#'
#' # Create an integer buffer from an array
#' arr <- array(1:8, dim = c(2, 2, 2))
#' buf <- pjrt_buffer(arr)
#'
#' @export
pjrt_buffer <- S7::new_generic(
  "pjrt_buffer",
  "data",
  function(data, dtype = NULL, device = NULL, shape = NULL, ...) {
    S7::S7_dispatch()
  }
)

buffer_identity <- function(data, dtype = NULL, device = NULL, shape = NULL, ...) {
  if (!is.null(dtype) && !identical(dtype, as.character(elt_type(data)))) {
    cli_abort("Must use the same data type as the data")
  }
  if (!is.null(device)) {
    device <- as_pjrt_device(device)
    buf_dev <- device(data)
    if (device != buf_dev) {
      cli_abort("Must use the same device as the data")
    }
  }
  if (!is.null(shape) && !identical(shape, shape(data))) {
    cli_abort("Must use the same shape as the data")
  }
  data
}

method(pjrt_buffer, S7::new_S3_class("PJRTBuffer")) <- buffer_identity

#' @rdname pjrt_buffer
#' @examplesIf plugin_is_downloaded()
#' # Create a scalar (0-dimensional array)
#' scalar <- pjrt_scalar(42, dtype = "f32")
#' scalar
#' @export
pjrt_scalar <- S7::new_generic("pjrt_scalar", "data", function(data, dtype = NULL, device = NULL, ...) {
  S7::S7_dispatch()
})

method(pjrt_scalar, S7::new_S3_class("PJRTBuffer")) <- function(data, dtype = NULL, device = NULL, ...) {
  buffer_identity(data, dtype, device, shape = integer())
}

#' @rdname pjrt_buffer
#' @examplesIf plugin_is_downloaded()
#' # Create an empty buffer
#' empty <- pjrt_empty(dtype = "f32", shape = c(0, 3))
#' empty
#' @export
pjrt_empty <- function(dtype, shape, device = NULL) {
  if (!any(shape == 0)) {
    cli_abort("Empty buffers must have at least one dimension equal to 0")
  }
  if (dtype == "i1") {
    dtype <- "pred"
  }
  device <- as_pjrt_device(device)
  data <- if (identical(dtype, "pred")) {
    logical()
  } else {
    integer()
  }
  pjrt_buffer(array(data, dim = shape), dtype, device)
}

recycle_data <- function(data, shape) {
  data_len <- length(data)
  numel <- if (length(shape)) prod(shape) else 1L

  if (numel == 0) {
    if (!any(shape == 0)) {
      cli_abort("Empty buffers must have at least one dimension equal to 0")
    }
    array(data, dim = shape)
  } else if (data_len == numel) {
    return(data)
  } else if ((data_len == 1) && (numel != 0)) {
    rep(data, numel)
  } else {
    cli_abort(
      "Data has length {data_len}, but specified shape is {paste0(shape, collapse = 'x')}"
    )
  }
}

convert_buffer_args <- function(data, dtype, device, shape, default, recycle = TRUE, ...) {
  dtype <- dtype %??% default
  if (dtype == "i1") {
    dtype <- "pred"
  }
  shape <- shape %??% get_dims(data)
  device <- as_pjrt_device(device)
  client <- client_from_device(device)
  if (...length()) {
    cli_abort("Unused arguments")
  }
  if (recycle) {
    data <- recycle_data(data, shape)
  }
  list(
    dtype = dtype,
    client = client,
    device = device,
    data = data,
    dims = shape
  )
}

S7::method(pjrt_buffer, S7::class_logical) <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  ...
) {
  do.call(impl_client_buffer_from_logical, convert_buffer_args(data, dtype, device, shape, "pred", ...))
}

S7::method(pjrt_buffer, S7::class_integer) <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  ...
) {
  do.call(impl_client_buffer_from_integer, convert_buffer_args(data, dtype, device, shape, "i32", ...))
}

S7::method(pjrt_buffer, S7::class_double) <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  ...
) {
  do.call(impl_client_buffer_from_double, convert_buffer_args(data, dtype, device, shape, "f32", ...))
}

S7::method(pjrt_buffer, S7::class_raw) <- function(
  data,
  ...,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  row_major
) {
  if (is.null(shape)) {
    cli_abort("shape must be provided")
  }
  if (is.null(dtype)) {
    cli_abort("dtype must be provided")
  }
  if (...length()) {
    cli_abort("Unused arguments")
  }
  device <- as_pjrt_device(device)
  client <- client_from_device(device)
  impl_client_buffer_from_raw(
    client = client,
    device = device,
    data = data,
    dims = shape,
    dtype = dtype,
    row_major = row_major
  )
}

S7::method(pjrt_scalar, S7::class_logical) <- function(
  data,
  dtype = NULL,
  device = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  do.call(impl_client_buffer_from_logical, convert_buffer_args(data, dtype, device, integer(), "pred", ...))
}

S7::method(pjrt_scalar, S7::class_integer) <- function(
  data,
  dtype = NULL,
  device = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  do.call(impl_client_buffer_from_integer, convert_buffer_args(data, dtype, device, integer(), "i32", ...))
}

S7::method(pjrt_scalar, S7::class_double) <- function(
  data,
  dtype = NULL,
  device = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  do.call(impl_client_buffer_from_double, convert_buffer_args(data, dtype, device, integer(), "f32", ...))
}

S7::method(pjrt_scalar, S7::class_raw) <- function(
  data,
  ...,
  dtype = NULL,
  device = NULL
) {
  if (is.null(dtype)) {
    cli_abort("dtype must be provided")
  }
  do.call(impl_client_buffer_from_raw, convert_buffer_args(data, dtype, device, integer(), "f32", recycle = FALSE, ...))
}

#' @title Element Type
#' @description
#' Get the element type of a buffer.
#' @param x ([`PJRTBuffer`][pjrt_buffer])\cr
#'   Buffer.
#' @examplesIf plugin_is_downloaded("cpu")
#' buf <- pjrt_buffer(c(1.0, 2.0, 3.0))
#' elt_type(buf)
#' @export
elt_type <- function(x) {
  impl_buffer_elt_type(x)
}

#' @export
as_array.PJRTBuffer <- function(x, client = NULL, ...) {
  client <- as_pjrt_client(client)
  impl_client_buffer_to_array(client, x)
}

#' @export
as_raw.PJRTBuffer <- function(x, client = NULL, row_major, ...) {
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

is_elt_type <- function(x) {
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

#' @export
device.PJRTBuffer <- function(x, ...) {
  impl_buffer_device(x)
}


#' @include client.R
S7::method(platform, S7::new_S3_class("PJRTBuffer")) <- function(x) {
  desc <- as.character(device(x))
  letters_only <- regmatches(desc, regexpr("^[A-Za-z]+", desc, perl = TRUE))
  tolower(sub("Device$", "", letters_only))
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
    cli_abort("Either provide a negative value for max_width or a value > 1")
  }
  max_rows_slice <- assert_int(max_rows_slice, coerce = TRUE, lower = 1L)

  if (header) {
    cat(class(x)[[1L]], "\n")
  }
  shp <- shape(x)
  shape_str <- if (length(shp)) {
    paste0(paste0(shp, collapse = "x"))
  } else {
    ""
  }
  impl_buffer_print(
    x,
    max_rows = max_rows,
    max_width = max_width,
    max_rows_slice = max_rows_slice
  )
  cat(sprintf("[ %s%s{%s} ]", toupper(platform(x)), elt_type(x), shape_str), "\n")
  invisible(x)
}

#' @export
shape.PJRTBuffer <- function(x, ...) {
  impl_buffer_dimensions(x)
}

#' @export
`==.PJRTDevice` <- function(e1, e2) {
  if (!inherits(e2, "PJRTDevice")) {
    return(FALSE)
  }
  identical(as.character(e1), as.character(e2))
}

#' @export
`!=.PJRTDevice` <- function(e1, e2) {
  !(e1 == e2) # nolint
}

#' @export
`==.PJRTElementType` <- function(e1, e2) {
  identical(as.character(e1), as.character(e2))
}

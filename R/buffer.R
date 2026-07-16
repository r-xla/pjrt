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
#' @param dtype (`NULL` | `character(1)` | [`DataType`][tengen::DataType])\cr
#'   The type of the buffer.
#'   Currently supported types are:
#'   - `"pred"`: predicate (i.e. a boolean)
#'   - `"{s,u}{8,16,32,64}"`: Signed and unsigned integer (for `integer` data).
#'   - `"f{16,32,64}"`: Floating point (for `double` or `integer` data).
#'     For `"f16"`, values are rounded to the nearest binary16 value (ties to
#'     even); values beyond the finite range (65504) become `Inf`.
#'     [`as_array()`] returns the exactly representable values as `double`.
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
#' @param check (`logical(1)`)\cr
#'   If `TRUE`, scan `data` for `NA` values before transferring to the device and
#'   raise an error if any are present. R's `NA` markers have no representation
#'   at the XLA level (e.g. `NA_integer_` is just the bit pattern `-2147483648`,
#'   and `NA` of `logical` type is silently coerced to `TRUE`), so missing values
#'   are silently lost on transfer. Defaults to `FALSE` for performance; set to
#'   `TRUE` to fail loudly instead of silently corrupting data.
#'   Not applicable to `raw` input.
#' @param ... (any)\cr
#'   Additional arguments.
#'   For `raw` types, this includes:
#'   - `row_major`: Whether to read the data in row-major format or column-major format.
#'     R uses column-major format.
#' @return `PJRTBuffer`
#' @examplesIf plugins_downloaded()
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
pjrt_buffer <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = FALSE,
  ...
) {
  UseMethod("pjrt_buffer")
}

check_input_na <- function(data, check) {
  assert_flag(check)
  if (check && anyNA(data)) {
    n_na <- sum(is.na(data))
    cli_abort(c(
      "Input {.arg data} contains {n_na} {.val NA} value{?s}, which {?has/have} no representation at the XLA level.",
      i = "Replace or drop missing values before transferring, or set {.code check = FALSE} to skip this check."
    ))
  }
  invisible(NULL)
}

buffer_identity <- function(data, dtype = NULL, device = NULL, shape = NULL, ...) {
  buf <- data
  if (inherits(dtype, "DataType")) {
    dtype <- as.character(dtype)
  }
  if (!is.null(dtype) && !identical(dtype, as.character(elt_type(buf)))) {
    cli_abort("Must use the same data type as the data")
  }
  if (!is.null(device)) {
    device <- as_pjrt_device(device)
    buf_dev <- device(buf)
    if (device != buf_dev) {
      cli_abort("Must use the same device as the data")
    }
  }
  if (!is.null(shape) && !identical(shape, shape(buf))) {
    cli_abort("Must use the same shape as the data")
  }
  data
}

#' @export
pjrt_buffer.PJRTBuffer <- buffer_identity


#' @rdname pjrt_buffer
#' @examplesIf plugins_downloaded()
#' # Create a scalar (0-dimensional array)
#' scalar <- pjrt_scalar(42, dtype = "f32")
#' scalar
#' @export
pjrt_scalar <- function(data, dtype = NULL, device = NULL, check = FALSE, ...) {
  UseMethod("pjrt_scalar")
}

#' @export
pjrt_scalar.PJRTBuffer <- function(data, dtype = NULL, device = NULL, ...) {
  buffer_identity(data, dtype, device, shape = integer())
}


#' @rdname pjrt_buffer
#' @description
#' `pjrt_empty()` allocates a buffer of the given `shape` and `dtype` with
#' **unspecified contents**. The bytes should be treated as uninitialized —
#' read them only after they have been written to (e.g. as a donated output
#' of [`pjrt_execute()`]). Shapes with at least one zero-sized dimension
#' are supported as a degenerate case (the buffer holds zero elements).
#' @examplesIf plugins_downloaded()
#' # Allocate an uninitialized 2x3 f32 buffer (contents are unspecified)
#' empty <- pjrt_empty(dtype = "f32", shape = c(2, 3))
#' empty
#' @export
pjrt_empty <- function(dtype, shape, device = NULL) {
  dtype <- as_dtype_string(dtype)
  device <- as_pjrt_device(device)
  client <- client_from_device(device)
  impl_client_buffer_empty(client, device, as.integer(shape), dtype)
}

check_raw_buffer_size <- function(data, dtype, shape) {
  element_size <- pjrt_dtype_size(dtype)
  numel <- if (length(shape)) prod(shape) else 1L
  expected_bytes <- numel * element_size
  actual_bytes <- length(data)
  if (actual_bytes != expected_bytes) {
    cli_abort(
      "Raw data has {actual_bytes} byte{?s}, but dtype {.val {dtype}} with shape ({paste(shape, collapse = ', ')}) requires {expected_bytes} byte{?s}."
    )
  }
}

recycle_data <- function(data, shape) {
  data_len <- length(data)
  numel <- if (length(shape)) prod(shape) else 1L

  if (numel == 0) {
    if (!any(shape == 0)) {
      cli_abort("Empty buffers must have at least one dimension equal to 0")
    }
    out <- array(data, dim = shape)
    oldClass(out) <- oldClass(data)
    out
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

as_dtype_string <- function(dtype) {
  if (inherits(dtype, "DataType")) {
    dtype <- as.character(dtype)
  }
  if (dtype %in% c("i1", "bool")) {
    dtype <- "pred"
  }
  dtype
}

convert_buffer_args <- function(data, dtype, device, shape, default, recycle = TRUE, ...) {
  dtype <- as_dtype_string(dtype %||% default)
  shape <- shape %||% get_dims(data)
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

#' @export
pjrt_buffer.logical <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = FALSE,
  ...
) {
  check_input_na(data, check)
  args <- convert_buffer_args(data, dtype, device, shape, "pred", ...)
  buffer <- do.call(impl_client_buffer_from_logical, args)
  buffer
}

#' @export
pjrt_buffer.integer <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = FALSE,
  ...
) {
  check_input_na(data, check)
  args <- convert_buffer_args(data, dtype, device, shape, "i32", ...)
  buffer <- do.call(impl_client_buffer_from_integer, args)
  buffer
}

#' @export
pjrt_buffer.numeric <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = FALSE,
  ...
) {
  check_input_na(data, check)
  args <- convert_buffer_args(data, dtype, device, shape, "f32", ...)
  buffer <- do.call(impl_client_buffer_from_double, args)
  buffer
}

#' @export
pjrt_buffer.integer64 <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  ...
) {
  args <- convert_buffer_args(data, dtype, device, shape, "i64", ...)
  if (!args$dtype %in% c("i64", "ui64")) {
    cli_abort(
      "{.cls integer64} input only supports {.val i64} or {.val ui64} dtype, got {.val {args$dtype}}."
    )
  }
  impl_client_buffer_from_integer64(
    client = args$client,
    device = args$device,
    data = args$data,
    dims = args$dims,
    dtype = args$dtype
  )
}

#' @export
pjrt_buffer.raw <- function(
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
  dtype <- as_dtype_string(dtype)
  if (...length()) {
    cli_abort("Unused arguments")
  }
  check_raw_buffer_size(data, dtype, shape)
  device <- as_pjrt_device(device)
  client <- client_from_device(device)
  buffer <- impl_client_buffer_from_raw(
    client = client,
    device = device,
    data = data,
    dims = shape,
    dtype = dtype,
    row_major = row_major
  )
  buffer
}

#' @export
pjrt_scalar.logical <- function(
  data,
  dtype = NULL,
  device = NULL,
  check = FALSE,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_input_na(data, check)
  args <- convert_buffer_args(data, dtype, device, integer(), "pred", ...)
  buffer <- do.call(impl_client_buffer_from_logical, args)
  buffer
}

#' @export
pjrt_scalar.integer <- function(
  data,
  dtype = NULL,
  device = NULL,
  check = FALSE,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_input_na(data, check)
  args <- convert_buffer_args(data, dtype, device, integer(), "i32", ...)
  buffer <- do.call(impl_client_buffer_from_integer, args)
  buffer
}

#' @export
pjrt_scalar.numeric <- function(
  data,
  dtype = NULL,
  device = NULL,
  check = FALSE,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_input_na(data, check)
  args <- convert_buffer_args(data, dtype, device, integer(), "f32", ...)
  buffer <- do.call(impl_client_buffer_from_double, args)
  buffer
}

#' @export
pjrt_scalar.integer64 <- function(
  data,
  dtype = NULL,
  device = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  pjrt_buffer.integer64(data, dtype = dtype, device = device, shape = integer(), ...)
}

#' @export
pjrt_scalar.raw <- function(
  data,
  ...,
  dtype = NULL,
  device = NULL
) {
  if (is.null(dtype)) {
    cli_abort("dtype must be provided")
  }
  dtype <- as_dtype_string(dtype)
  check_raw_buffer_size(data, dtype, integer())
  args <- convert_buffer_args(data, dtype, device, integer(), "f32", recycle = FALSE, ...)
  buffer <- do.call(impl_client_buffer_from_raw, args)
  buffer
}

#' @title Element Type
#' @description
#' Get the element type of a buffer.
#' @param x ([`PJRTBuffer`][pjrt_buffer])\cr
#'   Buffer.
#' @examplesIf plugins_downloaded("cpu")
#' buf <- pjrt_buffer(c(1.0, 2.0, 3.0))
#' elt_type(buf)
#' @export
elt_type <- function(x) {
  impl_buffer_elt_type(x)
}

#' @rdname as_array.PJRTBuffer
#' @title Convert a PJRTBuffer to an R Array
#' @description
#' Transfer buffer data from device to host and return an R array.
#'
#' @param x ([`PJRTBuffer`][pjrt_buffer])\cr
#'   Buffer to convert.
#' @param check (`logical(1)`)\cr
#'   If `TRUE`, sanity-check the materialized R vector against losing
#'   information across the device-to-host boundary, and abort if any
#'   problematic value is detected:
#'   * **`i32` / `i64`**: any `NA` in the result. R's `NA_integer_` shares
#'     the bit pattern `INT_MIN`; `bit64`'s `NA_integer64_` shares
#'     `INT64_MIN`. A legitimate device value at those bit patterns is
#'     indistinguishable from `NA` once materialized in R.
#'   * **`ui64`**: any negative value in the result. `ui64` is stored as
#'     `bit64::integer64` (signed 64-bit), which wraps values `>= 2^63`
#'     to negative — exactly `2^63` becomes `NA_integer64_`, anything
#'     above becomes a non-NA negative integer64.
#'
#'   No-op for float, boolean, and small/unsigned-32 integer dtypes —
#'   `ui32` is now stored as `integer64` and has full headroom, so it
#'   cannot produce a wrapped or NA value.
#' @param ... Additional arguments (unused).
#' @return An R `array` (or `vector` for shape `integer()`).
#' @export
as_array.PJRTBuffer <- function(x, check = FALSE, ...) {
  result <- value(as_array_async(x))
  assert_flag(check)
  if (check) {
    dt <- as.character(elt_type(x))
    if (dt %in% c("i32", "i64") && anyNA(result)) {
      cli_abort(c(
        "Materialized {.cls {dt}} buffer contains a value that R cannot distinguish from {.val NA}.",
        i = "{.val i32} reserves the bit pattern {.val -2147483648} ({.code INT_MIN}); {.val i64} reserves {.val -9223372036854775808} ({.code INT64_MIN}).",
        i = "Set {.code check = FALSE} to skip this check."
      ))
    } else if (identical(dt, "ui64") && (anyNA(result) || any(result < 0, na.rm = TRUE))) {
      # ui64 values >= 2^63 wrap when stored as signed int64 — exactly 2^63
      # becomes NA_integer64_ (INT64_MIN); 2^63 + k becomes a non-NA negative
      # int64. Either way, the unsigned magnitude was lost.
      # (ui32 is now materialized as integer64 and has full headroom, so it
      # cannot produce a negative value; no check needed.)
      cli_abort(c(
        "Materialized {.cls ui64} buffer contains a value `>= 2^63` that wrapped through R's signed {.cls integer64}.",
        i = "Exactly {.code 2^63} becomes {.code NA_integer64_}; larger values become negative {.cls integer64}.",
        i = "Set {.code check = FALSE} to skip this check."
      ))
    }
  }
  result
}

#' @title Convert buffer to R array asynchronously
#' @description
#' Start an asynchronous transfer of buffer data from device to host.
#' Returns immediately with a `PJRTArrayPromise` object.
#'
#' Use `value()` to get the R array (blocks if not ready).
#' Use `is_ready()` to check if transfer has completed (non-blocking).
#'
#' @param x A `PJRTBuffer` object.
#' @param ... Additional arguments (unused).
#' @return A `PJRTArrayPromise` object. Call `value()` to get the R array.
#' @seealso [as_array()], [value()], [is_ready()], [pjrt_execute()], [await()]
#' @examplesIf plugins_downloaded()
#' buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
#' result <- as_array_async(buf)
#' is_ready(result)
#' value(result)
#' @export
as_array_async <- function(x, ...) {
  UseMethod("as_array_async")
}

#' @export
as_array_async.PJRTBuffer <- function(x, ...) {
  result <- impl_buffer_to_host_async(x)
  pjrt_array_promise(result$data, result$dtype, result$dims, result$minor_to_major)
}


#' @export
as_raw.PJRTBuffer <- function(x, row_major, ...) {
  assert_flag(row_major)
  client <- client_from_device(device(x))
  impl_buffer_to_raw(client, x, row_major = row_major)
}

#' @title Copy Buffer to Device
#' @description
#' Copy a [`PJRTBuffer`][pjrt_buffer] to a different device.
#' Returns a new buffer on the target device; the original buffer is unchanged.
#'
#' If the buffer already lives in the requested device, no copy is performed.
#'
#' When the target device belongs to a different client (e.g. copying from CPU
#' to CUDA), the transfer is performed via a host roundtrip.
#'
#' @template param_buffer
#' @param device (`PJRTDevice` | `character(1)`)\cr
#'   The target device. A `PJRTDevice` object or a device specification
#'   (e.g., `"cpu:0"`, `"cpu:1"`, `"cuda:0"`).
#' @return A new `PJRTBuffer` on the target device.
#' @examplesIf plugins_downloaded(c("cpu", "cuda"))
#' buf <- pjrt_buffer(c(1, 2, 3), device = "cpu")
#' buf2 <- copy_buffer(buf, "cuda")
#' device(buf2)
#' @export
copy_buffer <- function(buffer, device) {
  check_buffer(buffer)
  device <- as_pjrt_device(device)
  if (device(buffer) == device) {
    return(buffer)
  }
  cross_client <- platform(device(buffer)) != platform(device)
  dst_client <- client_from_device(device)
  impl_buffer_copy_to_device(buffer, device, dst_client, cross_client)
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
  cached_device(impl_buffer_device(x))
}


#' @include client.R
#' @export
platform.PJRTBuffer <- function(x, ...) {
  platform_from_device_string(as.character(device(x)))
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
#' @param footer (`NULL` or `character(1)`)\cr
#'   The footer line to print. If `NULL` (default), prints the standard
#'   `[ <PLATFORM><TYPE>{<SHAPE>} ]` summary. Use `""` to suppress.
#' @param ... Additional arguments (unused).
#' @export
print.PJRTBuffer <- function(
  x,
  max_rows = getOption("pjrt.print_max_rows", 30L),
  max_width = getOption("pjrt.print_max_width", 85L),
  max_rows_slice = getOption("pjrt.print_max_rows_slice", max_rows),
  header = TRUE,
  footer = NULL,
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
  if (is.null(footer)) {
    footer <- sprintf("[ %s%s{%s} ]", toupper(platform(x)), dtype_display_name(x), shape_str)
  }
  if (nzchar(footer)) {
    cat(footer, "\n")
  }
  invisible(x)
}

#' @export
dtype.PJRTBuffer <- function(x, ...) {
  as_dtype(as.character(elt_type(x)))
}

# The dtype name shown in the print footer. Same rule as dtype_display_name()
# in src/ffi.cpp: pjrt's element-type strings, with the boolean type shown
# under tengen's spelling ("bool", not "pred"). Derived from elt_type() rather
# than dtype() because the buffer layer may support dtypes tengen cannot
# express yet (e.g. "f16"), and the footer must still print for those.
dtype_display_name <- function(x) {
  name <- as.character(elt_type(x))
  if (name == "pred") {
    return("bool")
  }
  name
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
  # Canonical devices (from `cached_device()`) share one xptr, so equal devices
  # are usually identical -- a pointer compare that skips stringification.
  if (identical(e1, e2)) {
    return(TRUE)
  }
  identical(as.character(e1), as.character(e2))
}

#' @export
`!=.PJRTDevice` <- function(e1, e2) {
  # jarl-ignore comparison_negation: delegates to ==.PJRTDevice
  !(e1 == e2) # nolint
}

#' @export
`==.PJRTElementType` <- function(e1, e2) {
  identical(as.character(e1), as.character(e2))
}

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
#' Uploading a numeric vector at an integer element type rejects any value that
#' type cannot hold: one outside its range, or a missing value, is an error
#' rather than a wrapped or clamped result. A fractional value is *not* an
#' error -- it truncates toward zero, as [`as.integer()`] does -- and the range
#' is checked on the truncated value, so `255.7` still fits `"ui8"`.
#'
#' A missing value is *not* rejected where R's own `NA` and the element type
#' already share a bit pattern and the vector travels zero-copy: an
#' `NA_integer_` (or a logical `NA`) at `"i32"` arrives as `INT_MIN`, and a
#' `bit64::integer64` `NA` at `"i64"` arrives as `INT64_MIN`. Both are accepted
#' but change meaning on the device, where they are ordinary numbers, so
#' `check` reports them: a warning by default, an error with `check = "err"`,
#' and nothing with `check = FALSE`. [`as_array()`] reports them again on the
#' way back, at the level of its own `check`. At every other integer element
#' type, and at `"bool"`, a missing value is an error at every level of `check`
#' -- including an `integer64` `NA` at `"ui64"`, where those same bits are the
#' ordinary value `2^63`.
#'
#' At a floating-point element type a missing value is neither rejected nor
#' warned about: it becomes `NaN`, from an `NA_real_` and an `NA_integer_`
#' alike, as [`as.double()`] would give.
#'
#' No other checks are performed when creating the buffer -- a double too large
#' for `"f32"`, for instance, still becomes `Inf`.
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
#' A buffer can also be serialized like any R object, e.g. with [`saveRDS()`]:
#' its data is copied to the host, and unserializing it uploads the data to the
#' default device (the first device of the `PJRT_PLATFORM` platform, `"cpu"`
#' unless set) -- not to the device it was saved from. A buffer saved on a GPU
#' therefore also loads on a machine without one; use [`copy_buffer()`] to move
#' it afterwards.
#' A buffer referenced several times in one serialized object is written once,
#' and loads as one buffer.
#' A loaded buffer is a NULL external pointer until its first use (e.g. by
#' [`as_array()`], [`shape()`] or printing), which fills in its address.
#' This requires serialization format version 3 (R's default); a buffer
#' serialized under version 2 loads without its data.
#'
#' @section Scalars:
#' When calling this function on a vector of length 1, the resulting shape is `1L`.
#' To create a 0-dimensional buffer, use `pjrt_scalar` where the resulting shape is `integer()`.
#'
#' @param data (any)\cr
#'  Data to convert to a `PJRTBuffer`.
#' @param dtype (`NULL` | `character(1)` | [`DataType`][xlamisc::DataType])\cr
#'   The type of the buffer.
#'   Currently supported types are:
#'   - `"bool"`: boolean (`"pred"` and `"i1"` are accepted as aliases)
#'   - `"{s,u}{8,16,32,64}"`: Signed and unsigned integer (for `integer` or
#'     `double` data).
#'   - `"f{32,64}"`: Floating point (for `double` or `integer` data).
#'   - `"bf16"`: bfloat16 (for `double` or `integer` data). Values are rounded
#'     to the nearest bfloat16 value, ties to even; the largest finite value is
#'     about 3.3895e38, and magnitudes at or above the overflow midpoint
#'     (about 3.3962e38) round to `Inf`. [`as_array()`] returns the exactly
#'     representable values as `double`.
#'   - `"f16"`: IEEE 754 binary16 (for `double` or `integer` data). Values are
#'     rounded to the nearest binary16 value, ties to even; the largest finite
#'     value is 65504, and magnitudes at or above the overflow midpoint 65520
#'     round to `Inf`. [`as_array()`] returns the exactly representable values
#'     as `double`.
#'   The default (`NULL`) depends on the method:
#'   - `logical` -> `"bool"`
#'   - `integer` -> `"i32"`
#'   - `double` -> `"f32"`
#'   - `raw` -> must be supplied
#'
#'   A `double` at an integer dtype is truncated toward zero, like
#'   [as.integer()] but without its 32-bit intermediate, so
#'   `pjrt_buffer(2^40, dtype = "i64")` stores `1099511627776` rather than
#'   overflowing. A value the dtype cannot hold is an error rather than a
#'   wrapped or clamped result, and the range is tested after truncation, so
#'   `255.7` still fits `"ui8"`.
#' @param shape (`NULL` | `integer()`)\cr
#'   The dimensions of the buffer.
#'   The default (`NULL`) is to infer them from the data if possible.
#'   The default (`NULL`) depends on the method.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr
#'   A `PJRTDevice` object or the name of the platform to use ("cpu", "cuda", ...), in which
#'   case the first device for that platform is used.
#'   The default is to use the CPU platform, but this can be configured via the `PJRT_PLATFORM`
#'   environment variable.
#' @param check (`NULL` | `character(1)` | `FALSE`)\cr
#'   How to report a missing value the element type keeps as an ordinary
#'   number (an `NA` at `"i32"`, or a `bit64::integer64` `NA` at `"i64"`; see
#'   *Important*): `"warn"` warns and uploads it, `"err"` aborts, and `FALSE`
#'   skips the scan for it altogether. `TRUE` is not accepted -- with two
#'   levels of strictness it does not say which one is meant. The default
#'   (`NULL`) takes the option `pjrt.upload_check`, and `"warn"` when it is not
#'   set. A value the element type cannot hold is rejected whatever this is set
#'   to, and a missing value at a floating-point element type is never
#'   reported, so `check` only governs the missing values that are kept.
#' @param ... (any)\cr
#'   Additional arguments.
#'   The `logical`, `integer`, `numeric`, `integer64` and `raw` methods take:
#'   - `row_major` (`logical(1)`): Whether `data` holds its elements in
#'     row-major order rather than in R's column-major order. Defaults to
#'     `FALSE`, except for `raw` data, which must name it. As R's column-major
#'     data are the row-major data of the reversed shape,
#'     `pjrt_buffer(x, shape = rev(dim(x)), row_major = TRUE)` stores the array
#'     `x` with its axes reversed, without transposing it; see the `row_major`
#'     argument of [`as_array()`][as_array.PJRTBuffer] for the way back.
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
  check = NULL,
  ...
) {
  UseMethod("pjrt_buffer")
}

# The level an upload's `check` asks for, normalized like as_array()'s:
# the argument when given, else the option `pjrt.upload_check`, else "warn".
# The dispatcher reads the option in C++ and calls this only to reject a value
# it does not accept, so the two agree on what is valid.
upload_check_level <- function(check = NULL) {
  if (!is.null(check)) {
    return(check_level(check))
  }
  option <- getOption("pjrt.upload_check")
  if (is.null(option)) {
    return("warn")
  }
  check_level(option, what = "The option {.code pjrt.upload_check}")
}

# Report, at level `check`, an NA in `data` that a buffer of `dtype` keeps as an
# ordinary number. `keeps` is the dtype the source type travels to zero-copy,
# carrying R's own NA bit pattern with it: "i32" for an R integer -- or a
# logical, which reaches it through as.integer() -- where NA_integer_ is
# INT_MIN, and "i64" for a bit64::integer64, where NA_integer64_ is INT64_MIN.
# At every other integer dtype the NA is rejected in C++, whatever `check` is.
# `anyNA()` is safe on an integer64: it falls back to bit64's `is.na()`, so it
# tests for INT64_MIN and not for a double NaN, which some legitimate int64
# values reinterpret to.
check_na_kept <- function(data, dtype, keeps, check, call = rlang::caller_env()) {
  if (identical(dtype, keeps) && !identical(check, "none") && anyNA(data)) {
    report_na_kept(dtype, check, call = call)
  }
  invisible(NULL)
}

# Reports such an NA. The dispatcher calls this too, for an R input it uploads,
# with `input` naming that input. The sentinel is a string:
# -9223372036854775808 has no exact double to format from. The message does not
# say how many NAs there are: the callers stop at the first one, and counting
# them would mean a second full pass over the input on what is otherwise a
# zero-copy upload.
report_na_kept <- function(dtype, check, input = "Input `data`", call = NULL) {
  sentinel <- if (identical(dtype, "i64")) "-9223372036854775808" else "-2147483648"
  message <- c(
    "{input} contains at least one {.val NA}, which {.val {dtype}} stores as {.val {sentinel}}.",
    i = "It materializes as {.val NA} again in R, which {.code as_array()} reports on the way back."
  )
  if (identical(check, "err")) {
    cli_abort(
      c(message, i = "Set {.code check = FALSE} or the option {.code pjrt.upload_check} to accept it."),
      call = call
    )
  }
  cli::cli_warn(c(
    message,
    i = "Set {.code check = \"err\"} to make this an error, or {.code check = FALSE} to silence it; the option {.code pjrt.upload_check} sets the default."
  ))
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
pjrt_scalar <- function(data, dtype = NULL, device = NULL, check = NULL, ...) {
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
  if (dtype %in% c("i1", "pred")) {
    dtype <- "bool"
  }
  dtype
}

# A scalar has no layout, so `pjrt_scalar()` takes no further arguments.
check_no_dots <- function(...) {
  if (...length()) {
    cli_abort("Unused arguments")
  }
}

convert_buffer_args <- function(
  data,
  dtype,
  device,
  shape,
  default,
  recycle = TRUE,
  row_major = FALSE,
  ...
) {
  dtype <- as_dtype_string(dtype %||% default)
  shape <- shape %||% get_dims(data)
  device <- as_pjrt_device(device)
  client <- client_from_device(device)
  assert_flag(row_major)
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
    dims = shape,
    row_major = row_major
  )
}

#' @export
pjrt_buffer.logical <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = NULL,
  row_major = FALSE,
  ...
) {
  check <- upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, shape, "bool", row_major = row_major, ...)
  check_na_kept(data, args$dtype, "i32", check)
  buffer <- do.call(impl_client_buffer_from_logical, args)
  buffer
}

#' @export
pjrt_buffer.integer <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = NULL,
  row_major = FALSE,
  ...
) {
  check <- upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, shape, "i32", row_major = row_major, ...)
  check_na_kept(data, args$dtype, "i32", check)
  buffer <- do.call(impl_client_buffer_from_integer, args)
  buffer
}

#' @export
pjrt_buffer.numeric <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = NULL,
  row_major = FALSE,
  ...
) {
  # A double keeps no NA: it becomes NaN at a float dtype and is rejected at
  # any other, so `check` is only validated.
  upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, shape, "f32", row_major = row_major, ...)
  buffer <- do.call(impl_client_buffer_from_double, args)
  buffer
}

#' @export
pjrt_buffer.integer64 <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = NULL,
  row_major = FALSE,
  ...
) {
  check <- upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, shape, "i64", row_major = row_major, ...)
  if (!args$dtype %in% c("i64", "ui64")) {
    cli_abort(
      "{.cls integer64} input only supports {.val i64} or {.val ui64} dtype, got {.val {args$dtype}}."
    )
  }
  # i64 is the passthrough; at ui64 the same bit pattern is the ordinary value
  # 2^63, so a missing value is rejected there instead (in C++, with the
  # message every other rejected NA gets).
  check_na_kept(args$data, args$dtype, "i64", check)
  impl_client_buffer_from_integer64(
    client = args$client,
    device = args$device,
    data = args$data,
    dims = args$dims,
    dtype = args$dtype,
    row_major = args$row_major
  )
}

#' @export
pjrt_buffer.raw <- function(
  data,
  ...,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  check = NULL,
  row_major
) {
  # Bytes have no missing value, so `check` is only validated.
  upload_check_level(check)
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
  check = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_no_dots(...)
  check <- upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, integer(), "bool")
  check_na_kept(data, args$dtype, "i32", check)
  buffer <- do.call(impl_client_buffer_from_logical, args)
  buffer
}

#' @export
pjrt_scalar.integer <- function(
  data,
  dtype = NULL,
  device = NULL,
  check = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_no_dots(...)
  check <- upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, integer(), "i32")
  check_na_kept(data, args$dtype, "i32", check)
  buffer <- do.call(impl_client_buffer_from_integer, args)
  buffer
}

#' @export
pjrt_scalar.numeric <- function(
  data,
  dtype = NULL,
  device = NULL,
  check = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_no_dots(...)
  upload_check_level(check)
  args <- convert_buffer_args(data, dtype, device, integer(), "f32")
  buffer <- do.call(impl_client_buffer_from_double, args)
  buffer
}

#' @export
pjrt_scalar.integer64 <- function(
  data,
  dtype = NULL,
  device = NULL,
  check = NULL,
  ...
) {
  if (length(data) != 1) {
    cli_abort("data must have length 1")
  }
  check_no_dots(...)
  pjrt_buffer.integer64(data, dtype = dtype, device = device, shape = integer(), check = check)
}

#' @export
pjrt_scalar.raw <- function(
  data,
  ...,
  dtype = NULL,
  device = NULL,
  check = NULL
) {
  upload_check_level(check)
  if (is.null(dtype)) {
    cli_abort("dtype must be provided")
  }
  dtype <- as_dtype_string(dtype)
  check_raw_buffer_size(data, dtype, integer())
  check_no_dots(...)
  args <- convert_buffer_args(data, dtype, device, integer(), "f32", recycle = FALSE)
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
#' With `await = FALSE`, the transfer runs asynchronously: the array is returned
#' immediately with its final type, `dim` and class, and reading its values
#' blocks until the transfer has completed. Use [is_ready()] to check whether it
#' has (non-blocking) and [await()] to block until it has. The values are then
#' checked when they are first read, or by `await()`.
#'
#' `check = "err"` requires `await = TRUE`. With `await = FALSE` the check runs
#' only once the values are first read, and that read can come from any code --
#' `print()`, arithmetic, another package's C code. An error would then surface
#' there, far from the `as_array()` call that caused it, and interrupt code that
#' does not expect reading a vector to fail. A warning does neither, so
#' `check = "warn"` and `check = FALSE` are allowed.
#'
#' @param x ([`PJRTBuffer`][pjrt_buffer])\cr
#'   Buffer to convert.
#' @param check (`character(1)` | `FALSE`)\cr
#'   How to report a materialized value that R's type cannot hold:
#'   `"warn"` (the default) warns and returns it anyway, `"err"` aborts
#'   (only with `await = TRUE`),
#'   and `FALSE` skips the scan altogether. `TRUE` is not accepted — with
#'   two levels of strictness it does not say which one is meant.
#'
#'   The cases scanned for are:
#'   * **`i32` / `i64`**: any `NA` in the result. R's `NA_integer_` shares
#'     the bit pattern `INT_MIN`; `bit64`'s `NA_integer64_` shares
#'     `INT64_MIN`. A legitimate device value at those bit patterns is
#'     indistinguishable from `NA` once materialized in R.
#'   * **`ui64`**: any negative value in the result. `ui64` is stored as
#'     `bit64::integer64` (signed 64-bit), which wraps values `>= 2^63`
#'     to negative — exactly `2^63` becomes `NA_integer64_`, anything
#'     above becomes a non-NA negative integer64.
#'
#'   Each case is a value the R type genuinely cannot hold, so the check
#'   has no false positives: it fires exactly when the returned vector
#'   would misrepresent the buffer.
#'
#'   No-op for float, boolean, and small/unsigned-32 integer dtypes —
#'   `ui32` is stored as `integer64` and has full headroom, so it cannot
#'   produce a wrapped or NA value.
#' @param await (`logical(1)`)\cr
#'   Whether to wait for the transfer to the host (the default). If `FALSE`,
#'   return immediately with an array whose values arrive asynchronously.
#' @param row_major (`logical(1)`)\cr
#'   Whether to return the buffer's data in row-major order, with the shape
#'   reversed: the result is the R array of shape `rev(shape(x))` whose
#'   column-major data are the buffer's row-major data, so for a buffer in
#'   the default layout the transfer needs no transpose. It is the way back from
#'   `pjrt_buffer(x, shape = rev(dim(x)), row_major = TRUE)`, which it returns
#'   as `x`. Default `FALSE`, which returns the R array of shape `shape(x)`.
#'   Either way, a buffer of rank 0 or 1 gives the same result.
#' @param ... Additional arguments (unused).
#' @return An R `array` (or `vector` for shape `integer()`).
#' @seealso [is_ready()], [await()], [pjrt_execute()]
#' @examplesIf plugins_downloaded()
#' buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
#' as_array(buf)
#'
#' # Start the transfer, and only block when the values are read
#' result <- as_array(buf, await = FALSE)
#' is_ready(result)
#' result
#'
#' # The shape reversed, without a transpose
#' as_array(buf, row_major = TRUE)
#' @export
as_array.PJRTBuffer <- function(x, check = "warn", await = TRUE, row_major = FALSE, ...) {
  check <- check_level(check)
  assert_flag(await)
  assert_flag(row_major)
  if (!await) {
    # The check runs on the array's first read, which can come from any code;
    # erroring there would surface far from this call (see ?as_array.PJRTBuffer).
    if (identical(check, "err")) {
      cli_abort(c(
        "{.code check = \"err\"} requires {.code await = TRUE}.",
        i = "Use {.code check = \"warn\"} or {.code check = FALSE} with {.code await = FALSE}."
      ))
    }
    return(as_array_async(x, check, row_major))
  }
  result <- impl_buffer_to_host(x, row_major)
  check_representable(result, as.character(elt_type(x)), check)
  result
}

# Reports, at the level `check` asks for, a value in `result` -- materialized
# from a buffer of dtype `dt` -- that R's type cannot hold. An error names
# `call`.
check_representable <- function(result, dt, check, call = rlang::caller_env()) {
  if (!identical(check, "none")) {
    if (dt %in% c("i32", "i64") && anyNA(result)) {
      report_lossy(
        check,
        call,
        c(
          "Materialized {.cls {dt}} buffer contains a value that R cannot distinguish from {.val NA}.",
          i = "{.val i32} reserves the bit pattern {.val -2147483648} ({.code INT_MIN}); {.val i64} reserves {.val -9223372036854775808} ({.code INT64_MIN})."
        )
      )
    } else if (identical(dt, "ui64") && (anyNA(result) || any(result < 0, na.rm = TRUE))) {
      # ui64 values >= 2^63 wrap when stored as signed int64 — exactly 2^63
      # becomes NA_integer64_ (INT64_MIN); 2^63 + k becomes a non-NA negative
      # int64. Either way, the unsigned magnitude was lost.
      # (ui32 is materialized as integer64 and has full headroom, so it cannot
      # produce a negative value; no check needed.)
      report_lossy(
        check,
        call,
        c(
          "Materialized {.cls ui64} buffer contains a value `>= 2^63` that wrapped through R's signed {.cls integer64}.",
          i = "Exactly {.code 2^63} becomes {.code NA_integer64_}; larger values become negative {.cls integer64}."
        )
      )
    }
  }
}

# `check` is "warn", "err" or FALSE, normalized to "warn" / "err" / "none".
# TRUE is rejected rather than mapped: with two levels of strictness it does
# not say which one is meant. `what` names the value in the error.
check_level <- function(check, what = "{.arg check}") {
  if (isFALSE(check)) {
    return("none")
  }
  if (is.character(check) && length(check) == 1L && check %in% c("warn", "err")) {
    return(check)
  }
  cli_abort(c(
    paste(what, "must be {.val warn}, {.val err} or {.code FALSE}, not {.val {check}}."),
    i = "{.code TRUE} is not accepted; pick {.val warn} or {.val err}."
  ))
}

# Report a value the R type cannot hold, at the level `check` asks for; an
# error names `call`. The closing bullet names the way out of whichever level
# was used. `.envir` is forwarded so that cli interpolates the caller's
# variables and not this frame's -- an unbound `{dt}` would otherwise reach
# `stats::dt()`.
report_lossy <- function(check, call, message, .envir = parent.frame()) {
  if (identical(check, "err")) {
    cli_abort(
      c(message, i = "Set {.code check = FALSE} to skip this check."),
      call = call,
      .envir = .envir
    )
  }
  cli::cli_warn(
    c(
      message,
      i = "Set {.code check = \"err\"} to make this an error, or {.code check = FALSE} to silence it."
    ),
    .envir = .envir
  )
}

# Starts the transfer of `x` to the host and returns its host array: an R array
# whose values materialize when first read, checked at level `check` ("warn" or
# "none"), in row-major order with the shape reversed for `row_major`.
as_array_async <- function(x, check, row_major) {
  impl_buffer_to_host_async(x, check, row_major)
}

# Called by a host array (src/host_array.cpp) on the first read of its data,
# and by await(): waits for its transfer, checks the values and makes the host
# array wrap them. A no-op for any other object.
host_array_materialize <- function(x) {
  pending <- impl_host_array_pending(x)
  if (!is.null(pending)) {
    # No call: the warning surfaces from whatever first read the array.
    check_representable(pending$values, pending$dtype, pending$check, call = NULL)
    impl_host_array_set(x, pending$values)
  }
  invisible(x)
}

#' @export
as_raw.PJRTBuffer <- function(x, row_major, ...) {
  assert_flag(row_major)
  client <- client_from_device(device(x))
  impl_buffer_to_raw(client, x, row_major = row_major)
}

# Called by the buffer's ALTREP carrier (src/buffer_sexp.cpp) to serialize a
# buffer: its host bytes, plus what is needed to upload them again. The device
# is left out, so a buffer loads on any machine. The bytes are row-major, the
# device's own layout, so neither direction transposes them.
buffer_serialized_state <- function(x) {
  list(
    data = as_raw(x, row_major = TRUE),
    dtype = as.character(elt_type(x)),
    shape = shape(x)
  )
}

# Called by the buffer's ALTREP carrier to unserialize a buffer from the state
# buffer_serialized_state() produced. It is uploaded to
# `the[["unserialize_device"]]` -- a device, a device specification, or a
# function returning one -- else to the default device. anvl sets that entry so
# that buffers load onto its own default device.
buffer_unserialize <- function(state) {
  device <- the[["unserialize_device"]]
  if (is.function(device)) {
    device <- device()
  }
  pjrt_buffer(
    state$data,
    dtype = state$dtype,
    shape = state$shape,
    device = device,
    row_major = TRUE
  )
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
#'   Set to `-1` for no limit.
#' @param max_width (`integer(1)`)\cr
#'   The maximum width (in characters) of the printed buffer.
#'   Set to negative values for no limit.
#'   Note that for very small values, the actual printed width might be slightly smaller
#'   as at least one column will be printed.
#'   Also, this limit only affects the printed rows containing the actual data,
#'   other rows might exceed the width.
#' @param max_rows_slice (`integer(1)`)\cr
#'   The maximum number of rows to print for each slice.
#'   Set to `-1` for no limit.
#' @param header (`logical(1)`)\cr
#'   Whether to print the header.
#' @param footer (`NULL` or `character(1)`)\cr
#'   The footer line to print. If `NULL` (default), prints the standard
#'   `[ <PLATFORM><TYPE>{<SHAPE>} ]` summary. Use `""` to suppress.
#' @param reversed (`logical(1)`)\cr
#'   Whether to print the buffer with its shape reversed, as the R array of
#'   shape `rev(shape(x))` whose column-major data are the buffer's row-major
#'   data (see the `row_major` argument of
#'   [`as_array()`][as_array.PJRTBuffer]). That is the R array a buffer
#'   uploaded with `pjrt_buffer(x, shape = rev(dim(x)), row_major = TRUE)`
#'   stores, which then prints as `pjrt_buffer(x)` would. Default `FALSE`.
#' @param ... Additional arguments (unused).
#' @export
print.PJRTBuffer <- function(
  x,
  max_rows = getOption("pjrt.print_max_rows", 30L),
  max_width = getOption("pjrt.print_max_width", 85L),
  max_rows_slice = getOption("pjrt.print_max_rows_slice", max_rows),
  header = TRUE,
  footer = NULL,
  reversed = FALSE,
  ...
) {
  assert_flag(header)
  assert_flag(reversed)
  max_rows <- assert_print_rows(max_rows)
  max_width <- assert_int(max_width, coerce = TRUE)
  if (max_width %in% c(0, 1L)) {
    # we disallow 1, because every data line starts with ' '
    cli_abort("Either provide a negative value for max_width or a value > 1")
  }
  max_rows_slice <- assert_print_rows(max_rows_slice)

  if (header) {
    cat(class(x)[[1L]], "\n")
  }
  shp <- shape(x)
  if (reversed) {
    shp <- rev(shp)
  }
  shape_str <- if (length(shp)) {
    paste0(paste0(shp, collapse = "x"))
  } else {
    ""
  }
  impl_buffer_print(
    x,
    max_rows = max_rows,
    max_width = max_width,
    max_rows_slice = max_rows_slice,
    reversed = reversed
  )
  if (is.null(footer)) {
    footer <- sprintf("[ %s%s{%s} ]", toupper(platform(x)), as.character(dtype(x)), shape_str)
  }
  if (nzchar(footer)) {
    cat(footer, "\n")
  }
  invisible(x)
}

assert_print_rows <- function(x, .var.name = vname(x)) {
  x <- assert_int(x, coerce = TRUE, lower = -1L, .var.name = .var.name)
  if (x == 0L) {
    cli_abort("{.arg {(.var.name)}} must be -1 (no limit) or positive.")
  }
  x
}

#' @export
dtype.PJRTBuffer <- function(x, ...) {
  as_dtype(as.character(elt_type(x)))
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

# Helper function to check scalar roundtrip
test_pjrt_scalar <- function(
  data,
  dtype = NULL,
  tolerance = testthat::testthat_tolerance()
) {
  stopifnot(is.atomic(data) && length(data) == 1)
  args <- list(data = data)
  args$dtype <- dtype
  buffer <- do.call(pjrt_scalar, args)

  expect_class(buffer, "PJRTBuffer")
  result <- as_array(buffer)
  testthat::expect_true(!is.array(result))

  # Check that scalar becomes 1D array with length 1
  testthat::expect_true(is.null(dim(result)))

  modify <- function(data) {
    data[1L] <- if (is.numeric(data)) {
      1L
    } else if (is.logical(data)) {
      !data[1L]
    } else {
      cli_abort("Unsupported data type: ", typeof(data))
    }
  }

  # i64/ui64 buffers materialize as bit64::integer64; coerce the expected value
  # so the equality check sees matching types.
  coerce_expected <- function(x) {
    if (identical(dtype, "i64") || identical(dtype, "ui64")) {
      bit64::as.integer64(x)
    } else {
      x
    }
  }

  # Test that modifying the original data doesn't change the buffer
  original_data <- data
  data[1L] <- modify(data)

  result_after_modification <- as_array(buffer)
  testthat::expect_equal(
    result_after_modification,
    coerce_expected(original_data),
    tolerance = tolerance
  )

  # Test that modifying the result doesn't persist when recreating from buffer
  result[1L] <- modify(result)

  result_recreated <- as_array(buffer)
  testthat::expect_equal(
    result_recreated,
    coerce_expected(original_data),
    tolerance = tolerance
  )

  TRUE
}

# Helper function to check buffer roundtrip
test_pjrt_buffer <- function(
  data,
  dtype = NULL,
  tolerance = testthat::testthat_tolerance()
) {
  args <- list(data = data)
  args$dtype <- dtype
  buffer <- do.call(pjrt_buffer, args)

  expect_class(buffer, "PJRTBuffer")
  result <- as_array(buffer)
  testthat::expect_true(is.array(result))

  data_arr <- as.array(data)

  testthat::expect_equal(result, data_arr, tolerance = tolerance)

  modify_first <- function(data) {
    data[1L] +
      if (is.numeric(data)) {
        1L
      } else if (is.logical(data)) {
        !data[1L]
      } else {
        cli_abort("Unsupported data type: ", typeof(data))
      }
  }

  # Check dimensions are preserved
  if (is.null(dim(data))) {
    # Vector should become 1D array
    testthat::expect_equal(dim(result), length(data))
  } else {
    # Array should preserve dimensions
    testthat::expect_equal(dim(result), dim(data))
  }

  # Test that modifying the original data doesn't change the buffer
  data[1L] <- modify_first(data)

  result_after_modification <- as_array(buffer)
  testthat::expect_equal(
    result_after_modification,
    data_arr,
    tolerance = tolerance
  )

  # Test that modifying the result doesn't persist when recreating from buffer
  result[1L] <- modify_first(result)

  result_recreated <- as_array(buffer)
  if (!is.null(tolerance)) {
    testthat::expect_equal(result_recreated, data_arr, tolerance = tolerance)
  } else {
    testthat::expect_equal(result_recreated, data_arr)
  }

  return(buffer)
}

test_that("dtype works for PJRTBuffer", {
  buf_f32 <- pjrt_buffer(1.0, dtype = "f32")
  expect_equal(dtype(buf_f32), tengen::FloatType(32L))

  buf_f64 <- pjrt_buffer(1.0, dtype = "f64")
  expect_equal(dtype(buf_f64), tengen::FloatType(64L))

  buf_i32 <- pjrt_buffer(1L, dtype = "i32")
  expect_equal(dtype(buf_i32), tengen::IntegerType(32L))

  buf_i8 <- pjrt_buffer(1L, dtype = "i8")
  expect_equal(dtype(buf_i8), tengen::IntegerType(8L))

  buf_pred <- pjrt_buffer(TRUE, dtype = "pred")
  expect_equal(dtype(buf_pred), tengen::BooleanType())

  buf_bool <- pjrt_buffer(TRUE, dtype = "bool")
  expect_equal(dtype(buf_bool), tengen::BooleanType())

  buf_ui8 <- pjrt_buffer(1L, dtype = "ui8")
  expect_equal(dtype(buf_ui8), tengen::UIntegerType(8L))
})

test_that("pjrt_scalar roundtrip works for scalar data", {
  test_pjrt_scalar(TRUE)
  test_pjrt_scalar(FALSE)

  # Test integer scalar
  test_pjrt_scalar(42L)
  test_pjrt_scalar(-42L)
  test_pjrt_scalar(0L)
  test_pjrt_scalar(-1L, "i8")
  test_pjrt_scalar(2L, "i16")
  test_pjrt_scalar(100L, "i32")
  test_pjrt_scalar(-3L, "i64")
  test_pjrt_scalar(1L, "ui8")
  test_pjrt_scalar(2L, "ui16")
  test_pjrt_scalar(3L, "ui64")

  # Test double scalar
  test_pjrt_scalar(3.14, "f64")
  test_pjrt_scalar(-3, "f32")
})

test_that("pjrt_buffer roundtrip works for logical data", {
  # Test logical vector
  logical_vec <- array(c(TRUE, FALSE, TRUE, FALSE))
  test_pjrt_buffer(logical_vec)

  # Test logical matrix
  logical_matrix <- matrix(
    c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    nrow = 2,
    ncol = 3
  )
  test_pjrt_buffer(logical_matrix)
})

test_that("pjrt_buffer roundtrip works for double data with different types", {
  # Test single precision (32-bit)
  data_32 <- c(1.0, -1.0, 0.0, 3.14159, -2.71828)
  test_pjrt_buffer(data_32, "f32", tolerance = 1e-6)

  # Test double precision (64-bit)
  data_64 <- c(1.0, -1.0, 0.0, 3.14159265359, -2.71828182846)
  test_pjrt_buffer(data_64, "f64", tolerance = 1e-12)

  # Test arrays with dimensions
  data_matrix <- matrix(c(1.1, 2.2, 3.3, 4.4), nrow = 2, ncol = 2)
  test_pjrt_buffer(data_matrix, "f32", tolerance = 1e-6)
})

test_that("pjrt_buffer roundtrip works for f16 data", {
  # exactly representable in binary16, so no tolerance is needed
  test_pjrt_buffer(c(1.0, -1.0, 0.0, 0.5, 0.25), "f16")
  test_pjrt_buffer(matrix(c(1.5, 2.5, -3.5, 4.75), nrow = 2), "f16")
  test_pjrt_scalar(0.5, "f16")
})

test_that("f16 buffers materialize the exactly representable doubles", {
  # binary16 targets computed independently: 1/3 -> 0x3555 = 0.333251953125,
  # 0.1 -> 0x2E66 = 0.0999755859375, 65504 = largest finite value, 65520 is
  # the overflow midpoint and rounds (ties to even) to Inf, 2^-24 = smallest
  # subnormal
  input <- c(1, 0.5, 1 / 3, 0.1, 65504, 65520, 2^-24, -2.5, 0)
  expected <- c(
    1,
    0.5,
    0.333251953125,
    0.0999755859375,
    65504,
    Inf,
    2^-24,
    -2.5,
    0
  )
  buf <- pjrt_buffer(input, dtype = "f16")
  expect_equal(as.character(elt_type(buf)), "f16")
  expect_identical(as_array(buf), array(expected))

  # integer data is accepted like for the other float dtypes
  buf_int <- pjrt_buffer(matrix(1:6, nrow = 2), dtype = "f16")
  expect_equal(as.character(elt_type(buf_int)), "f16")
  expect_identical(as_array(buf_int), matrix(as.double(1:6), nrow = 2))

  s <- pjrt_scalar(3.14159, dtype = "f16")
  expect_identical(shape(s), integer())
  expect_identical(as_array(s), 3.140625)
})

test_that("doubles round to f16 to nearest, ties to even, not via float", {
  # 65519.999: rounding the double directly to binary16 gives 65504, but
  # rounding through float first lands exactly on 65520 (a float), which then
  # ties up to Inf -- the classic double-rounding failure this test pins.
  expect_identical(as_array(pjrt_buffer(65519.999, dtype = "f16")), array(65504))

  # ties to even at the subnormal boundary: 2^-25 is midway between 0 and
  # 2^-24 and rounds to the even neighbour 0; 1.5 * 2^-24 is midway between
  # 2^-24 and 2^-23 and rounds to 2^-23.
  buf <- pjrt_buffer(c(2^-25, 1.5 * 2^-24), dtype = "f16")
  expect_identical(as_array(buf), array(c(0, 2^-23)))
})

test_that("f16 raw and empty buffers work", {
  empty <- pjrt_empty(dtype = "f16", shape = c(0, 2))
  expect_identical(shape(empty), c(0L, 2L))
  expect_equal(as.character(elt_type(empty)), "f16")

  # 0x3C00 = 1.0 and 0xB800 = -0.5, as little-endian byte pairs
  raw_bytes <- as.raw(c(0x00, 0x3c, 0x00, 0xb8))
  buf <- pjrt_buffer(raw_bytes, dtype = "f16", shape = 2L, row_major = TRUE)
  expect_identical(as_array(buf), array(c(1, -0.5)))
  expect_identical(as_raw(buf, row_major = TRUE), raw_bytes)

  expect_error(
    pjrt_buffer(as.raw(c(0, 0, 0)), dtype = "f16", shape = 2L, row_major = TRUE),
    "requires 4 bytes"
  )
})

test_that("f16 rejects logical input like the other float dtypes", {
  expect_error(pjrt_buffer(TRUE, dtype = "f16"), "Unsupported type")
})

test_that("dtype() on an f16 buffer errors until tengen can express it", {
  # elt_type() reports the buffer-level dtype; dtype() returns a
  # tengen::DataType, which has no f16 representation yet.
  buf <- pjrt_buffer(1.5, dtype = "f16")
  expect_equal(as.character(elt_type(buf)), "f16")
  expect_error(dtype(buf), "Unsupported dtype")
})

test_that("pjrt_buffer handles edge cases", {
  # Test empty vectors
  expect_error(pjrt_buffer(logical(0), shape = c(1, 4)), "but specified shape is")
  expect_error(pjrt_buffer(integer(0), shape = c(1, 4)), "but specified shape is")
  expect_error(pjrt_buffer(numeric(0), shape = c(1, 4)), "but specified shape is")
})

test_that("pjrt_buffer check = FALSE silently transfers NA", {
  # default behaviour: NAs flow through and become dtype-specific bit patterns.
  expect_no_error(pjrt_buffer(c(1, NA, 3)))
  expect_no_error(pjrt_buffer(c(1L, NA_integer_, 3L)))
  expect_no_error(pjrt_buffer(c(TRUE, NA, FALSE)))
  expect_no_error(pjrt_scalar(NA_integer_))
})

test_that("pjrt_buffer check = TRUE errors on NA input", {
  expect_error(
    pjrt_buffer(c(1, NA, 3), check = TRUE),
    "no representation at the XLA level"
  )
  expect_error(
    pjrt_buffer(c(1L, NA_integer_, 3L), check = TRUE),
    "no representation at the XLA level"
  )
  expect_error(
    pjrt_buffer(c(TRUE, NA, FALSE), check = TRUE),
    "no representation at the XLA level"
  )
  expect_error(
    pjrt_scalar(NA_integer_, check = TRUE),
    "no representation at the XLA level"
  )
  expect_error(
    pjrt_scalar(NA_real_, check = TRUE),
    "no representation at the XLA level"
  )
  expect_error(
    pjrt_scalar(NA, check = TRUE),
    "no representation at the XLA level"
  )

  # Clean inputs pass through unaffected.
  expect_no_error(pjrt_buffer(c(1, 2, 3), check = TRUE))
  expect_no_error(pjrt_buffer(c(1L, 2L, 3L), check = TRUE))
  expect_no_error(pjrt_buffer(c(TRUE, FALSE), check = TRUE))
})

test_that("as_array check = TRUE catches i32 / i64 NA collisions", {
  client <- pjrt_client("cpu")

  # i32: NA_integer_ bit pattern is INT_MIN (-2147483648).
  buf_i32 <- pjrt_buffer(NA_integer_, dtype = "i32")
  expect_true(anyNA(as_array(buf_i32)))
  expect_error(as_array(buf_i32, check = TRUE), "distinguish from")

  # i64: planting INT64_MIN.
  bytes_i64 <- as.raw(c(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80))
  buf_i64 <- impl_client_buffer_from_raw(client, devices(client)[[1L]], bytes_i64, 1L, "i64")
  expect_true(anyNA(as_array(buf_i64)))
  expect_error(as_array(buf_i64, check = TRUE), "distinguish from")

  # Clean buffers — no error.
  expect_no_error(as_array(pjrt_buffer(1:3, dtype = "i32"), check = TRUE))
  expect_no_error(as_array(pjrt_buffer(1L, dtype = "i64"), check = TRUE))
})

test_that("as_array check = TRUE catches ui64 wrap (>= 2^63)", {
  client <- pjrt_client("cpu")
  # 2^63 wraps to INT64_MIN (negative integer64).
  bytes <- as.raw(c(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80))
  buf <- impl_client_buffer_from_raw(client, devices(client)[[1L]], bytes, 1L, "ui64")
  expect_error(as_array(buf, check = TRUE), "wrapped through")
})

test_that("as_array preserves the full ui32 range losslessly (no wrap)", {
  client <- pjrt_client("cpu")
  # Bit pattern 0x80000000 used to wrap to INT_MIN / NA_integer_ via signed
  # int32. ui32 now materializes as integer64 (53 bits of headroom over u32).
  bytes <- as.raw(c(0x00, 0x00, 0x00, 0x80))
  buf <- impl_client_buffer_from_raw(client, devices(client)[[1L]], bytes, 1L, "ui32")
  result <- as_array(buf, check = TRUE)
  expect_true(bit64::is.integer64(result))
  expect_equal(as.character(result), "2147483648")
})

test_that("as_array check is a no-op for float / bool / small-integer dtypes", {
  buf_f32 <- pjrt_buffer(c(1, NaN, 3), dtype = "f32")
  expect_no_error(as_array(buf_f32, check = TRUE))

  buf_pred <- pjrt_buffer(c(TRUE, FALSE), dtype = "pred")
  expect_no_error(as_array(buf_pred, check = TRUE))

  buf_i8 <- pjrt_buffer(c(-128L, 0L, 127L), dtype = "i8")
  expect_no_error(as_array(buf_i8, check = TRUE))
})

test_that("pjrt_buffer preserves 3d dimensions", {
  # Test 3D array
  arr_3d <- array(1:24, dim = c(2, 3, 4))
  test_pjrt_buffer(arr_3d)
})

test_that("pjrt_buffer dispatches on integer64 to i64", {
  x <- bit64::as.integer64(c(1, 2^32, -2^40, 9223372036854775000))
  buf <- pjrt_buffer(x)
  expect_equal(as.character(elt_type(buf)), "i64")
  expect_equal(shape(buf), 4L)
})

test_that("pjrt_buffer / as_array round-trip i64 with full 64-bit range", {
  x <- bit64::as.integer64(c(1, 2^32, -2^40, 9223372036854775000))
  buf <- pjrt_buffer(x)
  back <- as_array(buf)
  expect_s3_class(back, "integer64")
  expect_equal(as.character(back), as.character(x))
})

test_that("pjrt_scalar.integer64 round-trips a single 64-bit value", {
  x <- bit64::as.integer64(9223372036854775000)
  buf <- pjrt_scalar(x)
  expect_equal(shape(buf), integer())
  expect_equal(as.character(elt_type(buf)), "i64")

  back <- as_array(buf)
  expect_s3_class(back, "integer64")
  expect_equal(as.character(back), as.character(x))

  expect_error(pjrt_scalar(bit64::as.integer64(c(1, 2))), "length 1")
})

test_that("pjrt_buffer.integer64 rejects non-i64/ui64 dtype", {
  expect_error(
    pjrt_buffer(bit64::as.integer64(1), dtype = "i32"),
    "only supports.*i64"
  )
})

test_that("ui64 buffers also materialize as integer64", {
  # bit64::integer64 is signed; ui64 -> integer64 preserves bit pattern but
  # values >= 2^63 will appear as negative integer64.
  buf <- pjrt_buffer(c(0L, 1L, 100L), dtype = "ui64")
  back <- as_array(buf)
  expect_s3_class(back, "integer64")
  expect_equal(as.character(back), c("0", "1", "100"))
})

test_that("pjrt_buffer / as_array round-trip ui64 with full 64-bit range", {
  x <- bit64::as.integer64(c(0, 1, 2^32, -2^40, 9223372036854775000))
  dim(x) <- 5L
  buf <- pjrt_buffer(x, dtype = "ui64")
  expect_equal(as.character(elt_type(buf)), "ui64")
  expect_equal(as_array(buf), x)
})

test_that("raw", {
  sample_signed <- function(precision, shape) {
    precision <- as.integer(precision)
    lower <- as.integer(-2^(precision - 1))
    upper <- as.integer(2^(precision - 1) - 1)
    array(
      sample(seq(lower, upper), prod(shape), replace = TRUE),
      dim = shape
    )
  }
  sample_unsigned <- function(precision, shape) {
    array(
      sample(seq(0, 2^(precision) - 1), prod(shape), replace = TRUE),
      dim = shape
    )
  }
  # Because powers are floating point operations we sample from a bit below the
  # available range to ensure that we stay within as we otherwise get overflows
  test_cases <- list(
    f32 = list(data = matrix(runif(4), nrow = 2), dtype = "f32"),
    f64 = list(data = matrix(runif(6), nrow = 3), dtype = "f64"),

    i8 = list(data = sample_signed(6, c(3, 2)), dtype = "i8"),
    i16 = list(data = sample_signed(14, c(4, 3)), dtype = "i16"),
    i32 = list(data = sample_signed(30, c(4, 3, 1)), dtype = "i32"),
    i64 = list(data = sample_signed(30, c(2, 3, 7)), dtype = "i64"),

    ui8 = list(data = sample_unsigned(6, c(1, 1)), dtype = "ui8"),
    ui16 = list(data = sample_unsigned(14, c(2, 1)), dtype = "ui16"),
    ui32 = list(data = sample_unsigned(30, c(2, 2, 2)), dtype = "ui32"),
    ui64 = list(data = sample_unsigned(30, c(2, 1, 2, 1)), dtype = "ui64"),

    pred = list(
      data = matrix(c(TRUE, FALSE, TRUE, FALSE), nrow = 2),
      dtype = "pred"
    )
  )

  for (test_name in names(test_cases)) {
    test_case <- test_cases[[test_name]]
    original_data <- test_case$data
    dtype <- test_case$dtype

    # Full roundtrip: R → PJRT buffer → raw → PJRT buffer → R
    buf1 <- pjrt_buffer(original_data, dtype = dtype)
    raw_data <- as_raw(buf1, row_major = FALSE)
    buf2 <- pjrt_buffer(
      raw_data,
      dtype = dtype,
      shape = dim(original_data),
      row_major = FALSE
    )
    roundtrip_data <- as_array(buf2)

    # Compare original with roundtrip
    if (dtype %in% c("f32", "f64")) {
      expect_equal(roundtrip_data, original_data, tolerance = 1e-6)
    } else if (dtype %in% c("i64", "ui64")) {
      # 64-bit integer buffers materialize as bit64::integer64 — array() drops
      # the class, so compare the unboxed bit pattern via as.character().
      expect_equal(
        as.character(roundtrip_data),
        as.character(bit64::as.integer64(as.vector(original_data)))
      )
      expect_equal(dim(roundtrip_data), dim(original_data))
    } else {
      expect_equal(roundtrip_data, original_data)
    }
  }
})

test_that("roundtrip tests for scalars", {
  test_scalars <- list(
    f32 = 3.14,
    f64 = 3.14159265359,
    i32 = 42L,
    ui8 = 255L,
    pred = TRUE
  )

  for (dtype in names(test_scalars)) {
    original <- test_scalars[[dtype]]

    buf1 <- pjrt_scalar(original, dtype = dtype)
    raw_data <- as_raw(buf1, row_major = FALSE)
    buf2 <- pjrt_scalar(raw_data, dtype = dtype)
    roundtrip <- as_array(buf2)

    if (dtype %in% c("f32", "f64")) {
      expect_equal(
        roundtrip,
        original,
        tolerance = 1e-6,
        info = paste("Scalar roundtrip failed for type:", type)
      )
    } else {
      expect_equal(
        roundtrip,
        original,
        info = paste("Scalar roundtrip failed for type:", type)
      )
    }
  }
})

test_that("dtype returns correct data types", {
  # Test logical buffer
  logical_data <- c(TRUE, FALSE, TRUE)
  buffer <- pjrt_buffer(logical_data)
  x <- elt_type(buffer)
  expect_true(is_elt_type(x))
  expect_equal(as.character(x), "pred")

  # Test integer buffer (signed 32-bit)
  integer_data <- c(1L, 2L, 3L)
  buffer <- pjrt_buffer(integer_data, "i32")
  x <- elt_type(buffer)
  expect_true(is_elt_type(x))
  expect_equal(as.character(x), "i32")

  # Test unsigned integer buffer (8-bit)
  buffer <- pjrt_buffer(integer_data, "ui8")
  x <- elt_type(buffer)
  expect_true(is_elt_type(x))
  expect_equal(as.character(x), "ui8")

  # Test double buffer (32-bit)
  double_data <- c(1.1, 2.2, 3.3)
  buffer <- pjrt_buffer(double_data, "f32")
  x <- elt_type(buffer)
  expect_true(is_elt_type(x))
  expect_equal(as.character(x), "f32")

  # Test double buffer (64-bit)
  buffer <- pjrt_buffer(double_data, "f64")
  x <- elt_type(buffer)
  expect_true(is_elt_type(x))
  expect_equal(as.character(x), "f64")

  # Test scalar buffer
  scalar_data <- 42L
  buffer <- pjrt_scalar(scalar_data)
  x <- elt_type(buffer)
  expect_true(is_elt_type(x))
  expect_equal(as.character(x), "i32")
})

test_that("R layout and PJRT layout (2D)", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  path <- system.file("programs/jax-stablehlo-subset-2d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  x_buf <- pjrt_buffer(x)
  check <- function(i1, i2) {
    i1_buf <- pjrt_scalar(i1, "i32")
    i2_buf <- pjrt_scalar(i2, "i32")

    result <- as_array(pjrt_execute(executable, x_buf, i1_buf, i2_buf))
    expect_equal(x[i1 + 1, i2 + 1], result)
  }
  check(0L, 0L)
  check(0L, 1L)
  check(1L, 0L)
  check(1L, 1L)
})

test_that("R layout and PJRT layout (3D)", {
  skip_if_metal("-:26:28: error: expected ')' in inline location")
  path <- system.file("programs/jax-stablehlo-subset-3d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  x <- array(as.double(1:24), dim = c(2, 3, 4))
  x_buf <- pjrt_buffer(x)

  check <- function(i1, i2, i3) {
    i1_buf <- pjrt_scalar(i1, "i32")
    i2_buf <- pjrt_scalar(i2, "i32")
    i3_buf <- pjrt_scalar(i3, "i32")

    result <- as_array(
      pjrt_execute(
        executable,
        x_buf,
        i1_buf,
        i2_buf,
        i3_buf
      )
    )
    expect_equal(x[i1 + 1, i2 + 1, i3 + 1], result)
  }
  for (i1 in 0:1) {
    for (i2 in 0:2) {
      for (i3 in 0:3) {
        check(i1, i2, i3)
      }
    }
  }

  # slicing also works (internal optimization w.r.t. transposition)
  path <- system.file(
    "programs/jax-stablehlo-slice-column-keep.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  x <- array(as.double(1:12), dim = c(3, 4))
  x_buf <- pjrt_buffer(x)
  i1 <- 1L
  i1_buf <- pjrt_scalar(i1, "i32")
  result <- as_array(pjrt_execute(executable, x_buf, i1_buf))
  expect_equal(x[, i1 + 1, drop = FALSE], result)

  path <- system.file(
    "programs/jax-stablehlo-slice-column-drop.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- as_array(pjrt_execute(executable, x_buf, i1_buf))
  expect_equal(array(x[, i1 + 1], dim = 3L), result)
})

test_that("buffer <-> raw: row_major parameter", {
  types <- list(
    list(pjrt_type = "ui8", rtype = "integer"),
    list(pjrt_type = "ui16", rtype = "integer"),
    list(pjrt_type = "ui32", rtype = "integer"),
    list(pjrt_type = "ui64", rtype = "integer"),
    list(pjrt_type = "i8", rtype = "integer"),
    list(pjrt_type = "i16", rtype = "integer"),
    list(pjrt_type = "i32", rtype = "integer"),
    list(pjrt_type = "i64", rtype = "integer"),
    list(pjrt_type = "f32", rtype = "double"),
    list(pjrt_type = "f64", rtype = "double"),
    list(pjrt_type = "i32", rtype = "integer"),
    list(pjrt_type = "i64", rtype = "integer"),
    list(pjrt_type = "pred", rtype = "logical")
  )

  check <- function(pjrt_type, rtype) {
    data_raw <- as.raw(1:6)

    ## pjrt_buffer()

    # buf1: [1, 2, 3, 4, 5, 6] that represents [[1, 2], [3, 4], [5, 6]]
    buf1 <- pjrt_buffer(
      data_raw,
      dtype = "ui8",
      row_major = TRUE,
      shape = c(3, 2)
    )
    # buf2: [1, 2, 3, 4, 5, 6] that represents [[1, 3, 5], [2, 4, 6]]
    buf2 <- pjrt_buffer(
      data_raw,
      dtype = "ui8",
      row_major = FALSE,
      shape = c(2, 3)
    )

    # we send both back as they are, i.e. using the correct metadata
    raw1 <- array(as_raw(buf1, row_major = FALSE), dim = c(3, 2))
    raw2 <- array(as_raw(buf2, row_major = FALSE), dim = c(2, 3))
    expect_equal(raw1, t(raw2))

    ## as_raw()

    # buf: [1, 2, 3, 4, 5, 6] that represents [[1, 3, 5], [2, 4, 6]]
    buf <- pjrt_buffer(
      data_raw,
      dtype = "ui8",
      row_major = FALSE,
      shape = c(2, 3)
    )
    # this will return it as-is
    raw1 <- array(as_raw(buf, row_major = FALSE), dim = c(2, 3))
    # this will transpose it
    raw2 <- array(as_raw(buf, row_major = TRUE), dim = c(3, 2))
    expect_equal(raw1, t(raw2))
  }

  for (type in types) {
    check(type$pjrt_type, type$rtype)
  }
})

test_that("device works", {
  buf <- pjrt_buffer(1)
  expect_class(device(buf), "PJRTDevice")
  skip_if(is_metal() || is_cuda())
  expect_snapshot(as.character(device(buf)))
})

test_that("platform for PJRTBuffer", {
  buf_cpu <- pjrt_buffer(1, device = "cpu")
  expect_equal(platform(buf_cpu), "cpu")
  skip_if(!is_cuda())
  buf_cuda <- pjrt_buffer(1, device = "cuda")
  expect_equal(platform(buf_cuda), "cuda")
})

test_that("tests can compare buffers", {
  expect_equal(pjrt_buffer(1), pjrt_buffer(1))
  expect_failure(expect_equal(pjrt_buffer(1), pjrt_buffer(2)))
  expect_failure(expect_equal(pjrt_buffer(1), pjrt_scalar(1)))
})

test_that("No dim with pjrt_buffer", {
  skip_if(is_cuda() || is_metal())
  expect_equal(shape(pjrt_buffer(1)), 1L)
})

test_that("device print", {
  skip_if(is_cuda() || is_metal())
  expect_snapshot(print(device(pjrt_buffer(1))))
})

test_that("dim is integer", {
  expect_true(is.integer(shape(pjrt_buffer(1))))
})

test_that("can move back buffer without specifying client", {
  skip_if(!(is_metal() || is_cuda()))
  device_name <- if (is_metal()) "metal" else "cuda"
  x <- pjrt_buffer(1, device = device_name)
  expect_equal(as_array(x), array(1))
  y <- pjrt_scalar(1, device = device_name)
  expect_equal(as_array(y), 1)
})

test_that("can create f32 and f64 buffers from integer data", {
  expect_equal(
    pjrt_buffer(c(1, 2, 3, 4), "f32"),
    pjrt_buffer(1:4, "f32")
  )

  expect_equal(
    pjrt_buffer(c(1, 2, 3, 4), "f64", shape = c(2, 2)),
    pjrt_buffer(1:4, "f64", shape = c(2, 2))
  )
})

test_that("can specify dims", {
  expect_equal(shape(pjrt_buffer(1:4, shape = c(2, 2))), c(2, 2))
})

test_that("prevent dubious recycling behavior", {
  expect_error(pjrt_buffer(1:2, shape = c(1, 4)), "but specified shape is")
  expect_error(pjrt_buffer(c(1, 2), shape = c(1, 4)), "but specified shape is")
  expect_error(pjrt_buffer(c(TRUE, FALSE), shape = c(1, 4)), "but specified shape is")

  # but 1 element works:
  expect_equal(
    pjrt_buffer(1, shape = c(1, 4)),
    pjrt_buffer(rep(1, 4), shape = c(1, 4))
  )
  expect_equal(
    pjrt_buffer(1L, shape = c(1, 4)),
    pjrt_buffer(rep(1L, 4), shape = c(1, 4))
  )
  expect_equal(
    pjrt_buffer(TRUE, shape = c(1, 4)),
    pjrt_buffer(rep(TRUE, 4), shape = c(1, 4))
  )

  x <- array(1:4, dim = c(1, 4))
  expect_error(pjrt_buffer(x, shape = c(1, 8)), "but specified shape is")
})

test_that("can compare dtypes", {
  expect_true(elt_type(pjrt_buffer(1, "f32")) == elt_type(pjrt_buffer(1, "f32")))
  expect_false(elt_type(pjrt_buffer(1, "f32")) == elt_type(pjrt_buffer(1, "f64")))
  # can compare to character
  expect_true(elt_type(pjrt_buffer(1, "f32")) == "f32")
  expect_true("f32" == elt_type(pjrt_buffer(1, "f32")))
  expect_false(elt_type(pjrt_buffer(1, "f32")) == "f64")
  expect_false("f64" == elt_type(pjrt_buffer(1, "f32")))
})

test_that("can change shape of array when creating buffer", {
  x <- array(1L, dim = 1L)
  expect_equal(shape(pjrt_buffer(x, shape = c(1, 1))), c(1, 1))
})

test_that("can create float from int", {
  expect_equal(
    pjrt_buffer(1:4, "f32"),
    pjrt_buffer(as.double(1:4), "f32")
  )
  expect_equal(
    pjrt_buffer(1:4, "f64"),
    pjrt_buffer(as.double(1:4), "f64")
  )
})

test_that("create 0-dim array from integer", {
  expect_equal(
    as_array(pjrt_buffer(integer(), "f32", shape = c(0, 1, 2))),
    array(integer(), dim = c(0, 1, 2))
  )
})

test_that("pjrt_empty allocates an uninitialized buffer of the requested shape", {
  e <- pjrt_empty("f32", c(1, 2, 3))
  expect_s3_class(e, "PJRTBuffer")
  expect_equal(shape(e), c(1, 2, 3))
  expect_true(elt_type(e) == "f32")
})

test_that("identity of buffer", {
  skip_if(is_metal() | is_cuda())
  x <- pjrt_buffer(1, device = "cpu")
  expect_equal(pjrt_buffer(x), x)
  expect_error(pjrt_buffer(x, dtype = "i32"), "Must use the same data type as the data")
  expect_error(pjrt_buffer(x, shape = c(1, 2)), "Must use the same shape as the data")
  skip_if(!is_cuda())
  expect_error(pjrt_buffer(x, device = as_pjrt_device("cuda")), "Must use the same device as the data")

  x <- pjrt_scalar(1, device = "cpu")
  expect_equal(pjrt_scalar(x), x)
  expect_error(pjrt_scalar(x, dtype = "i32"), "Must use the same data type as the data")
  skip_if(!is_cuda())
  expect_error(pjrt_scalar(x, device = as_pjrt_device("cuda")), "Must use the same device as the data")
})

test_that("recycle scalar to any length", {
  x <- pjrt_buffer(1, shape = c(1, 2))
  expect_equal(shape(x), c(1, 2))
})

test_that("can create dtype 'pred' from double", {
  expect_equal(pjrt_buffer(1, dtype = "pred"), pjrt_buffer(TRUE))
  expect_equal(pjrt_buffer(c(0, 1, 2), dtype = "pred"), pjrt_buffer(c(FALSE, TRUE, TRUE)))
  expect_equal(pjrt_buffer(c(0, 1, -2), dtype = "pred"), pjrt_buffer(c(FALSE, TRUE, TRUE)))
})

test_that("pjrt_buffer identity when working on a different client", {
  skip_if(!(is_metal() || is_cuda()))
  x <- pjrt_buffer(1, device = "cpu")
  device <- if (is_metal()) "metal" else "cuda"
  expect_equal(x, pjrt_buffer(x, device = NULL))
  x <- pjrt_scalar(1, device = "cpu")
  expect_equal(x, pjrt_scalar(x, device = NULL))
})

test_that("Can create 'i32' from double", {
  expect_equal(pjrt_buffer(1:4, dtype = "f32"), pjrt_buffer(as.double(1:4), dtype = "f32"))
})

test_that("i1 is alias for pred", {
  expect_equal(pjrt_buffer(1, "i1"), pjrt_buffer(1, "pred"))
  expect_equal(pjrt_scalar(1, "i1"), pjrt_scalar(1, "pred"))
  expect_equal(pjrt_empty(shape = c(1, 0), "i1"), pjrt_empty(shape = c(1, 0), "pred"))
})

test_that("pjrt_buffer accepts DataType objects", {
  # pjrt_buffer with DataType
  expect_equal(
    pjrt_buffer(c(1, 2, 3), dtype = tengen::FloatType(32)),
    pjrt_buffer(c(1, 2, 3), dtype = "f32")
  )
  expect_equal(
    pjrt_buffer(c(1, 2, 3), dtype = tengen::FloatType(64)),
    pjrt_buffer(c(1, 2, 3), dtype = "f64")
  )
  expect_equal(
    pjrt_buffer(1L, dtype = tengen::IntegerType(32)),
    pjrt_buffer(1L, dtype = "i32")
  )
  expect_equal(
    pjrt_buffer(1L, dtype = tengen::UIntegerType(8)),
    pjrt_buffer(1L, dtype = "ui8")
  )
  expect_equal(
    pjrt_buffer(TRUE, dtype = tengen::BooleanType()),
    pjrt_buffer(TRUE, dtype = "pred")
  )

  # pjrt_scalar with DataType
  expect_equal(
    pjrt_scalar(42L, dtype = tengen::IntegerType(32)),
    pjrt_scalar(42L, dtype = "i32")
  )
  expect_equal(
    pjrt_scalar(3.14, dtype = tengen::FloatType(64)),
    pjrt_scalar(3.14, dtype = "f64")
  )

  # pjrt_empty with DataType
  expect_equal(
    pjrt_empty(dtype = tengen::FloatType(32), shape = c(0, 3)),
    pjrt_empty(dtype = "f32", shape = c(0, 3))
  )

  # raw buffer with DataType
  raw_data <- as.raw(rep(0, 24))
  expect_equal(
    pjrt_buffer(raw_data, dtype = tengen::FloatType(32), shape = c(2, 3), row_major = FALSE),
    pjrt_buffer(raw_data, dtype = "f32", shape = c(2, 3), row_major = FALSE)
  )

  # raw scalar with DataType
  raw_scalar <- as.raw(rep(0, 4))
  expect_equal(
    pjrt_scalar(raw_scalar, dtype = tengen::FloatType(32)),
    pjrt_scalar(raw_scalar, dtype = "f32")
  )

  # identity preserves buffer when DataType matches
  buf <- pjrt_buffer(c(1, 2), dtype = "f32")
  expect_equal(pjrt_buffer(buf, dtype = tengen::FloatType(32)), buf)
  expect_error(pjrt_buffer(buf, dtype = tengen::IntegerType(32)), "Must use the same data type")
})

test_that("raw buffer validates dtype and shape compatibility", {
  # f32 is 4 bytes per element, shape c(2, 3) = 6 elements = 24 bytes
  expect_error(
    pjrt_buffer(as.raw(1:10), dtype = "f32", shape = c(2, 3), row_major = FALSE),
    "Raw data has 10 bytes, but dtype.*f32.*with shape.*2, 3.*requires 24 bytes"
  )

  # too many bytes
  expect_error(
    pjrt_buffer(as.raw(1:8), dtype = "f32", shape = 1L, row_major = FALSE),
    "Raw data has 8 bytes, but dtype.*f32.*with shape.*1.*requires 4 bytes"
  )

  # correct size should work
  buf <- pjrt_buffer(as.raw(rep(0, 24)), dtype = "f32", shape = c(2, 3), row_major = FALSE)
  expect_equal(shape(buf), c(2, 3))

  # scalar from raw: f32 needs exactly 4 bytes
  expect_error(
    pjrt_scalar(as.raw(1:2), dtype = "f32"),
    "Raw data has 2 bytes, but dtype.*f32.*requires 4 bytes"
  )
})
# Async buffer-to-host tests

test_that("as_array_async returns PJRTArrayPromise", {
  buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
  result <- as_array_async(buf)
  expect_class(result, "PJRTArrayPromise")
})

test_that("is_ready works for async buffers", {
  buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), dtype = "f32")
  result <- as_array_async(buf)
  ready <- is_ready(result)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("value() returns correct array for async buffer", {
  original <- matrix(c(1.0, 2.0, 3.0, 4.0), nrow = 2)
  buf <- pjrt_buffer(original, dtype = "f32")
  result <- as_array_async(buf)
  arr <- value(result)
  expect_equal(arr, original, tolerance = 1e-6)
})

test_that("as_array() works for async buffers", {
  original <- array(c(1.0, 2.0, 3.0))
  buf <- pjrt_buffer(original, dtype = "f32")
  result <- as_array_async(buf)
  arr <- as_array(result)
  expect_equal(arr, original, tolerance = 1e-6)
})

test_that("async buffer works with different dtypes", {
  # f64
  buf <- pjrt_buffer(c(1.0, 2.0), dtype = "f64")
  result <- value(as_array_async(buf))
  expect_equal(as.vector(result), c(1.0, 2.0))

  # i32
  buf <- pjrt_buffer(c(1L, 2L, 3L), dtype = "i32")
  result <- value(as_array_async(buf))
  expect_equal(as.vector(result), c(1L, 2L, 3L))

  # pred
  buf <- pjrt_buffer(c(TRUE, FALSE, TRUE), dtype = "pred")
  result <- value(as_array_async(buf))
  expect_equal(as.vector(result), c(TRUE, FALSE, TRUE))
})

test_that("print.PJRTArrayPromise works", {
  buf <- pjrt_buffer(c(1.0, 2.0), dtype = "f32")
  result <- as_array_async(buf)
  expect_output(print(result), "PJRTArrayPromise")
})

# is_ready / await for PJRTBuffer

test_that("is_ready works for PJRTBuffer", {
  x <- pjrt_buffer(c(1.0, 2.0), dtype = "f32")
  ready <- is_ready(x)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("await works for PJRTBuffer", {
  original <- c(1.0, 2.0, 3.0, 4.0)
  x <- pjrt_buffer(original, shape = c(2, 2), dtype = "f32")
  buf <- await(x)
  expect_class(buf, "PJRTBuffer")
  expect_equal(as.vector(as_array(buf)), original, tolerance = 1e-6)
})

test_that("pjrt_memory returns a PJRTMemory", {
  skip_if_metal("PJRT_Buffer_Memory not implemented")
  buf <- pjrt_buffer(1, dtype = "f32")
  mem <- pjrt_memory(buf)
  expect_class(mem, "PJRTMemory")
  expect_output(print(mem), "PJRTMemory")
})

test_that("await properly releases preserved objects", {
  # await() calls process_pending_releases(), which drains the deferred
  # release queue. Without this, memory would leak on backends with
  # async transfers (GPU/TPU).

  gc()
  gc()
  baseline <- gc()[2, 2] # Vcells used (MB)

  for (i in seq_len(50)) {
    # Zero-copy path: f64 from double, i32 from integer
    data_f64 <- as.double(seq_len(50000)) # ~400KB each
    buf_f64 <- pjrt_buffer(data_f64, dtype = "f64")
    await(buf_f64)
    rm(buf_f64, data_f64)

    data_i32 <- seq_len(50000) # ~200KB each
    buf_i32 <- pjrt_buffer(data_i32, dtype = "i32")
    await(buf_i32)
    rm(buf_i32, data_i32)

    # Copy path: f32 from double (requires type conversion)
    data_f32 <- as.double(seq_len(50000)) # ~200KB copy
    buf_f32 <- pjrt_buffer(data_f32, dtype = "f32")
    await(buf_f32)
    rm(buf_f32, data_f32)
  }

  # Force GC to reclaim any properly-released objects
  gc()
  gc()
  after <- gc()[2, 2]

  # Memory growth should be minimal
  memory_growth_mb <- after - baseline
  expect_lt(memory_growth_mb, 10)
})

describe("copy_buffer", {
  it("copies buffer to a different cpu device", {
    buf <- pjrt_buffer(c(1, 2, 3), device = "cpu:0")
    buf2 <- copy_buffer(buf, "cpu:1")

    expect_class(buf2, "PJRTBuffer")
    expect_equal(device(buf2), pjrt_device("cpu:1"))
    expect_equal(as_array(buf2), as_array(buf))
  })

  it("preserves dtype and shape", {
    buf <- pjrt_buffer(matrix(1:6, nrow = 2), dtype = "i32", device = "cpu:0")
    buf2 <- copy_buffer(buf, "cpu:1")

    expect_equal(shape(buf2), shape(buf))
    expect_equal(dtype(buf2), dtype(buf))
    expect_equal(as_array(buf2), as_array(buf))
  })

  it("leaves original buffer unchanged", {
    buf <- pjrt_buffer(c(1, 2, 3), device = "cpu:0")
    buf2 <- copy_buffer(buf, "cpu:1")

    expect_equal(device(buf), pjrt_device("cpu:0"))
    expect_equal(as_array(buf), array(c(1, 2, 3)))
  })

  it("returns identical buffer for same device", {
    buf <- pjrt_buffer(c(1, 2, 3), device = "cpu:0")
    buf2 <- copy_buffer(buf, "cpu:0")
    expect_identical(buf2, buf)
  })

  it("copies from cpu to cuda (cross-client)", {
    skip_if(!is_cuda())
    buf <- pjrt_buffer(matrix(c(1, 2, 3, 4), nrow = 2), dtype = "f32", device = "cpu:0")
    buf2 <- copy_buffer(buf, "cuda:0")

    expect_equal(device(buf2), pjrt_device("cuda:0"))
    expect_equal(as_array(buf2), as_array(buf))
    expect_equal(shape(buf2), shape(buf))
    expect_equal(dtype(buf2), dtype(buf))
  })

  it("copies from cuda to cpu (cross-client)", {
    skip_if(!is_cuda())
    buf <- pjrt_buffer(matrix(c(1, 2, 3, 4), nrow = 2), dtype = "f32", device = "cuda:0")
    buf2 <- copy_buffer(buf, "cpu:0")

    expect_equal(platform(device(buf2)), "cpu")
    expect_equal(as_array(buf2), as_array(buf))
    expect_equal(shape(buf2), shape(buf))
    expect_equal(dtype(buf2), dtype(buf))
  })
})

test_that("as_array respects a non-row-major executable output layout", {
  skip_if(!is_cpu())
  # Pin a column-major output layout via `mhlo.layout_mode = "{0,1}"` on the
  # function result (default is row-major, `{1,0}`). The device buffer is then
  # NOT row-major; readback must still return the correct logical matrix rather
  # than a transposed/garbled one.
  mlir <- '
module {
  func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32> {mhlo.layout_mode = "{0,1}"}) {
    return %arg0 : tensor<2x3xf32>
  }
}
'
  prog <- pjrt_program(src = mlir, format = "mlir")
  exec <- pjrt_compile(prog, device = "cpu")
  m <- matrix(as.double(1:6), nrow = 2, ncol = 3)
  out <- pjrt_execute(exec, pjrt_buffer(m, dtype = "f32"))
  expect_equal(as_array(out), m, tolerance = 1e-6)
  # as_raw goes through the same device→host path, so it must agree too.
  expect_equal(as_raw(out, row_major = FALSE), as_raw(pjrt_buffer(m, dtype = "f32"), row_major = FALSE))
})

describe("CPU buffer memory management", {
  vcells_mb <- function() {
    g <- gc(full = TRUE, verbose = FALSE)
    g["Vcells", "(Mb)"]
  }

  # The backing RAWSXP holds the payload plus up to 63 bytes of padding so the
  # data PJRT aliases can be 64-byte aligned (XLA's CPU client refuses the
  # zero-copy import below cpu::MinAlign()). The aliasing check proves the
  # buffer's device memory genuinely lives inside the RAWSXP -- i.e. the
  # zero-copy path was actually taken, not a silent copy into pool memory.
  expect_raw_backing <- function(buf, bytes) {
    p <- impl_test_xptr_prot(buf)
    expect_true(is.raw(p))
    expect_gte(length(p), bytes)
    expect_lt(length(p), bytes + 64L)
    expect_true(impl_test_buffer_aliases_prot(buf))
  }

  it("keeps pjrt_buffer's backing RAWSXP in the XPtr's prot slot", {
    skip_if(!is_cpu())
    nfloats <- 1024L * 256L
    b <- pjrt_buffer(matrix(1.25, nfloats, 1), dtype = "f32")
    expect_raw_backing(b, nfloats * 4L)
  })

  it("keeps pjrt_empty's backing RAWSXP in the XPtr's prot slot", {
    skip_if(!is_cpu())
    e <- pjrt_empty("f32", c(256L, 256L))
    expect_raw_backing(e, 256L * 256L * 4L)
  })

  # Every CPU buffer must be backed by a prot-slot RAWSXP, including the
  # paths where R's in-memory layout already matches the dtype byte-for-byte
  # (so no per-element conversion happens) and the raw path. Otherwise that
  # buffer's host memory would be invisible to R's garbage collector.
  it("keeps the backing RAWSXP for a no-conversion f64 buffer", {
    skip_if(!is_cpu())
    n <- 4096L
    b <- pjrt_buffer(as.double(seq_len(n)), dtype = "f64")
    expect_raw_backing(b, n * 8L)
  })

  it("keeps the backing RAWSXP for a no-conversion i32 buffer", {
    skip_if(!is_cpu())
    n <- 4096L
    b <- pjrt_buffer(seq_len(n), dtype = "i32")
    expect_raw_backing(b, n * 4L)
  })

  it("keeps the backing RAWSXP for an integer64 buffer", {
    skip_if(!is_cpu())
    n <- 4096L
    b <- pjrt_buffer(bit64::as.integer64(seq_len(n)), dtype = "i64")
    expect_raw_backing(b, n * 8L)
  })

  it("keeps the backing RAWSXP for a raw buffer", {
    skip_if(!is_cpu())
    b <- pjrt_buffer(raw(4096L * 4L), dtype = "f32", shape = 4096L, row_major = FALSE)
    expect_raw_backing(b, 4096L * 4L)
  })

  it("reports the backing RAWSXP size via object.size, not just the pointer", {
    skip_if(!is_cpu())
    nfloats <- 1024L * 256L
    expected_bytes <- nfloats * 4L
    b <- pjrt_buffer(matrix(1.25, nfloats, 1), dtype = "f32")
    size <- as.numeric(object.size(b))
    # The RAWSXP backing the buffer lives in the XPtr's prot slot, so
    # object.size traverses it and reports (at least) the data's bytes,
    # rather than the handful of bytes an external pointer alone occupies.
    expect_gte(size, expected_bytes)
    # And it's the data dominating the size, not some unrelated allocation.
    expect_lt(size - expected_bytes, 1024L)
  })

  it("keeps RAWSXPs alive under GC pressure while the XPtr is reachable", {
    skip_if(!is_cpu())
    bufs <- lapply(1:8, function(i) {
      pjrt_buffer(matrix(0.5, 1024L * 1024L, 1), dtype = "f32")
    })
    for (i in 1:5) {
      invisible(rnorm(1e5))
      gc(full = TRUE)
    }
    expect_true(all(as.numeric(as_array(bufs[[1]])) == 0.5))
  })

  it("reclaims RAWSXPs when XPtrs go out of scope", {
    skip_if(!is_cpu())
    gc(full = TRUE)
    gc(full = TRUE)
    before <- vcells_mb()

    bufs <- lapply(1:20, function(i) {
      pjrt_buffer(matrix(0.5, 1024L * 1024L, 1), dtype = "f32")
    })
    during <- vcells_mb()
    # 20 buffers x 4 MB f32 = ~80 MB. Allow some slack for accounting noise.
    expect_gt(during - before, 70)

    rm(bufs)
    gc(full = TRUE)
    gc(full = TRUE)
    after <- vcells_mb()
    expect_lt(abs(after - before), 5)
  })
})

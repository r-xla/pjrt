# Helper function to check scalar roundtrip
test_pjrt_scalar <- function(
  data,
  elt_type = NULL,
  tolerance = testthat::testthat_tolerance()
) {
  stopifnot(is.atomic(data) && length(data) == 1)
  args <- list(data = data)
  args$elt_type <- elt_type
  buffer <- do.call(pjrt_scalar, args)

  expect_class(buffer, "PJRTBuffer")
  result <- as_array(buffer)
  expect_true(!is.array(result))

  # Check that scalar becomes 1D array with length 1
  expect_true(is.null(dim(result)))

  modify <- function(data) {
    data[1L] <- if (is.numeric(data)) {
      1L
    } else if (is.logical(data)) {
      !data[1L]
    } else {
      stop("Unsupported data type: ", typeof(data))
    }
  }

  # Test that modifying the original data doesn't change the buffer
  original_data <- data
  data[1L] <- modify(data)

  result_after_modification <- as_array(buffer)
  expect_equal(result_after_modification, original_data, tolerance = tolerance)

  # Test that modifying the result doesn't persist when recreating from buffer
  result[1L] <- modify(result)

  result_recreated <- as_array(buffer)
  expect_equal(result_recreated, original_data, tolerance = tolerance)

  TRUE
}

# Helper function to check buffer roundtrip
test_pjrt_buffer <- function(
  data,
  elt_type = NULL,
  tolerance = testthat::testthat_tolerance()
) {
  args <- list(data = data)
  args$elt_type <- elt_type
  buffer <- do.call(pjrt_buffer, args)

  expect_class(buffer, "PJRTBuffer")
  result <- as_array(buffer)
  expect_true(is.array(result))

  data_arr <- as.array(data)

  expect_equal(result, data_arr, tolerance = tolerance)

  modify_first <- function(data) {
    data[1L] +
      if (is.numeric(data)) {
        1L
      } else if (is.logical(data)) {
        !data[1L]
      } else {
        stop("Unsupported data type: ", typeof(data))
      }
  }

  # Check dimensions are preserved
  if (is.null(dim(data))) {
    # Vector should become 1D array
    expect_equal(dim(result), length(data))
  } else {
    # Array should preserve dimensions
    expect_equal(dim(result), dim(data))
  }

  # Test that modifying the original data doesn't change the buffer
  original_data <- data
  data[1L] <- modify_first(data)

  result_after_modification <- as_array(buffer)
  expect_equal(result_after_modification, data_arr, tolerance = tolerance)

  # Test that modifying the result doesn't persist when recreating from buffer
  result[1L] <- modify_first(result)

  result_recreated <- as_array(buffer)
  if (!is.null(tolerance)) {
    expect_equal(result_recreated, data_arr, tolerance = tolerance)
  } else {
    expect_equal(result_recreated, data_arr)
  }

  return(buffer)
}

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

test_that("pjrt_buffer handles edge cases", {
  # Test empty vectors
  expect_error(pjrt_buffer(logical(0)), "Data must be a non-empty vector")
  expect_error(pjrt_buffer(integer(0)), "Data must be a non-empty vector")
  expect_error(pjrt_buffer(numeric(0)), "Data must be a non-empty vector")
})

test_that("pjrt_buffer preserves 3d dimensions", {
  # Test 3D array
  arr_3d <- array(1:24, dim = c(2, 3, 4))
  test_pjrt_buffer(arr_3d)
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
    f32 = list(data = matrix(runif(4), nrow = 2), elt_type = "f32"),
    f64 = list(data = matrix(runif(6), nrow = 3), elt_type = "f64"),

    i8 = list(data = sample_signed(6, c(3, 2)), elt_type = "i8"),
    i16 = list(data = sample_signed(14, c(4, 3)), elt_type = "i16"),
    i32 = list(data = sample_signed(30, c(4, 3, 1)), elt_type = "i32"),
    i64 = list(data = sample_signed(30, c(2, 3, 7)), elt_type = "i64"),

    ui8 = list(data = sample_unsigned(6, c(1, 1)), elt_type = "ui8"),
    ui16 = list(data = sample_unsigned(14, c(2, 1)), elt_type = "ui16"),
    ui32 = list(data = sample_unsigned(30, c(2, 2, 2)), elt_type = "ui32"),
    ui64 = list(data = sample_unsigned(30, c(2, 1, 2, 1)), elt_type = "ui64"),

    pred = list(
      data = matrix(c(TRUE, FALSE, TRUE, FALSE), nrow = 2),
      elt_type = "pred"
    )
  )

  for (test_name in names(test_cases)) {
    test_case <- test_cases[[test_name]]
    original_data <- test_case$data
    elt_type <- test_case$elt_type

    # Full roundtrip: R → PJRT buffer → raw → PJRT buffer → R
    buf1 <- pjrt_buffer(original_data, elt_type = elt_type)
    raw_data <- as_raw(buf1, row_major = FALSE)
    buf2 <- pjrt_buffer(
      raw_data,
      elt_type = elt_type,
      shape = dim(original_data),
      row_major = FALSE
    )
    roundtrip_data <- as_array(buf2)

    # Compare original with roundtrip
    if (elt_type %in% c("f32", "f64")) {
      expect_equal(roundtrip_data, original_data, tolerance = 1e-6)
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

  for (elt_type in names(test_scalars)) {
    original <- test_scalars[[elt_type]]

    buf1 <- pjrt_scalar(original, elt_type = elt_type)
    raw_data <- as_raw(buf1, row_major = FALSE)
    buf2 <- pjrt_scalar(raw_data, elt_type = elt_type)
    roundtrip <- as_array(buf2)

    if (elt_type %in% c("f32", "f64")) {
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

test_that("element_type returns correct data types", {
  # Test logical buffer
  logical_data <- c(TRUE, FALSE, TRUE)
  buffer <- pjrt_buffer(logical_data)
  dtype <- element_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "pred")

  # Test integer buffer (signed 32-bit)
  integer_data <- c(1L, 2L, 3L)
  buffer <- pjrt_buffer(integer_data, "i32")
  dtype <- element_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "i32")

  # Test unsigned integer buffer (8-bit)
  buffer <- pjrt_buffer(integer_data, "ui8")
  dtype <- element_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "ui8")

  # Test double buffer (32-bit)
  double_data <- c(1.1, 2.2, 3.3)
  buffer <- pjrt_buffer(double_data, "f32")
  dtype <- element_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "f32")

  # Test double buffer (64-bit)
  buffer <- pjrt_buffer(double_data, "f64")
  dtype <- element_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "f64")

  # Test scalar buffer
  scalar_data <- 42L
  buffer <- pjrt_scalar(scalar_data)
  dtype <- element_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "i32")
})

test_that("R layout and PJRT layout (2D)", {
  skip_if_metal()
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
  skip_if_metal()
  path <- system.file("programs/jax-stablehlo-subset-3d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  x <- array(as.double(1:24), dim = c(2, 3, 4))
  x_buf <- pjrt_buffer(x)

  check <- function(i1, i2, i3) {
    i1_buf <- pjrt_scalar(i1, "i32")
    i2_buf <- pjrt_scalar(i2, "i32")
    i3_buf <- pjrt_scalar(i3, "i32")

    result <- as_array(pjrt_execute(
      executable,
      x_buf,
      i1_buf,
      i2_buf,
      i3_buf
    ))
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
  types = list(
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
    data_sexp <- switch(
      rtype,
      integer = 1:6L,
      double = as.double(1:6),
      logical = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
      stop()
    )
    size <- switch(rtype, integer = 4, double = 8, logical = 4, stop())
    data_raw <- writeBin(data_sexp, raw(), size = size)

    ## pjrt_buffer()

    # for ints we have:
    # buf1: [1, 2, 3, 4, 5, 6] that represents [[1, 2], [3, 4], [5, 6]]
    buf1 <- pjrt_buffer(
      data_raw,
      elt_type = "ui8",
      row_major = TRUE,
      shape = c(3, 2)
    )
    # buf2: [1, 2, 3, 4, 5, 6] that represents [[1, 3, 5], [2, 4, 6]]
    buf2 <- pjrt_buffer(
      data_raw,
      elt_type = "ui8",
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
      elt_type = "ui8",
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

test_that("tests can compare buffers", {
  expect_equal(pjrt_buffer(1), pjrt_buffer(1))
  expect_failure(expect_equal(pjrt_buffer(1), pjrt_buffer(2)))
  expect_failure(expect_equal(pjrt_buffer(1), pjrt_scalar(1)))
})

test_that("No dim with pjrt_buffer", {
  skip_if(is_cuda() || is_metal())
  expect_equal(dim(pjrt_buffer(1)), 1L)
})

test_that("device print", {
  skip_if(is_cuda() || is_metal())
  expect_snapshot(print(device(pjrt_buffer(1))))
})

test_that("dim is integer", {
  expect_true(is.integer(dim(pjrt_buffer(1))))
})

test_that("can move back buffer without specifying client", {
  skip_if(!(is_metal() || is_cuda()))
  client <- if (is_metal()) "metal" else "cuda"
  x <- pjrt_buffer(1, client = client)
  expect_equal(as_array(x), array(1))
  y <- pjrt_scalar(1, client = client)
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
  expect_equal(dim(pjrt_buffer(1:4, shape = c(2, 2))), c(2, 2))
})

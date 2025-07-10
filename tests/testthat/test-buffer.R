# Helper function to check scalar roundtrip
test_pjrt_scalar <- function(
  data,
  type = NULL,
  tolerance = testthat::testthat_tolerance()
) {
  stopifnot(is.atomic(data) && length(data) == 1)
  args <- list(data = data)
  args$type <- type
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
  type = NULL,
  tolerance = testthat::testthat_tolerance()
) {
  args <- list(data = data)
  args$type <- type
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
  test_pjrt_scalar(-1L, type = "s8")
  test_pjrt_scalar(-2L, type = "s16")
  test_pjrt_scalar(-3L, type = "s64")

  test_pjrt_scalar(1L, type = "u8")
  test_pjrt_scalar(2L, type = "u16")
  test_pjrt_scalar(3L, type = "u64")

  # Test double scalar
  test_pjrt_scalar(3.14, type = "f64")
  test_pjrt_scalar(-3, type = "f32")
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
  test_pjrt_buffer(data_32, type = "f32", tolerance = 1e-6)

  # Test double precision (64-bit)
  data_64 <- c(1.0, -1.0, 0.0, 3.14159265359, -2.71828182846)
  test_pjrt_buffer(data_64, type = "f64", tolerance = 1e-12)

  # Test arrays with dimensions
  data_matrix <- matrix(c(1.1, 2.2, 3.3, 4.4), nrow = 2, ncol = 2)
  test_pjrt_buffer(data_matrix, type = "f32", tolerance = 1e-6)
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


test_that("pjrt_elt_type returns correct data types", {
  # Test logical buffer
  logical_data <- c(TRUE, FALSE, TRUE)
  buffer <- pjrt_buffer(logical_data)
  dtype <- pjrt_elt_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "pred")

  # Test integer buffer (signed 32-bit)
  integer_data <- c(1L, 2L, 3L)
  buffer <- pjrt_buffer(integer_data, type = "s32")
  dtype <- pjrt_elt_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "s32")

  # Test unsigned integer buffer (8-bit)
  buffer <- pjrt_buffer(integer_data, type = "u8")
  dtype <- pjrt_elt_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "u8")

  # Test double buffer (32-bit)
  double_data <- c(1.1, 2.2, 3.3)
  buffer <- pjrt_buffer(double_data, type = "f32")
  dtype <- pjrt_elt_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "f32")

  # Test double buffer (64-bit)
  buffer <- pjrt_buffer(double_data, type = "f64")
  dtype <- pjrt_elt_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "f64")

  # Test scalar buffer
  scalar_data <- 42L
  buffer <- pjrt_scalar(scalar_data)
  dtype <- pjrt_elt_type(buffer)
  expect_true(is_element_type(dtype))
  expect_equal(as.character(dtype), "s32")
})

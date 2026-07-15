# Extracted from test-buffer.R:1134

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "pjrt", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
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

# test -------------------------------------------------------------------------
vcells_mb <- function() {
    g <- gc(full = TRUE, verbose = FALSE)
    g["Vcells", "(Mb)"]
  }
it("keeps pjrt_buffer's backing RAWSXP in the XPtr's prot slot", {
    skip_if(!is_cpu())
    nfloats <- 1024L * 256L
    b <- pjrt_buffer(matrix(1.25, nfloats, 1), dtype = "f32")
    p <- impl_test_xptr_prot(b)
    expect_true(is.raw(p))
    expect_equal(length(p), nfloats * 4L)
  })
it("keeps pjrt_empty's backing RAWSXP in the XPtr's prot slot", {
    skip_if(!is_cpu())
    e <- pjrt_empty("f32", c(256L, 256L))
    p <- impl_test_xptr_prot(e)
    expect_true(is.raw(p))
    expect_equal(length(p), 256L * 256L * 4L)
  })
it("keeps the backing RAWSXP for a no-conversion f64 buffer", {
    skip_if(!is_cpu())
    n <- 4096L
    b <- pjrt_buffer(as.double(seq_len(n)), dtype = "f64")
    p <- impl_test_xptr_prot(b)
    expect_true(is.raw(p))
    expect_equal(length(p), n * 8L)
  })
it("keeps the backing RAWSXP for a no-conversion i32 buffer", {
    skip_if(!is_cpu())
    n <- 4096L
    b <- pjrt_buffer(seq_len(n), dtype = "i32")
    p <- impl_test_xptr_prot(b)
    expect_true(is.raw(p))
    expect_equal(length(p), n * 4L)
  })
it("keeps the backing RAWSXP for an integer64 buffer", {
    skip_if(!is_cpu())
    n <- 4096L
    b <- pjrt_buffer(bit64::as.integer64(seq_len(n)), dtype = "i64")
    p <- impl_test_xptr_prot(b)
    expect_true(is.raw(p))
    expect_equal(length(p), n * 8L)
  })

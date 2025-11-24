test_that("format_buffer works for floats", {
  expect_snapshot(format_buffer(pjrt_buffer(1.23, "f32")))
  expect_snapshot(format_buffer(pjrt_buffer(1.23, "f64")))
  expect_snapshot(format_buffer(pjrt_buffer(-3.33, "f64")))
  expect_snapshot(format_buffer(pjrt_buffer(NaN, "f32")))
  expect_snapshot(format_buffer(pjrt_buffer(Inf, "f32")))
  expect_snapshot(format_buffer(pjrt_buffer(-Inf, "f32")))
})

test_that("format_buffer works for integers", {
  check <- function(x, dtype) {
    buf <- pjrt_buffer(x, dtype = dtype)
    res <- format_buffer(buf)
    expect_equal(as.character(res), as.character(x))
  }

  check(-5:5, "i32")
  check(-5:5, "i64")
  check(-5:5, "i16")
  check(-5:5, "i8")
  check(1:5, "ui32")
  check(1:5, "ui64")
  check(1:5, "ui16")
  check(1:5, "ui8")

  # Large integers
  # [0000 0000] ... [0000 0001]
  # 0x00        ... 0x80 (16 * 8 = 128 = 2^7 = 0000 0001)
  smallest_i64_raw <- as.raw(c(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80))
  buf <- pjrt_buffer(smallest_i64_raw, dtype = "i64", shape = integer(), row_major = TRUE)
  expect_equal(format_buffer(buf), "-9223372036854775808")

  # [1111 1111] ... [1111 1110]
  largest_i64_raw <- as.raw(c(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f))
  buf <- pjrt_buffer(largest_i64_raw, dtype = "i64", shape = integer(), row_major = TRUE)
  expect_equal(format_buffer(buf), "9223372036854775807")

  # [1111 1111] ... [1111 1111]
  largest_u64_raw <- as.raw(rep(0xff, 8))
  buf <- pjrt_buffer(largest_u64_raw, dtype = "ui64", shape = integer(), row_major = TRUE)
  expect_equal(format_buffer(buf), "18446744073709551615")
})

test_that("format_buffer works for logicals", {
  skip_if_not_installed("pjrt")

  buf <- pjrt_buffer(c(TRUE, FALSE), dtype = "pred")
  res <- format_buffer(buf)
  expect_equal(res, array(c("true", "false"), dim = 2L))
})

test_that("format_buffer returns vector of characters", {
  skip_if_not_installed("pjrt")

  buf <- pjrt_buffer(array(1:6, dim = c(2, 3)), dtype = "i32")
  res <- format_buffer(buf)
  expect_length(res, 6)
  expect_type(res, "character")
  expect_equal(as.vector(res), as.character(1:6))
})

test_that("format_buffer preserves dimensions", {
  dims <- c(2L, 3L)
  buf <- pjrt_buffer(array(1:6, dim = dims), dtype = "i32")
  expect_equal(format_buffer(buf), array(as.character(1:6), dim = dims))
})

test_that("format_buffer works with empty buffer", {
  buf <- pjrt_empty("i32", shape = c(2, 3, 0))
  expect_equal(format_buffer(buf), array(character(), dim = c(2, 3, 0)))
})

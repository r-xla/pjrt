test_that("printer for integers", {
  expect_snapshot(pjrt_buffer(1L, "i8"))
  expect_snapshot(pjrt_buffer(1L, "i16"))
  expect_snapshot(pjrt_buffer(1L, "i32"))

  expect_snapshot(pjrt_buffer(1L, "ui8"))
  expect_snapshot(pjrt_buffer(1L, "ui16"))
  expect_snapshot(pjrt_buffer(1L, "ui32"))

  expect_snapshot(pjrt_buffer(c(1L, 2L, 3L)))
  expect_snapshot(pjrt_buffer(matrix(1:10, nrow = 2)))
})

test_that("printer for integers with large difference", {
  x <- matrix(as.integer(c(1:3 * 10000000L, 1L)), nrow = 2)
  expect_snapshot(pjrt_buffer(x))
})

test_that("inf, nan", {
  expect_snapshot(pjrt_buffer(c(Inf, -Inf, NaN, 1)))
  expect_snapshot(pjrt_buffer(c(Inf, Inf, NA, 100011.234567)))
  expect_snapshot(pjrt_buffer(c(Inf, Inf, NA, 100011.234567), shape = c(1, 4)))
})

test_that("Up to 6 digits are printed for integers", {
  # up to 6 digits are printed
  expect_snapshot(pjrt_buffer(1234567L))
  expect_snapshot(pjrt_buffer(-1234567L))
  expect_snapshot(pjrt_buffer(123456L))
  expect_snapshot(pjrt_buffer(-123456L))
})

test_that("printer for doubles", {
  expect_snapshot(pjrt_buffer(1:10, "f32"))
  expect_snapshot(pjrt_buffer(1:10, "f64"))

  # very small values:
  expect_snapshot(pjrt_buffer(1e-10))
  # large and small values:
  expect_snapshot(pjrt_buffer(c(1e10, 1e-10)))
})

test_that("printer for arrays with many dimensions", {
  expect_snapshot(pjrt_buffer(1:20, shape = c(1, 1, 1, 1, 1, 5, 4)))
})

test_that("printer for arrays with many elements", {
  expect_snapshot(pjrt_buffer(1:20000, shape = c(100, 200)))
})

test_that("column width is determined per slice", {
  x <- c(1, 100, 2, 200, 3, 300, 4, 400)
  expect_snapshot(pjrt_buffer(x, shape = c(2, 2, 2)))
})

test_that("1d vector", {
  expect_snapshot(pjrt_buffer(1:50L))
  expect_snapshot(pjrt_buffer(as.double(1:50)))
})

test_that("pjrt_buffer print integers and logicals correctly", {
  skip_if(is_cuda() || is_metal())
  int_mat <- matrix(c(-12L, 3L, 45L, -7L), nrow = 2)
  buf_int <- pjrt_buffer(int_mat, etype = "i32")
  expect_snapshot(print(buf_int))

  log_mat <- matrix(c(TRUE, FALSE, TRUE, FALSE), nrow = 2)
  buf_log <- pjrt_buffer(log_mat, etype = "pred")
  expect_snapshot(print(buf_log))
})

test_that("printer shows last two dims as matrix for high-rank arrays", {
  skip_if(is_cuda() || is_metal())
  x <- array(1:20, dim = c(1, 1, 1, 1, 1, 5, 4))
  buf <- pjrt_buffer(x, etype = "i32")
  expect_snapshot(print(buf))
})

test_that("alignment is as expected", {
  expect_snapshot(pjrt_buffer(c(1000L, 1L, 10L, 100L), shape = c(1, 4)))
})

test_that("wide arrays", {
  expect_snapshot(pjrt_buffer(1:100, shape = c(1, 2, 50)))
  expect_snapshot(pjrt_buffer(1:1000, shape = c(1, 2, 500)))
})

test_that("scalar", {
  expect_snapshot(pjrt_scalar(1, "f32"))
  expect_snapshot(pjrt_scalar(-10.1213, "f64"))

  # no prefix when printing just one element
  expect_snapshot(pjrt_scalar(10^6, "f32"))
  expect_snapshot(pjrt_scalar(-10^6, "f32"))
  expect_snapshot(pjrt_scalar(10^5, "f32"))
  expect_snapshot(pjrt_scalar(-10^5, "f32"))

  expect_snapshot(pjrt_scalar(250L, "ui8"))
  expect_snapshot(pjrt_scalar(12L, "ui16"))
  expect_snapshot(pjrt_scalar(0L, "ui32"))
  expect_snapshot(pjrt_scalar(998L, "ui64"))

  expect_snapshot(pjrt_scalar(14L, "i8"))
  expect_snapshot(pjrt_scalar(-12L, "i16"))
  expect_snapshot(pjrt_scalar(0L, "i32"))
  expect_snapshot(pjrt_scalar(998L, "i64"))

  expect_snapshot(pjrt_scalar(TRUE))
  expect_snapshot(pjrt_scalar(FALSE))
})

test_that("printer options", {
  # can restrict max_rows
  expect_snapshot(print(pjrt_buffer(1:100), max_rows = 10))

  # truncation not printed when everything is printed
  expect_snapshot(print(pjrt_buffer(1:100), max_rows = 100))

  # truncation is printed when not all columns are
  expect_snapshot(print(pjrt_buffer(1:10000)))

  # truncation is printed when not all rows are
  expect_snapshot(print(pjrt_buffer(1:11, shape = c(11, 1)), max_rows = 10))
})

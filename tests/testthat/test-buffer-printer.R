# we skip snapshot tests on metal and cuda, because the printer prints the platform name

test_that("printer for integers", {
  skip_if(is_metal() | is_cuda())
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
  skip_if(is_metal() | is_cuda())
  x <- matrix(as.integer(c(1:3 * 10000000L, 1L)), nrow = 2)
  expect_snapshot(pjrt_buffer(x))
})

test_that("inf, nan", {
  skip_if(is_metal() | is_cuda())
  expect_snapshot(pjrt_buffer(c(Inf, -Inf, NaN, 1)))
  expect_snapshot(pjrt_buffer(c(Inf, Inf, NA, 100011.234567)))
  expect_snapshot(pjrt_buffer(c(Inf, Inf, NA, 100011.234567), shape = c(1, 4)))
})

test_that("Up to 6 digits are printed for integers", {
  skip_if(is_metal() | is_cuda())
  # up to 6 digits are printed
  expect_snapshot(pjrt_buffer(1234567L))
  expect_snapshot(pjrt_buffer(-1234567L))
  expect_snapshot(pjrt_buffer(123456L))
  expect_snapshot(pjrt_buffer(-123456L))
})

test_that("printer for doubles", {
  skip_if(is_metal() | is_cuda())
  # integer-valued floats are printed without decimals
  expect_snapshot(pjrt_buffer(1:10, "f32"))
  expect_snapshot(pjrt_buffer(1:10, "f64"))

  # non-integer-valued floats keep decimals
  expect_snapshot(pjrt_buffer(c(1, 2.5), "f32"))

  # very small values:
  expect_snapshot(pjrt_buffer(1e-10))
  # large and small values:
  expect_snapshot(pjrt_buffer(c(1e10, 1e-10)))
})

test_that("integer-valued floats with truncation", {
  skip_if(is_metal() | is_cuda())
  # > 30 rows triggers row truncation
  expect_snapshot(pjrt_buffer(1:50, "f32", shape = c(50, 1)))

  # > 30 columns triggers column splitting
  expect_snapshot(pjrt_buffer(1:100, "f32", shape = c(2, 50)))

  # rank >= 3 with integer-valued floats
  expect_snapshot(pjrt_buffer(1:24, "f32", shape = c(2, 3, 4)))
})

test_that("printer for arrays with many dimensions", {
  skip_if(is_metal() | is_cuda())
  expect_snapshot(pjrt_buffer(1:20, shape = c(1, 1, 1, 1, 1, 5, 4)))
})

test_that("column width is determined per slice", {
  skip_if(is_metal() | is_cuda())
  x <- c(1, 100, 2, 200, 3, 300, 4, 400)
  pjrt_buffer(x, shape = c(2, 2, 2))
  expect_snapshot(pjrt_buffer(x, shape = c(2, 2, 2)))
})

test_that("1d vector", {
  skip_if(is_metal() | is_cuda())
  expect_snapshot(pjrt_buffer(1:50L))
  expect_snapshot(pjrt_buffer(as.double(1:50)))
})

test_that("logicals", {
  skip_if(is_metal() | is_cuda())
  log_mat <- matrix(c(TRUE, FALSE, TRUE, FALSE), nrow = 2)
  buf_log <- pjrt_buffer(log_mat, dtype = "pred")
  expect_snapshot(buf_log)
})

test_that("alignment is as expected", {
  skip_if(is_metal() | is_cuda())
  expect_snapshot(pjrt_buffer(c(1000L, 1L, 10L, 100L), shape = c(1, 4)))
})

test_that("wide arrays", {
  skip_if(is_metal() | is_cuda())
  expect_snapshot(pjrt_buffer(1:100, shape = c(1, 2, 50)))
  expect_snapshot(pjrt_buffer(1:1000, shape = c(1, 2, 500)))
})

test_that("scalar", {
  skip_if(is_metal() | is_cuda())
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
  skip_if(is_metal() | is_cuda())
  # can restrict max_rows
  expect_snapshot(print(pjrt_buffer(1:100), max_rows = 10))

  # truncation not printed when everything is printed
  expect_snapshot(print(pjrt_buffer(1:100), max_rows = 100))

  # truncation is printed when not all columns are
  expect_snapshot(print(pjrt_buffer(1:10000)))

  # truncation is printed when not all rows are
  expect_snapshot(print(pjrt_buffer(1:11, shape = c(11, 1)), max_rows = 10))

  # when max_width is too small, we print one column
  expect_snapshot(
    print(
      pjrt_buffer(c(100L), shape = c(1, 1)),
      max_width = 3,
      max_rows = 1
    )
  )

  # max_width; note that every data line starts with ' '
  x <- pjrt_buffer(rep(100L, 3), shape = c(1, 3))
  expect_snapshot(print(x, max_width = 7, max_rows = 1))
  expect_snapshot(print(x, max_width = 8, max_rows = 1))

  expect_snapshot(
    print(pjrt_buffer(rep(0.0000001, 50), shape = c(1, 50)), max_width = -1)
  )
})

test_that("scale prefix is printed per slice", {
  skip_if(is_metal() | is_cuda())
  x <- c(
    0.000001,
    0.000000001,
    0.000001,
    0.000000001
  )
  expect_snapshot(pjrt_buffer(x, shape = c(2, 1, 2)))

  expect_snapshot(
    print(
      pjrt_buffer(rep(x, 5), shape = c(2, 2, 5)),
      max_width = 15
    )
  )
})

test_that("metal", {
  skip_if(!is_metal())
  expect_snapshot(pjrt_buffer(1:10, "f32", device = "metal"))
})

test_that("stress test f32", {
  skip_if(!is_cpu())
  expect_snapshot(pjrt_buffer(c(1000000000L, 3L, 123L, Inf, -2, NaN), shape = c(3, 2), dtype = "f32"))
})

test_that("custom tail", {
  expect_snapshot(print(pjrt_buffer(1L), tail = "[abc]"))
})

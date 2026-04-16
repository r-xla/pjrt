test_that("copy buffer to a different cpu device", {
  buf <- pjrt_buffer(c(1, 2, 3), device = "cpu:0")
  buf2 <- pjrt_copy_to_device(buf, "cpu:1")

  expect_class(buf2, "PJRTBuffer")
  expect_equal(device(buf2), pjrt_device("cpu:1"))
  expect_equal(as_array(buf2), as_array(buf))
})

test_that("copy preserves dtype and shape", {
  buf <- pjrt_buffer(matrix(1:6, nrow = 2), dtype = "i32", device = "cpu:0")
  buf2 <- pjrt_copy_to_device(buf, "cpu:1")

  expect_equal(shape(buf2), shape(buf))
  expect_equal(dtype(buf2), dtype(buf))
  expect_equal(as_array(buf2), as_array(buf))
})

test_that("copy scalar buffer", {
  buf <- pjrt_scalar(42L, device = "cpu:0")
  buf2 <- pjrt_copy_to_device(buf, "cpu:1")

  expect_equal(shape(buf2), integer())
  expect_equal(as_array(buf2), 42L)
})

test_that("original buffer unchanged after copy", {
  buf <- pjrt_buffer(c(1, 2, 3), device = "cpu:0")
  buf2 <- pjrt_copy_to_device(buf, "cpu:1")

  expect_equal(device(buf), pjrt_device("cpu:0"))
  expect_equal(as_array(buf), array(c(1, 2, 3)))
})

test_that("copy to same device returns equivalent buffer", {
  buf <- pjrt_buffer(c(1, 2, 3), device = "cpu:0")
  buf2 <- pjrt_copy_to_device(buf, "cpu:0")
  expect_equal(as_array(buf2), as_array(buf))
})

test_that("copy buffer from cpu to cuda (cross-client)", {
  skip_if(!is_cuda())
  buf <- pjrt_buffer(matrix(c(1, 2, 3, 4), nrow = 2), dtype = "f32", device = "cpu:0")
  buf2 <- pjrt_copy_to_device(buf, "cuda:0")

  expect_equal(platform(device(buf2)), "cuda")
  expect_equal(as_array(buf2), as_array(buf))
  expect_equal(shape(buf2), shape(buf))
  expect_equal(dtype(buf2), dtype(buf))
})

test_that("copy buffer from cuda to cpu (cross-client)", {
  skip_if(!is_cuda())
  buf <- pjrt_buffer(matrix(c(1, 2, 3, 4), nrow = 2), dtype = "f32", device = "cuda:0")
  buf2 <- pjrt_copy_to_device(buf, "cpu:0")

  expect_equal(platform(device(buf2)), "cpu")
  expect_equal(as_array(buf2), as_array(buf))
  expect_equal(shape(buf2), shape(buf))
  expect_equal(dtype(buf2), dtype(buf))
})

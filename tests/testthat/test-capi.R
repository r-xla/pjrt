# pjrt's C interface for downstream packages (inst/include/pjrt/api.h).
#
# The helpers behind these tests live in src/test-capi.cpp, which includes the
# public header the way a downstream package does -- so every assertion here
# also covers the registration and the resolving stubs, not just the
# implementations in capi.cpp.

test_that("the registered API version matches the header", {
  v <- impl_capi_versions()
  expect_identical(v[["header"]], v[["registered"]])
})

test_that("class predicates identify each pjrt object", {
  skip_if_not(plugins_downloaded("cpu"))
  client <- pjrt_client("cpu")
  device <- pjrt_device("cpu:0")
  buffer <- pjrt_buffer(c(1, 2, 3), dtype = "f32")

  expect_identical(
    impl_capi_predicates(buffer),
    c(buffer = TRUE, client = FALSE, device = FALSE, executable = FALSE)
  )
  expect_identical(
    impl_capi_predicates(client),
    c(buffer = FALSE, client = TRUE, device = FALSE, executable = FALSE)
  )
  expect_identical(
    impl_capi_predicates(device),
    c(buffer = FALSE, client = FALSE, device = TRUE, executable = FALSE)
  )
  # A non-external-pointer must not be mistaken for any of them.
  expect_false(any(impl_capi_predicates(42L)))
  expect_false(any(impl_capi_predicates(NULL)))
})

test_that("buffer metadata reads back through the interface", {
  skip_if_not(plugins_downloaded("cpu"))
  buffer <- pjrt_buffer(array(as.numeric(1:6), c(2L, 3L)), dtype = "f32")
  meta <- impl_capi_buffer_meta(buffer)

  expect_identical(meta$dtype_name, "f32")
  expect_identical(meta$rank, 2L)
  expect_identical(meta$shape, c(2L, 3L))
  expect_s3_class(meta$device, "PJRTDevice")

  # A rank-0 buffer reports rank 0 and an empty shape, not a NULL dims pointer
  # mistaken for a failure.
  scalar_meta <- impl_capi_buffer_meta(pjrt_scalar(1, dtype = "f64"))
  expect_identical(scalar_meta$rank, 0L)
  expect_identical(scalar_meta$shape, integer())
  expect_identical(scalar_meta$dtype_name, "f64")
})

test_that("the dtype vocabulary round-trips", {
  for (nm in c("pred", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "f32", "f64")) {
    rt <- impl_capi_dtype_roundtrip(nm)
    expect_gte(rt$code, 0L)
    expect_identical(rt$name, nm)
  }
  # A dtype pjrt has no PJRT_Buffer_Type for is -1, not an error.
  expect_identical(impl_capi_dtype_roundtrip("bf16")$code, -1L)
})

test_that("failures come back on the error channel, never as an R error", {
  # The whole point of the channel: an R error here would be a longjmp through
  # the caller's C++ frames.
  res <- impl_capi_error_channel(42L)
  expect_identical(res$bad_dtype, -1L)
  expect_match(res$message, "expected a PJRTBuffer")
  # A subsequent success clears the message, so `last_error()` always describes
  # the most recent call rather than the most recent failure.
  expect_true(res$cleared)
  expect_gte(res$good_dtype, 0L)
  expect_identical(res$unknown_dtype, -1L)
  expect_match(res$unknown_message, "Unsupported type")
})

test_that("a device is one interned object however it is reached", {
  skip_if_not(plugins_downloaded("cpu"))
  buffer <- pjrt_buffer(c(1, 2), dtype = "f32", device = "cpu:0")
  device <- pjrt_device("cpu:0")
  expect_identical(
    impl_capi_device_identity(buffer, device),
    c(same_object = TRUE, same_token = TRUE, idempotent = TRUE)
  )
})

test_that("a buffer and its client report the same plugin", {
  skip_if_not(plugins_downloaded("cpu"))
  expect_true(impl_capi_same_client(
    pjrt_buffer(c(1, 2), dtype = "f32"),
    pjrt_client("cpu")
  ))
})

test_that("buffers can be uploaded and allocated through the interface", {
  skip_if_not(plugins_downloaded("cpu"))
  client <- pjrt_client("cpu")
  device <- pjrt_device("cpu:0")

  up <- impl_capi_buffer_from_r(client, device, c(1, 2, 3, 4), 4L, "f32")
  expect_s3_class(up, "PJRTBuffer")
  expect_equal(as.numeric(as_array(up)), c(1, 2, 3, 4))

  # Column-major round-trip of a 2-d array, the case the R layer handles.
  m <- array(as.numeric(1:6), c(2L, 3L))
  up2 <- impl_capi_buffer_from_r(client, device, m, c(2L, 3L), "f32")
  expect_equal(as_array(up2), m)

  empty <- impl_capi_buffer_empty(client, device, c(2L, 2L), "f32")
  expect_s3_class(empty, "PJRTBuffer")
  expect_identical(shape(empty), c(2L, 2L))
})

test_that("an executable runs through the interface", {
  skip_if_not(plugins_downloaded("cpu"))
  src <- "func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
            %0 = stablehlo.add %a, %b : tensor<3xf32>
            return %0 : tensor<3xf32>
          }"
  exec <- pjrt_compile(pjrt_program(src, format = "mlir"), device = "cpu:0")
  x <- pjrt_buffer(c(1, 2, 3), dtype = "f32")
  y <- pjrt_buffer(c(10, 20, 30), dtype = "f32")

  out <- impl_capi_execute(exec, list(x, y))
  expect_length(out, 1L)
  expect_equal(as.numeric(as_array(out[[1L]])), c(11, 22, 33))

  # A malformed call reports on the error channel instead of raising.
  expect_match(
    impl_capi_execute_error(exec, list(x, 42L)),
    "must be a PJRTBuffer"
  )
})

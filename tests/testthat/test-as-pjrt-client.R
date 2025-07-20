test_that("as_pjrt_client works with character input", {
  client <- as_pjrt_client("cpu")
  expect_s3_class(client, "PJRTClient")

  original_client <- pjrt_client("cpu")
  result <- as_pjrt_client(original_client)
  expect_identical(result, original_client)
})

test_that("cuda plugin can be downloaded", {
  skip_if(!is_cuda(), "Not running on CUDA platform")
  expect_no_error(pjrt_plugin("cuda"))
  expect_true(plugin_is_downloaded("cuda"))
})

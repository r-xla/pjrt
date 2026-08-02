test_that("cuda plugin can be downloaded", {
  skip_if(!is_cuda(), "Not running on CUDA platform")
  expect_no_error(pjrt_plugin("cuda"))
  expect_true(plugins_downloaded("cuda"))
})

test_that("mps plugin can be downloaded", {
  skip_if(!is_mps(), "Not running on MPS platform")
  expect_no_error(pjrt_plugin("mps"))
  expect_true(plugins_downloaded("mps"))
})

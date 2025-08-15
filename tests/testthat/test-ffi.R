test_that("can load the ffi extension", {
  expect_true(test_get_extension(pjrt_plugin(Sys.getenv("PJRT_PLATFORM", "cpu"))))
})
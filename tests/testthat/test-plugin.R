test_that("creation works", {
  expect_class(pjrt_plugin("cpu"), "PJRTPlugin")
})

test_that("attributes work", {
  attrs <- plugin_attributes(pjrt_plugin("cpu"))
  expect_subset(
    c("xla_version", "stablehlo_current_version", "stablehlo_minimum_version"),
    names(attrs)
  )
})

test_that("print works", {
  expect_snapshot(pjrt_plugin("cpu"))
})


test_that("invalid plugin name does not create folder", {
  expect_error(pjrt_plugin("abc"), "Invalid platform")
  expect_false(dir.exists(file.path(tools::R_user_dir("pjrt", which = "cache"), "invalid")))
})

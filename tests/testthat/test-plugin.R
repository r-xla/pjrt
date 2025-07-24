test_that("creation works", {
  skip_on_windows()
  expect_class(pjrt_plugin("cpu"), "PJRTPlugin")
})

test_that("attributes work", {
  attrs <- plugin_attributes(pjrt_plugin("cpu"))
  expect_subset(
    c("xla_version", "stablehlo_current_version", "stablehlo_minimum_version"),
    names(attrs)
  )
})

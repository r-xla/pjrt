test_that("attributes work", {
  attrs <- pjrt_plugin_attributes("cpu")
  expect_subset(
    c("xla_version", "stablehlo_current_version", "stablehlo_minimum_version"),
    names(attrs)
  )
})

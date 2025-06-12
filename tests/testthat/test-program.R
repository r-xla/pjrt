test_that("load a test program", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")

  expect_snapshot(print(program))
})



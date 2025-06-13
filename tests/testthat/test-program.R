test_that("load a test program", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- program_load(path, format = "hlo")

  expect_snapshot(print(program))
})

test_that("can load MLIR program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- program_load(path, format = "mlir")

  expect_snapshot(print(program))
})

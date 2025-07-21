test_that("load a test program", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")

  expect_snapshot(print(program))
})

test_that("can load MLIR program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")

  expect_snapshot(print(program))
})

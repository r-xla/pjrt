test_that("load a test program", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")

  expect_snapshot(print(program))
})

test_that("can load MLIR program", {
  # Windows hash a slightly different code size
  skip_on_os("windows")

  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")

  expect_snapshot(print(program))
})

test_that("error message", {
  on.exit(unlink(path))
  path <- tempfile(fileext = ".mlir")
  writeLines("foo", path)
  expect_error(pjrt_program(path), "You passed a file path to src")
})

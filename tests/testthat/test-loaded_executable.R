test_that("arguments must be unnamed", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")
  executable <- pjrt_compile(program)
  buf <- pjrt_buffer(1)
  expect_error(pjrt_execute(executable, a = buf, "Expected unnamed arguments"))
})

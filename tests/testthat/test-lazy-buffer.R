test_that("lazy execute returns lazy buffer and can materialize", {
  skip_if_metal("only supports MLIR programs")
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  lazy <- pjrt_execute_lazy(executable)
  expect_s3_class(lazy, "PJRTLazyBuffer")
  expect_true(is.logical(lazy_buffer_ready(lazy)))

  buf <- pjrt_lazy_buffer_materialize(lazy)
  expect_s3_class(buf, "PJRTBuffer")
  expect_equal(as_array(buf), 3)

  printed <- paste(capture.output(print(lazy)), collapse = "\n")
  expect_match(printed, "PJRTLazyBuffer")
})

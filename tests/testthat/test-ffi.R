skip_if_metal("FFI extension not available on metal")

test_that("can load the ffi extension", {
  platform <- Sys.getenv("PJRT_PLATFORM", "cpu")
  plugin <- pjrt_plugin(platform)
  expect_true(test_get_extension(plugin, ifelse(platform != "cuda", "host", "cuda")))
})

test_that("can use registered funs", {
  src <- r"(
func.func @main(
  %x: tensor<2x2xf32>
) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  stablehlo.custom_call @my_custom_call() {
  call_target_name = "test_handler",
  has_side_effect = true,
  api_version = 4 : i32
} : () -> ()
  "func.return"(%0): (tensor<2x2xf32>) -> ()
}
)"

  program <- pjrt_program(src)
  program <- pjrt_compile(program)
  out <- pjrt_execute(program, pjrt_buffer(matrix(1, nrow = 2, ncol = 2)))
  expect_no_error({
    x <- as_array(out)
  })
})

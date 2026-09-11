constant_folding_src <- "
func.func @main() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %1 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}
"

test_that("pjrt_optimize() folds constants", {
  skip_if_no_stablehlo_opt()

  optimized <- pjrt_optimize(constant_folding_src)

  expect_class(optimized, "PJRTProgram")
  code <- program_code(optimized)
  expect_match(code, "dense<[4.000000e+00, 6.000000e+00]>", fixed = TRUE)
  expect_false(grepl("stablehlo.add", code, fixed = TRUE))
})

test_that("pjrt_optimize() accepts a PJRTProgram and the result still runs", {
  skip_if_no_stablehlo_opt()

  program <- pjrt_program(src = constant_folding_src)
  optimized <- pjrt_optimize(program)

  expect_false(grepl("stablehlo.add", program_code(optimized), fixed = TRUE))

  executable <- pjrt_compile(optimized)
  expect_equal(as_array(pjrt_execute(executable)), array(c(4, 6)))
})

test_that("pjrt_optimize() runs the requested passes only", {
  skip_if_no_stablehlo_opt()

  # `stablehlo-legalize-to-vhlo` does not fold anything, so the add survives.
  optimized <- pjrt_optimize(
    constant_folding_src,
    passes = "stablehlo-legalize-to-vhlo"
  )
  expect_match(program_code(optimized), "vhlo.add", fixed = TRUE)

  # A leading `--` is optional.
  expect_equal(
    program_code(pjrt_optimize(constant_folding_src, "stablehlo-aggressive-folder")),
    program_code(pjrt_optimize(constant_folding_src, "--stablehlo-aggressive-folder"))
  )
})

test_that("pjrt_optimize() errors on invalid input", {
  skip_if_no_stablehlo_opt()

  expect_error(pjrt_optimize("this is not MLIR"), "stablehlo-opt.*failed")
  expect_error(
    pjrt_optimize(constant_folding_src, passes = "not-a-pass"),
    "stablehlo-opt.*failed"
  )
})

test_that("pjrt_optimize() rejects HLO programs", {
  skip_if_no_stablehlo_opt()

  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")

  expect_error(pjrt_optimize(program), "mlir")
})

mlp_types <- function(batch) {
  c(
    "tensor<3x4xf32>",
    "tensor<4xf32>",
    "tensor<4x2xf32>",
    "tensor<2xf32>",
    sprintf("tensor<%dx3xf32>", batch)
  )
}

test_that("pjrt_refine_shapes() turns a shape-polymorphic export into a runnable program", {
  skip_if_no_stablehlo_opt()

  # One pass over a real JAX export is enough: it shows we can invoke the
  # binary, hand it argument types, and get back a program XLA accepts and
  # runs. What the passes do to the IR on the way -- erasing the shape
  # assertion, folding `dynamic_broadcast_in_dim`, which `tensor<?x...>`
  # becomes what -- is stablehlo-opt's business, and asserting it here would
  # only pin us to its current output.
  path <- system.file("programs/jax-mlp-dynamic.mlir", package = "pjrt")
  refined <- pjrt_refine_shapes(pjrt_program(path = path), mlp_types(5L))
  expect_class(refined, "PJRTProgram")

  w1 <- pjrt_buffer(matrix(0, nrow = 3, ncol = 4), dtype = "f32")
  b1 <- pjrt_buffer(rep(0, 4), dtype = "f32")
  w2 <- pjrt_buffer(matrix(0, nrow = 4, ncol = 2), dtype = "f32")
  b2 <- pjrt_buffer(c(1, -1), dtype = "f32")
  x <- pjrt_buffer(matrix(1, nrow = 5, ncol = 3), dtype = "f32")

  executable <- pjrt_compile(refined)
  out <- as_array(pjrt_execute(executable, w1, b1, w2, b2, x))

  # tanh(0) = 0, so the result is just the output bias, broadcast over the batch.
  expect_equal(out, matrix(c(rep(1, 5), rep(-1, 5)), nrow = 5))
})

test_that("pjrt_refine_shapes() surfaces a stablehlo-opt failure as an R error", {
  skip_if_no_stablehlo_opt()

  # Our error path, not the binary's diagnostics: a non-zero exit has to become
  # an R condition rather than a silent pass-through.
  path <- system.file("programs/jax-mlp-dynamic.mlir", package = "pjrt")
  program <- pjrt_program(path = path)

  expect_error(
    pjrt_refine_shapes(program, mlp_types(5L)[-1L]),
    "stablehlo-opt.*failed"
  )
})

test_that("stablehlo_opt_passes() lists the StableHLO passes", {
  skip_if_no_stablehlo_opt()

  passes <- stablehlo_opt_passes()

  expect_character(passes, min.len = 1L, unique = TRUE, any.missing = FALSE)
  expect_true(all(startsWith(passes, "stablehlo-")))
  expect_true("stablehlo-target-independent-optimization" %in% passes)
})

test_that("stablehlo_opt_bin() respects PJRT_STABLEHLO_OPT_PATH", {
  bin <- withr::local_tempfile()
  file.create(bin)

  withr::local_envvar(PJRT_STABLEHLO_OPT_PATH = bin)
  expect_equal(stablehlo_opt_bin(), bin)
  expect_true(stablehlo_opt_available())

  withr::local_envvar(PJRT_STABLEHLO_OPT_PATH = file.path(bin, "nope"))
  expect_error(stablehlo_opt_bin(), "non-existing file")
})

test_that("stablehlo_opt_bin(install = FALSE) does not download", {
  withr::local_envvar(PJRT_STABLEHLO_OPT_PATH = "")
  skip_if(stablehlo_opt_available(), "stablehlo-opt is already downloaded")

  expect_error(stablehlo_opt_bin(install = FALSE), "not downloaded yet")
})

test_that("stablehlo_opt_url() points at an existing build", {
  withr::local_envvar(PJRT_STABLEHLO_OPT_URL = "", PJRT_STABLEHLO_OPT_VERSION = "")

  url <- stablehlo_opt_url()
  expect_match(url, "^https://github.com/r-xla/pjrt-builds/releases/download/stablehlo/")
  expect_match(url, "stablehlo-opt-main-(linux-x86_64|mac-arm64)\\.tar\\.gz$|stablehlo-opt-main-windows-x86_64\\.zip$")

  withr::local_envvar(PJRT_STABLEHLO_OPT_VERSION = "v1.12.1")
  expect_match(stablehlo_opt_url(), "stablehlo-opt-v1.12.1-", fixed = TRUE)

  withr::local_envvar(PJRT_STABLEHLO_OPT_URL = "https://example.com/x.tar.gz")
  expect_equal(stablehlo_opt_url(), "https://example.com/x.tar.gz")
})

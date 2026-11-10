test_that("compile program with one input", {
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")
  platform <- Sys.getenv("PJRT_PLATFORM", "cpu")

  plugin <- pjrt_plugin(platform)
  client <- plugin_client_create(plugin, platform)

  check_client_device(client)

  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  data <- 3.0
  scalar_buffer <- pjrt_scalar(data)

  result <- pjrt_execute(executable, scalar_buffer)
  r_res <- as_array(result)

  expect_equal(r_res, 9)
})

describe("pjrt_compile", {
  it("rejects an HLO program that declares input/output aliasing", {
    # test_hlo_alias.pb is test_hlo.pb with `input_output_alias` set (parameter
    # 0 may alias the output), encoded with protoc from inst/proto. XLA compiles
    # it; pjrt then rejects it, after the executable exists, and must free it.
    # That the executable is freed is not observable from R: this only covers
    # the error path.
    path <- system.file("programs/test_hlo_alias.pb", package = "pjrt")
    program <- pjrt_program(path = path, format = "hlo")
    expect_error(pjrt_compile(program), "aliasing is not supported for HLO")
  })

  it("applies xla_options to that compilation only", {
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    dir <- withr::local_tempdir()
    pjrt_compile(program, xla_options = list(xla_dump_to = dir, xla_dump_hlo_as_text = TRUE))
    dumped <- list.files(dir)
    expect_true(any(endsWith(dumped, "after_optimizations.txt")))

    pjrt_compile(program)
    expect_identical(list.files(dir), dumped)
  })

  it("passes each type of xla_options value through to XLA", {
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    executable <- pjrt_compile(
      program,
      xla_options = list(
        xla_cpu_enable_fast_math = FALSE,
        xla_backend_optimization_level = 2L,
        xla_cpu_parallel_codegen_split_count = "16",
        xla_dump_to = ""
      )
    )
    expect_equal(as_array(pjrt_execute(executable, pjrt_scalar(3))), 6)
  })

  it("reports an option XLA does not know", {
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    expect_error(
      pjrt_compile(program, xla_options = list(xla_no_such_option = TRUE)),
      "xla_no_such_option"
    )
  })

  it("rejects xla_options that are not named scalars", {
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    expect_error(pjrt_compile(program, xla_options = list(TRUE)), "names")
    expect_error(pjrt_compile(program, xla_options = setNames(list(TRUE, FALSE), c("a", "a"))), "unique")
    expect_error(pjrt_compile(program, xla_options = list(a = c(1L, 2L))), "scalar")
    expect_error(pjrt_compile(program, xla_options = list(a = NA)), "scalar")
    expect_error(pjrt_compile(program, xla_options = list(a = list(1L))), "scalar")
  })

  it("adds map-valued xla_options to the plugin's defaults", {
    skip_if(!is_cpu())
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    plain <- non_default_debug_options(program)
    with_map <- non_default_debug_options(
      program,
      list(xla_backend_extra_options = "xla_cpu_use_multi_output_fusion,xla_pjrt_test=1")
    )
    # Every other field, such as xla_backend_optimization_level, is still at the plain
    # compilation's value: a DebugOptions built from scratch would differ in hundreds.
    expect_setequal(
      setdiff(with_map, plain),
      c(
        'xla_backend_extra_options: { key: "xla_cpu_use_multi_output_fusion" value: "" }',
        'xla_backend_extra_options: { key: "xla_pjrt_test" value: "1" }'
      )
    )
    expect_true(all(plain %in% with_map))
    expect_identical(non_default_debug_options(program), plain)
  })

  it("compiles correctly with scalar and map-valued xla_options together", {
    skip_if(!is_cpu())
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    executable <- pjrt_compile(
      program,
      xla_options = list(
        xla_backend_extra_options = "xla_cpu_use_multi_output_fusion",
        xla_cpu_enable_fast_math = FALSE
      )
    )
    expect_equal(as_array(pjrt_execute(executable, pjrt_scalar(3))), 6)
  })

  it("keeps the map entries XLA_FLAGS sets", {
    skip_if(!is_cpu())
    skip_if_not_installed("callr")
    lines <- callr::r(
      function(path, helper) {
        source(helper)
        program <- pjrt::pjrt_program(path = path)
        non_default_debug_options(program, list(xla_backend_extra_options = "xla_pjrt_from_call"))
      },
      args = list(
        path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"),
        helper = test_path("helper-debug-options.R")
      ),
      env = c(callr::rcmd_safe_env(), XLA_FLAGS = "--xla_backend_extra_options=xla_pjrt_from_flags")
    )
    expect_contains(
      lines,
      c(
        'xla_backend_extra_options: { key: "xla_pjrt_from_flags" value: "" }',
        'xla_backend_extra_options: { key: "xla_pjrt_from_call" value: "" }'
      )
    )
  })

  it("rejects a map-valued option that is not in XLA_FLAGS syntax", {
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    for (value in list("", "a,,b", "a,", "=1", TRUE, 1L)) {
      expect_error(
        pjrt_compile(program, xla_options = list(xla_backend_extra_options = value)),
        "sets a map"
      )
    }
  })

  it("explains that the CUDA plugin cannot set map-valued options", {
    skip_if(!is_cuda())
    program <- pjrt_program(path = system.file("programs/jax-stablehlo.mlir", package = "pjrt"))
    expect_error(
      pjrt_compile(program, xla_options = list(xla_backend_extra_options = "xla_pjrt_test")),
      "xla_backend_extra_options.*XLA_FLAGS"
    )
  })
})

test_that("compile program with multiple inputs", {
  path <- system.file("programs/stablehlo.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")

  platform <- Sys.getenv("PJRT_PLATFORM", "cpu")
  plugin <- pjrt_plugin(platform)
  client <- plugin_client_create(plugin, platform)
  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  image <- array(0, dim = c(28, 28))
  weights <- array(0, dim = c(784, 10))
  bias <- array(0, dim = c(1, 10))

  image_buffer <- pjrt_buffer(image)
  weights_buffer <- pjrt_buffer(weights)
  bias_buffer <- pjrt_buffer(bias)

  result <- pjrt_execute(
    executable,
    image_buffer,
    weights_buffer,
    bias_buffer
  )
  r_res <- as_array(result)

  expect_equal(r_res, matrix(0, ncol = 10, nrow = 1))
})

test_that("can execute mlir program", {
  path <- system.file("programs/jax-stablehlo.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")

  executable <- pjrt_compile(program)

  expect_true(inherits(executable, "PJRTLoadedExecutable"))

  client <- pjrt_client()
  check_client_device(client)

  data <- 3.0
  scalar_buffer <- pjrt_scalar(data)

  result <- pjrt_execute(executable, scalar_buffer)
  r_res <- as_array(result)

  expect_equal(r_res, 6)
})

test_that("can use more than one client", {
  skip_if(!is_cuda())

  device <- "cuda"

  pjrt_buffer(1, device = device)

  expect_permutation(c(device, "cpu"), names(the[["clients"]]))
  expect_permutation(c(device, "cpu"), names(the[["plugins"]]))

  # not they are loaded and global env 'the' is not changed
  pjrt_buffer(1, device = "cpu")
  pjrt_buffer(1, device = device)

  expect_permutation(c(device, "cpu"), names(the[["clients"]]))
  expect_permutation(c(device, "cpu"), names(the[["plugins"]]))
})

test_that("platform", {
  expect_equal(platform(pjrt_client("cpu")), "cpu")
  skip_if(!is_cuda())
  expect_equal(platform(pjrt_client("cuda")), "cuda")
})

test_that("printer", {
  expect_snapshot(pjrt_client("cpu"))
})

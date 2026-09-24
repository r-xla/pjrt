scale_src <- r"(
template <typename T>
__global__ void scale(const T *x, T *out, T a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a * x[i];
}
)"

# A program with one `pjrt_cuda_kernel` call on 1-d operands of `dtype`.
cuda_kernel_program <- function(attrs, n_in, n_out, len, dtype = "f32") {
  type <- sprintf("tensor<%dx%s>", len, dtype)
  args <- sprintf("%%x%d: %s", seq_len(n_in) - 1L, type)
  operands <- sprintf("%%x%d", seq_len(n_in) - 1L)
  config <- vapply(
    names(attrs),
    function(nm) {
      v <- attrs[[nm]]
      if (is.character(v)) sprintf("%s = \"%s\"", nm, v) else sprintf("%s = %d : i32", nm, v)
    },
    character(1L)
  )
  results <- sprintf("%%r#%d", seq_len(n_out) - 1L)
  out_types <- rep(type, n_out)
  sprintf(
    r"(func.func @main(%s) -> (%s) {
  %%r:%d = stablehlo.custom_call @pjrt_cuda_kernel(%s) {
    call_target_name = "pjrt_cuda_kernel",
    api_version = 4 : i32,
    backend_config = {%s},
    operand_layouts = [%s],
    result_layouts = [%s]
  } : (%s) -> (%s)
  "func.return"(%s) : (%s) -> ()
})",
    paste(args, collapse = ", "),
    paste(out_types, collapse = ", "),
    n_out,
    paste(operands, collapse = ", "),
    paste(config, collapse = ", "),
    paste(rep("dense<0> : tensor<1xindex>", n_in), collapse = ", "),
    paste(rep("dense<0> : tensor<1xindex>", n_out), collapse = ", "),
    paste(rep(type, n_in), collapse = ", "),
    paste(out_types, collapse = ", "),
    paste(results, collapse = ", "),
    paste(out_types, collapse = ", ")
  )
}

run_cuda_kernel <- function(attrs, inputs, n_out = 1L, dtype = "f32") {
  src <- cuda_kernel_program(attrs, length(inputs), n_out, length(inputs[[1L]]), dtype)
  exec <- pjrt_compile(pjrt_program(src))
  bufs <- lapply(inputs, pjrt_buffer, dtype = dtype)
  outs <- do.call(pjrt_execute, c(list(exec), bufs))
  if (n_out == 1L) as.vector(as_array(outs)) else lapply(outs, function(o) as.vector(as_array(o)))
}

describe("pjrt_cuda_module", {
  it("identifies a module by its contents", {
    a <- pjrt_cuda_module(scale_src, kernels = "scale<float>")
    b <- pjrt_cuda_module(scale_src, kernels = "scale<float>")
    expect_s3_class(a, "PJRTCudaModule")
    expect_identical(a$id, b$id)
    expect_match(a$id, "^[0-9a-f]{16}$")
    expect_false(identical(a$id, pjrt_cuda_module(scale_src, kernels = "scale<double>")$id))
    expect_false(identical(a$id, pjrt_cuda_module(scale_src, kernels = "scale<float>", options = "-DX")$id))
  })

  it("reads source and images from files", {
    cu <- withr::local_tempfile(fileext = ".cu")
    writeLines(scale_src, cu)
    expect_identical(pjrt_cuda_module(file = cu)$id, pjrt_cuda_module(scale_src)$id)

    ptx <- withr::local_tempfile(fileext = ".ptx")
    writeLines(".version 8.0", ptx)
    mod <- pjrt_cuda_module(file = ptx)
    # PTX is handed to the driver as a C string
    expect_equal(mod$image[[length(mod$image)]], as.raw(0L))
    expect_identical(mod$code, "")
  })

  it("prints", {
    expect_output(print(pjrt_cuda_module(scale_src, kernels = "scale<float>")), "scale<float>")
  })

  it("validates its arguments", {
    expect_error(pjrt_cuda_module(), "exactly one")
    expect_error(pjrt_cuda_module("x", file = "y"), "exactly one")
    expect_error(pjrt_cuda_module(1), "character")
    expect_error(pjrt_cuda_module("x", kernels = NA_character_))
  })
})

describe("pjrt_cuda_launch_attrs", {
  it("pads the launch dimensions and encodes the scalars", {
    mod <- pjrt_cuda_module(scale_src, kernels = "scale<float>")
    attrs <- pjrt_cuda_launch_attrs(
      mod,
      "scale<float>",
      grid = 2L,
      block = c(32, 4),
      scalars = list(pjrt_cuda_scalar(1, "f32"), 10L)
    )
    expect_identical(attrs$module, mod$id)
    expect_identical(attrs$kernel, "scale<float>")
    expect_identical(c(attrs$grid_x, attrs$grid_y, attrs$grid_z), c(2L, 1L, 1L))
    expect_identical(c(attrs$block_x, attrs$block_y, attrs$block_z), c(32L, 4L, 1L))
    expect_identical(attrs$shared_mem, 0L)
    expect_identical(attrs$scalars, "0000803f,0a000000")
  })

  it("allows an empty grid", {
    mod <- pjrt_cuda_module(scale_src)
    expect_identical(pjrt_cuda_launch_attrs(mod, "k", grid = 0L, block = 1L)$grid_x, 0L)
  })

  it("rejects malformed launch dimensions", {
    mod <- pjrt_cuda_module(scale_src)
    expect_error(pjrt_cuda_launch_attrs(mod, "k", grid = 1:4, block = 1L), "grid")
    expect_error(pjrt_cuda_launch_attrs(mod, "k", grid = 1L, block = 0L), "block")
    expect_error(pjrt_cuda_launch_attrs(mod, "k", grid = 1.5, block = 1L), "grid")
  })
})

describe("pjrt_cuda_scalar", {
  it("maps R types to C types like .C() does", {
    expect_identical(encode_cuda_scalar(10L), "0a000000")
    expect_identical(encode_cuda_scalar(1), "000000000000f03f")
    expect_identical(encode_cuda_scalar(TRUE), "01")
  })

  it("encodes explicitly typed scalars", {
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(-1, "i8")), "ff")
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(258, "ui16")), "0201")
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(2^32 - 1, "ui32")), "ffffffff")
    expect_no_warning(expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(2^31, "ui32")), "00000080"))
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(-2^31, "i32")), "00000080")
    expect_identical(
      encode_cuda_scalar(pjrt_cuda_scalar(bit64::as.integer64("9007199254740993"), "i64")),
      "0100000000002000"
    )
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(2^40, "i64")), "0000000000010000")
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(-2, "i64")), "feffffffffffffff")
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(1, "f32")), "0000803f")
  })

  it("rejects values the C type cannot hold", {
    expect_error(encode_cuda_scalar(pjrt_cuda_scalar(128, "i8")), "not representable")
    expect_error(encode_cuda_scalar(pjrt_cuda_scalar(1.5, "i32")), "not representable")
    expect_error(encode_cuda_scalar(pjrt_cuda_scalar(-1, "ui32")), "not representable")
    expect_error(encode_cuda_scalar(pjrt_cuda_scalar(2^60, "i64")), "integer64")
    expect_error(encode_cuda_scalar(pjrt_cuda_scalar(1e39, "f32")), "not representable")
    expect_identical(encode_cuda_scalar(pjrt_cuda_scalar(Inf, "f32")), "0000807f")
    expect_error(encode_cuda_scalar("a"), "Cannot pass")
    expect_error(pjrt_cuda_scalar(1, "f16"))
  })

  it("prints", {
    expect_output(print(pjrt_cuda_scalar(3, "i64")), "i64")
  })
})

describe("the pjrt_cuda_kernel custom call", {
  mod <- pjrt_cuda_module(scale_src, kernels = c("scale<float>", "scale<double>"))
  launch <- function(kernel = "scale<float>", scalars = list(pjrt_cuda_scalar(2, "f32"), 5L), ...) {
    pjrt_cuda_launch_attrs(mod, kernel, grid = 1L, block = 32L, scalars = scalars, ...)
  }

  it("is rejected on the CPU", {
    skip_if(!is_cpu())
    expect_error(run_cuda_kernel(launch(), list(1:5)), "only runs on CUDA")
  })

  it("launches a kernel", {
    skip_if(!is_cuda())
    expect_equal(run_cuda_kernel(launch(), list(1:5)), 2 * (1:5))
  })

  it("launches each instantiation of a template", {
    skip_if(!is_cuda())
    attrs <- launch("scale<double>", scalars = list(0.5, 5L))
    expect_equal(run_cuda_kernel(attrs, list(1:5), dtype = "f64"), (1:5) / 2)
  })

  it("launches kernels with several outputs, and with an empty grid", {
    skip_if(!is_cuda())
    two <- pjrt_cuda_module(
      r"(
      extern "C" __global__ void sum_diff(const float *a, const float *b,
                                          float *s, float *d, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { s[i] = a[i] + b[i]; d[i] = a[i] - b[i]; }
      })"
    )
    out <- run_cuda_kernel(
      pjrt_cuda_launch_attrs(two, "sum_diff", grid = c(1L, 1L), block = 8L, scalars = list(3L)),
      list(c(1, 2, 3), c(3, 2, 1)),
      n_out = 2L
    )
    expect_equal(out, list(c(4, 4, 4), c(-2, 0, 2)))
    expect_no_error(run_cuda_kernel(
      pjrt_cuda_launch_attrs(two, "sum_diff", grid = 0L, block = 8L, scalars = list(3L)),
      list(c(1, 2, 3), c(3, 2, 1)),
      n_out = 2L
    ))
  })

  it("checks the launch against the kernel's signature", {
    skip_if(!is_cuda())
    expect_error(run_cuda_kernel(launch(scalars = list(5L)), list(1:5)), "takes 4 parameters")
    # a double where the kernel wants a float
    expect_error(
      run_cuda_kernel(launch(scalars = list(2, 5L)), list(1:5)),
      "Parameter 3 .* is 4 bytes, but a scalar of 8 bytes"
    )
  })

  it("reports missing kernels and compilation errors", {
    skip_if(!is_cuda())
    expect_error(run_cuda_kernel(launch("scale<int>"), list(1:5)), "must be listed in the `kernels`")
    broken <- pjrt_cuda_module("__global__ void f(float *x) { x[0] = y; }", kernels = "f")
    attrs <- pjrt_cuda_launch_attrs(broken, "f", grid = 1L, block = 1L)
    expect_error(run_cuda_kernel(attrs, list(1)), "identifier \"y\" is undefined")
  })

  it("caches compiled modules on disk", {
    skip_if(!is_cuda())
    dir <- withr::local_tempdir()
    withr::local_envvar(PJRT_CUDA_CACHE = dir)
    # registered before, with the default cache; launching it anew picks up
    # the directory set now
    fresh <- pjrt_cuda_module(scale_src, kernels = "scale<float>", options = "-DCACHE_TEST")
    withr::local_envvar(PJRT_CUDA_CACHE = file.path(dir, "later"))
    attrs <- pjrt_cuda_launch_attrs(
      fresh,
      "scale<float>",
      grid = 1L,
      block = 32L,
      scalars = list(pjrt_cuda_scalar(3, "f32"), 2L)
    )
    expect_equal(run_cuda_kernel(attrs, list(c(1, 2))), c(3, 6))
    expect_length(list.files(file.path(dir, "later"), pattern = "\\.cubin$"), 1L)
  })

  it("recompiles when a cache entry is corrupt", {
    skip_if(!is_cuda())
    dir <- withr::local_tempdir()
    program <- cuda_kernel_program
    environment(program) <- globalenv()
    # each run is a fresh session, which has to go to the disk cache
    run_fresh <- function() {
      callr::r(
        function(src, dir, program) {
          Sys.setenv(PJRT_CUDA_CACHE = dir)
          library(pjrt)
          mod <- pjrt_cuda_module(src, kernels = "scale<float>")
          attrs <- pjrt_cuda_launch_attrs(
            mod,
            "scale<float>",
            grid = 1L,
            block = 32L,
            scalars = list(pjrt_cuda_scalar(2, "f32"), 2L)
          )
          exec <- pjrt_compile(pjrt_program(program(attrs, 1L, 1L, 2L)), device = "cuda")
          x <- pjrt_buffer(c(1, 2), dtype = "f32", device = "cuda")
          as.vector(as_array(pjrt_execute(exec, x)))
        },
        list(scale_src, dir, program)
      )
    }
    expect_equal(run_fresh(), c(2, 4))
    entry <- list.files(dir, pattern = "\\.cubin$", full.names = TRUE)
    expect_length(entry, 1L)
    size <- file.size(entry)

    # keep the header of kernel names, replace the cubin by garbage
    bytes <- readBin(entry, "raw", size)
    header_end <- which(bytes == as.raw(10L))[[2L]]
    writeBin(c(bytes[seq_len(header_end)], as.raw(rep(1L, 64L))), entry)

    expect_equal(run_fresh(), c(2, 4))
    expect_identical(file.size(entry), size)
  })
})

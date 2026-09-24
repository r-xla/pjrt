#' @title CUDA Kernel Module
#' @description
#' Creates a module of CUDA kernels that programs can launch through the
#' built-in `pjrt_cuda_kernel` custom call, without writing an XLA FFI
#' handler.
#'
#' A module is either CUDA C++ source, or a prebuilt image. Source is
#' compiled with NVRTC -- which ships with the CUDA R package the CUDA plugin
#' uses, so no CUDA toolkit is needed -- the first time one of its kernels
#' runs on a device, for exactly that device's GPU architecture. The result
#' is cached on disk (see [`pjrt_cuda_cache_dir()`]), so a module is compiled
#' once per machine rather than once per session. A package can instead ship
#' its modules precompiled, see `package` and [`pjrt_cuda_build_kernels()`].
#'
#' Creating a module needs neither a GPU nor the CUDA plugin; it only
#' records the module so that the programs referring to it can find it.
#' Compilation errors therefore surface the first time a kernel runs, with
#' NVRTC's log -- with anvl, when the first result is read.
#'
#' A module is identified, and cached, by its own source, `options` and
#' `kernels`. A header it `#include`s is not part of that, so keep modules
#' self-contained: a changed header would not invalidate the cache.
#'
#' @section Kernel parameters:
#' A kernel is launched with the custom call's operands and results as
#' device pointers, in that order, followed by its scalar arguments:
#'
#' ```
#' __global__ void kernel(const T *in_1, ..., T *out_1, ..., <scalars>)
#' ```
#'
#' Buffers are laid out as the custom call's `operand_layouts` and
#' `result_layouts` say (row-major by default). See
#' [`pjrt_cuda_launch_attrs()`] for how scalars map to C types.
#'
#' @param code (`character()` | `NULL`)\cr
#'   CUDA C++ source, pasted together with newlines.
#' @param file (`character(1)` | `NULL`)\cr
#'   Alternatively to `code`, a file: CUDA source (`.cu`, `.cuh`) or a
#'   prebuilt image (`.ptx`, `.cubin`, `.fatbin`), e.g. one compiled ahead
#'   of time with `nvcc -fatbin`.
#' @param kernels (`character()`)\cr
#'   The name expressions of the kernels to instantiate, needed for
#'   templates and other C++ kernels, e.g. `c("scale<float>",
#'   "scale<double>")`. Kernels declared `extern "C"` need not be listed.
#'   Ignored for images.
#' @param options (`character()`)\cr
#'   Additional NVRTC options, e.g. `"-DBLOCK=256"` or `"--use_fast_math"`.
#'   The GPU architecture is set automatically.
#' @param package (`character(1)` | `NULL`)\cr
#'   The package shipping the module, when a package creates it in its
#'   `.onLoad()`. Its prebuilt kernels (see [`pjrt_cuda_build_kernels()`])
#'   are then used in place of compiling the module on the user's machine,
#'   whenever they cover the GPU.
#' @return `PJRTCudaModule`
#' @seealso [pjrt_cuda_launch_attrs()]
#' @examples
#' mod <- pjrt_cuda_module(r"(
#' extern "C" __global__ void add_one(const float *x, float *out, int n) {
#'   int i = blockIdx.x * blockDim.x + threadIdx.x;
#'   if (i < n) out[i] = x[i] + 1.0f;
#' })")
#' mod
#' @export
pjrt_cuda_module <- function(
  code = NULL,
  file = NULL,
  kernels = character(),
  options = character(),
  package = NULL
) {
  checkmate::assert_character(kernels, any.missing = FALSE)
  checkmate::assert_character(options, any.missing = FALSE)
  checkmate::assert_string(package, null.ok = TRUE)
  if (is.null(code) == is.null(file)) {
    cli_abort("Pass exactly one of {.arg code} and {.arg file}.")
  }
  image <- raw()
  filename <- "kernel.cu"
  if (!is.null(file)) {
    checkmate::assert_file_exists(file)
    filename <- basename(file)
    ext <- tolower(tools::file_ext(file))
    if (ext %in% c("ptx", "cubin", "fatbin")) {
      image <- readBin(file, "raw", file.size(file))
      # the driver expects PTX to be NUL-terminated
      if (ext == "ptx") image <- c(image, as.raw(0L))
    } else {
      code <- readLines(file, warn = FALSE)
    }
  }
  if (!length(image)) {
    checkmate::assert_character(code, min.len = 1L, any.missing = FALSE)
    code <- paste(code, collapse = "\n")
  }

  module <- structure(
    list(
      code = if (length(image)) "" else code,
      filename = filename,
      kernels = kernels,
      options = options,
      image = image,
      id = cuda_module_register(code %||% "", filename, options, kernels, image)
    ),
    class = "PJRTCudaModule"
  )
  if (!is.null(package) && !length(image)) {
    cuda_record_module(module, package)
  }
  module
}

cuda_module_register <- function(code, filename, options, kernels, image) {
  impl_cuda_module_register(code, filename, options, kernels, image, cuda_cache_dir_created())
}

cuda_cache_dir_created <- function() {
  dir <- pjrt_cuda_cache_dir()
  if (nzchar(dir) && !dir.exists(dir)) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  }
  dir
}

#' @export
print.PJRTCudaModule <- function(x, ...) {
  what <- if (length(x$image)) "image" else "source"
  cat(sprintf("<PJRTCudaModule %s> (%s, %s)\n", x$id, x$filename, what))
  if (length(x$kernels)) {
    cat("Kernels:", paste(x$kernels, collapse = ", "), "\n")
  }
  invisible(x)
}

#' @title CUDA Kernel Cache Directory
#' @description
#' Where CUDA modules compiled from source are cached, one file per module
#' and GPU architecture. Controlled by the `PJRT_CUDA_CACHE` environment
#' variable; set it to `""` to disable the cache.
#' @return (`character(1)`)\cr
#'   The directory, or `""` if caching is disabled.
#' @export
pjrt_cuda_cache_dir <- function() {
  Sys.getenv("PJRT_CUDA_CACHE", file.path(tools::R_user_dir("pjrt", "cache"), "cuda-kernels"))
}

#' @title Launch Attributes for a CUDA Kernel
#' @description
#' Builds the attributes of a `stablehlo.custom_call @pjrt_cuda_kernel`
#' that launches `kernel` from `module`, and makes sure `module` is known to
#' this session.
#'
#' This is the building block for front ends that emit StableHLO (such as
#' anvl's `nv_cuda_kernel()`); the attributes go into the call's
#' `backend_config`, with the integers as `i32`.
#'
#' @section Scalar arguments:
#' Each scalar is passed by value, with the C type following from its R
#' type as in R's own `.C()` interface: an integer is an `int`, a double a
#' `double`, and a logical a `bool`. For any other C type, wrap the value in
#' [`pjrt_cuda_scalar()`].
#'
#' @param module (`PJRTCudaModule`)\cr
#'   The module, from [`pjrt_cuda_module()`].
#' @param kernel (`character(1)`)\cr
#'   The kernel: its name, or one of the module's `kernels` expressions.
#' @param grid,block (`integer()`)\cr
#'   Number of blocks, and threads per block, in up to three dimensions. A
#'   grid with a zero dimension launches nothing.
#' @param shared_mem (`integer(1)`)\cr
#'   Bytes of dynamic shared memory per block.
#' @param scalars (`list()`)\cr
#'   Scalar arguments, passed after the buffers.
#' @return Named `list()` of attribute values.
#' @examples
#' mod <- pjrt_cuda_module(r"(
#' extern "C" __global__ void scale(const float *x, float *out, float a, int n) {
#'   int i = blockIdx.x * blockDim.x + threadIdx.x;
#'   if (i < n) out[i] = a * x[i];
#' })")
#' str(pjrt_cuda_launch_attrs(
#'   mod, "scale",
#'   grid = 4L, block = 256L,
#'   scalars = list(pjrt_cuda_scalar(2, "f32"), 1000L)
#' ))
#' @export
pjrt_cuda_launch_attrs <- function(module, kernel, grid, block, shared_mem = 0L, scalars = list()) {
  checkmate::assert_class(module, "PJRTCudaModule")
  checkmate::assert_string(kernel)
  checkmate::assert_count(shared_mem)
  checkmate::assert_list(scalars)
  # a module created in another session (e.g. at a package's build time) has
  # to be made known to this one
  if (!impl_cuda_module_refresh(module$id, cuda_cache_dir_created())) {
    id <- cuda_module_register(module$code, module$filename, module$options, module$kernels, module$image)
    if (!identical(id, module$id)) {
      cli_abort("{.arg module} was created by another version of {.pkg pjrt}; create it again.")
    }
  }

  # an empty grid launches nothing, e.g. for zero-length operands
  grid <- launch_dims(grid, "grid", lower = 0)
  block <- launch_dims(block, "block", lower = 1)
  list(
    module = module$id,
    kernel = kernel,
    grid_x = grid[[1L]],
    grid_y = grid[[2L]],
    grid_z = grid[[3L]],
    block_x = block[[1L]],
    block_y = block[[2L]],
    block_z = block[[3L]],
    shared_mem = as.integer(shared_mem),
    scalars = paste(vapply(scalars, encode_cuda_scalar, character(1L)), collapse = ",")
  )
}

launch_dims <- function(x, arg, lower) {
  checkmate::assert_integerish(
    x,
    lower = lower,
    upper = .Machine$integer.max,
    min.len = 1L,
    max.len = 3L,
    any.missing = FALSE,
    .var.name = arg
  )
  as.integer(c(x, rep(1L, 3L - length(x))))
}

#' @title Typed Scalar for a CUDA Kernel
#' @description
#' Marks a scalar kernel argument with the C type it is passed as, for types
#' R has no native equivalent of (see [`pjrt_cuda_launch_attrs()`]).
#' @param value (`numeric(1)` | `logical(1)` | `bit64::integer64`)\cr
#'   The value. An `"i64"` beyond 2^53 needs a `bit64::integer64`.
#' @param dtype (`character(1)`)\cr
#'   One of `"i8"`, `"i16"`, `"i32"`, `"i64"`, `"ui8"`, `"ui16"`, `"ui32"`,
#'   `"f32"`, `"f64"` and `"pred"` (a `bool`).
#' @return `PJRTCudaScalar`
#' @examples
#' pjrt_cuda_scalar(1e6, "i64")
#' @export
pjrt_cuda_scalar <- function(value, dtype) {
  checkmate::assert_choice(dtype, c("i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "f32", "f64", "pred"))
  checkmate::assert_scalar(value, na.ok = FALSE)
  structure(list(value = value, dtype = dtype), class = "PJRTCudaScalar")
}

#' @export
print.PJRTCudaScalar <- function(x, ...) {
  cat(sprintf("<PJRTCudaScalar %s> %s\n", x$dtype, format(x$value)))
  invisible(x)
}

# The little-endian bytes of a scalar, as hex -- decoded by the launcher in
# src/cuda_kernel.cpp.
encode_cuda_scalar <- function(x) {
  if (!inherits(x, "PJRTCudaScalar")) {
    checkmate::assert_scalar(x, na.ok = FALSE)
    dtype <- if (is.integer(x)) {
      "i32"
    } else if (is.double(x)) {
      "f64"
    } else if (is.logical(x)) {
      "pred"
    } else {
      cli_abort("Cannot pass a {.cls {class(x)[[1L]]}} to a CUDA kernel.")
    }
    x <- pjrt_cuda_scalar(x, dtype)
  }
  v <- x$value
  dtype <- x$dtype
  bytes <- if (dtype == "pred") {
    as.raw(as.logical(v))
  } else if (dtype %in% c("f32", "f64")) {
    if (dtype == "f32" && is.finite(v) && abs(v) > 3.4028234663852886e38) {
      cli_abort("{.val {v}} is not representable as {.cls f32}.")
    }
    writeBin(as.double(v), raw(), size = if (dtype == "f32") 4L else 8L, endian = "little")
  } else {
    int_scalar_bytes(v, dtype)
  }
  paste(as.character(bytes), collapse = "")
}

# Integers go through their 64-bit two's complement, which integer64 stores
# in a double; the low bytes are then the value at any narrower width.
int_scalar_bytes <- function(v, dtype) {
  size <- switch(dtype, i8 = , ui8 = 1L, i16 = , ui16 = 2L, i32 = , ui32 = 4L, i64 = 8L)
  signed <- !startsWith(dtype, "u")
  bits <- 8L * size
  if (bit64::is.integer64(v)) {
    ok <- !signed || size == 8L
  } else {
    lower <- if (signed) -2^(bits - 1L) else 0
    upper <- if (signed) 2^(bits - 1L) - 1 else 2^bits - 1
    # beyond 2^53 a double no longer holds every integer
    ok <- v == round(v) && v >= lower && v <= upper && abs(v) <= 2^53
  }
  if (!ok) {
    cli_abort(c(
      "{.val {format(v)}} is not representable as {.cls {dtype}}.",
      i = if (dtype == "i64") "Pass a {.cls bit64::integer64} for values beyond 2^53."
    ))
  }
  writeBin(unclass(bit64::as.integer64(v)), raw(), size = 8L, endian = "little")[seq_len(size)]
}

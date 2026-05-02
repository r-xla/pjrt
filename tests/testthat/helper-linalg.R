# Helpers for the linear-algebra custom_call tests in test-linalg.R.
#
# We test the FFI handlers end-to-end by hand-rolling a tiny StableHLO program
# (one custom_call op) for each test, compiling it, and executing it. This
# keeps pjrt's tests self-contained -- no dependency on the stablehlo R
# package or anvl.
#
# All custom_calls in this file pass `operand_layouts` and `result_layouts`
# in column-major order (`dense<[0, 1]>` for 2D, `dense<[0]>` for 1D). The
# LAPACK/cuSOLVER handlers expect column-major data, and XLA does the
# row-major <-> column-major conversion transparently when these layout
# attributes are set.

mlir_dtype <- function(dtype) {
  switch(dtype, f32 = "f32", f64 = "f64", i32 = "i32",
         stop("unsupported dtype: ", dtype))
}

# Layout attribute string for an n-D column-major tensor: dense<[0, 1, ...]>.
col_major_layout <- function(ndim) {
  dims <- paste(seq_len(ndim) - 1L, collapse = ", ")
  sprintf("dense<[%s]> : tensor<%dxindex>", dims, ndim)
}

# Build the MLIR text for a single-input, multi-output custom_call program.
# `out_specs` is a list of list(dims, dtype) for each result.
build_program <- function(target, in_dims, in_dtype, out_specs) {
  in_type <- sprintf(
    "tensor<%sx%s>",
    paste(in_dims, collapse = "x"),
    mlir_dtype(in_dtype)
  )
  out_types <- vapply(out_specs, function(s) {
    sprintf("tensor<%sx%s>",
            paste(s$dims, collapse = "x"),
            mlir_dtype(s$dtype))
  }, character(1))
  out_layouts <- vapply(out_specs, function(s) col_major_layout(length(s$dims)),
                        character(1))
  ret_names <- paste0("%out", seq_along(out_specs))

  sprintf(
    'func.func @main(%%a: %s) -> (%s) {
  %s = stablehlo.custom_call @%s(%%a) {
    call_target_name = "%s",
    api_version = 4 : i32,
    operand_layouts = [%s],
    result_layouts = [%s]
  } : (%s) -> (%s)
  func.return %s : %s
}',
    in_type,
    paste(out_types, collapse = ", "),
    paste(ret_names, collapse = ", "),
    target, target,
    col_major_layout(length(in_dims)),
    paste(out_layouts, collapse = ", "),
    in_type,
    paste(out_types, collapse = ", "),
    paste(ret_names, collapse = ", "),
    paste(out_types, collapse = ", ")
  )
}

# Compile a custom_call program once, run on `a`, return the outputs as
# a list of base-R arrays/vectors.
#
# `dtype` is the dtype of the input AND of any floating-point output. Pass
# explicitly when you want to test the f32 path with R-double input data
# (R has no native f32). Defaults to f64 for double inputs, i32 for int.
run_linalg <- function(target, a, out_specs, dtype = NULL) {
  in_dims <- dim(a)
  if (is.null(in_dims)) in_dims <- length(a)
  if (is.null(dtype)) {
    dtype <- if (is.double(a)) "f64" else if (is.integer(a)) "i32" else "f32"
  }

  program <- pjrt_program(build_program(target, in_dims, dtype, out_specs))
  program <- pjrt_compile(program)
  outs <- pjrt_execute(program, pjrt_buffer(a, dtype = dtype))
  if (!is.list(outs)) outs <- list(outs)
  lapply(outs, as_array)
}

run_qr <- function(a) {
  m <- nrow(a); n <- ncol(a); k <- min(m, n)
  dtype <- if (is.double(a)) "f64" else "f32"
  run_linalg("qr", a, list(
    list(dims = c(m, k), dtype = dtype),
    list(dims = c(k, n), dtype = dtype)
  ))
}

run_lu <- function(a) {
  m <- nrow(a); n <- ncol(a); k <- min(m, n)
  dtype <- if (is.double(a)) "f64" else "f32"
  run_linalg("lu", a, list(
    list(dims = c(m, n), dtype = dtype),
    list(dims = k, dtype = "i32")
  ))
}

run_svd <- function(a) {
  m <- nrow(a); n <- ncol(a); k <- min(m, n)
  dtype <- if (is.double(a)) "f64" else "f32"
  run_linalg("svd", a, list(
    list(dims = c(m, k), dtype = dtype),
    list(dims = k, dtype = dtype),
    list(dims = c(k, n), dtype = dtype)
  ))
}

run_eigh <- function(a) {
  n <- nrow(a)
  dtype <- if (is.double(a)) "f64" else "f32"
  run_linalg("eigh", a, list(
    list(dims = c(n, n), dtype = dtype),
    list(dims = n, dtype = dtype)
  ))
}

# Tolerance helpers. f32 LAPACK tends to round to ~1e-5; f64 to ~1e-10.
# We multiply by sqrt(min(m, n)) to give larger problems a bit more slack
# (roundoff accumulates with problem size).
linalg_tol <- function(a, scale = 1) {
  base <- if (is.double(a)) 1e-10 else 1e-4
  scale * base * max(1, sqrt(min(dim(a))))
}

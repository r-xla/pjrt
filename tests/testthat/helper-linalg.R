col_major_layout <- function(ndim) {
  dims <- paste(seq_len(ndim) - 1L, collapse = ", ")
  sprintf("dense<[%s]> : tensor<%dxindex>", dims, ndim)
}

tensor_type <- function(spec) {
  sprintf("tensor<%sx%s>", paste(spec$dims, collapse = "x"), spec$dtype)
}

row_major_layout <- function(ndim) {
  dims <- paste(rev(seq_len(ndim) - 1L), collapse = ", ")
  sprintf("dense<[%s]> : tensor<%dxindex>", dims, ndim)
}

# `attrs`: optional named list of extra `key = "<mlir>"` entries inserted into
# the custom_call attribute dictionary (e.g. `list(backend_config = "{...}")`).
# `in_layouts` / `out_layouts`: optional character vectors of MLIR layout
# expressions, parallel to `in_specs` / `out_specs`. NULL means column-major.
build_program <- function(
  target,
  in_specs,
  out_specs,
  attrs = list(),
  in_layouts = NULL,
  out_layouts = NULL
) {
  in_types <- vapply(in_specs, tensor_type, character(1))
  out_types <- vapply(out_specs, tensor_type, character(1))
  if (is.null(in_layouts)) {
    in_layouts <- vapply(in_specs, function(s) col_major_layout(length(s$dims)), character(1))
  }
  if (is.null(out_layouts)) {
    out_layouts <- vapply(out_specs, function(s) col_major_layout(length(s$dims)), character(1))
  }
  arg_names <- paste0("%a", seq_along(in_specs))
  ret_names <- paste0("%out", seq_along(out_specs))

  # Per-input `aliases` field (1-based output index) emits
  # `{tf.aliasing_output = <0-based> : i32}`, donating that input into the
  # named output buffer.
  param_decls <- paste(
    vapply(
      seq_along(in_specs),
      function(i) {
        decl <- paste0(arg_names[i], ": ", in_types[i])
        aliases <- in_specs[[i]]$aliases
        if (!is.null(aliases)) {
          decl <- paste0(decl, " {tf.aliasing_output = ", aliases - 1L, " : i32}")
        }
        decl
      },
      character(1)
    ),
    collapse = ", "
  )

  extra_attr_lines <- if (length(attrs) > 0L) {
    paste0(
      ",\n    ",
      paste(
        vapply(
          names(attrs),
          function(k) sprintf("%s = %s", k, attrs[[k]]),
          character(1)
        ),
        collapse = ",\n    "
      )
    )
  } else {
    ""
  }

  sprintf(
    'func.func @main(%s) -> (%s) {
  %s = stablehlo.custom_call @%s(%s) {
    call_target_name = "%s",
    api_version = 4 : i32,
    operand_layouts = [%s],
    result_layouts = [%s]%s
  } : (%s) -> (%s)
  func.return %s : %s
}',
    param_decls,
    paste(out_types, collapse = ", "),
    paste(ret_names, collapse = ", "),
    target,
    paste(arg_names, collapse = ", "),
    target,
    paste(in_layouts, collapse = ", "),
    paste(out_layouts, collapse = ", "),
    extra_attr_lines,
    paste(in_types, collapse = ", "),
    paste(out_types, collapse = ", "),
    paste(ret_names, collapse = ", "),
    paste(out_types, collapse = ", ")
  )
}

# Compile a custom_call program once, run it on `inputs` (parallel to
# `in_specs`), return the outputs as a list of base-R arrays / vectors. Each
# input is uploaded with the dtype declared in its in_spec.
#
# If an in_spec carries an `aliases` field, that input is donated into the
# named output (via the `tf.aliasing_output` attribute emitted by
# build_program).
run_linalg <- function(
  target,
  inputs,
  in_specs,
  out_specs,
  attrs = list(),
  in_layouts = NULL,
  out_layouts = NULL
) {
  stopifnot(length(inputs) == length(in_specs))
  src <- build_program(
    target,
    in_specs,
    out_specs,
    attrs = attrs,
    in_layouts = in_layouts,
    out_layouts = out_layouts
  )
  program <- pjrt_compile(pjrt_program(src))
  bufs <- Map(
    function(x, s) pjrt_buffer(x, dtype = s$dtype),
    inputs,
    in_specs
  )
  outs <- do.call(pjrt_execute, c(list(program), bufs))
  if (!is.list(outs)) {
    outs <- list(outs)
  }
  lapply(outs, as_array)
}

# Run a kernel and assert every input device buffer is byte-identical
# afterwards. Use this for ops whose output shapes match the input shapes,
# where an in-place LAPACK / cuSOLVER call could clobber the input if the
# kernel forgot to copy it into the output buffer first. Catches regressions
# (e.g. cuSOLVER getrf / syevd write A in place) that R-level array
# semantics would mask, since `pjrt_buffer(a, ...)` doesn't tie `a` and the
# device buffer together.
expect_inputs_preserved <- function(
  target,
  inputs,
  in_specs,
  out_specs,
  attrs = list(),
  in_layouts = NULL,
  out_layouts = NULL
) {
  src <- build_program(
    target,
    in_specs = in_specs,
    out_specs = out_specs,
    attrs = attrs,
    in_layouts = in_layouts,
    out_layouts = out_layouts
  )
  exec <- pjrt_compile(pjrt_program(src))
  bufs <- Map(
    function(x, s) pjrt_buffer(x, dtype = s$dtype),
    inputs,
    in_specs
  )
  pre <- lapply(bufs, as_array)
  do.call(pjrt_execute, c(list(exec), bufs))
  for (i in seq_along(bufs)) {
    expect_equal(as_array(bufs[[i]]), pre[[i]], tolerance = 0)
  }
}

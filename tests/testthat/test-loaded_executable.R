test_that("device mismatch raises error", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src), device = "cpu:0")
  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32", device = "cpu:1")
  expect_error(
    pjrt_execute(executable, input),
    "compiled for device"
  )
})

test_that("arguments must be unnamed", {
  skip_if_metal("only supports MLIR programs")
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")
  executable <- pjrt_compile(program)
  buf <- pjrt_buffer(1)
  expect_error(pjrt_execute(executable, a = buf, "Expected unnamed arguments"))
})

test_that("execute program without arguments", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_class(result, "PJRTBuffer")
  expect_equal(as_array(result), 3)
})

test_that("can return two values", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_list(result, types = "PJRTBuffer", len = 2L)
  expect_equal(as_array(result[[1]]), 3)
  expect_equal(as_array(result[[2]]), 7)
})

test_that("single-output returns list when simplify=FALSE", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable, simplify = FALSE)
  expect_list(result, types = "PJRTBuffer", len = 1L)
  expect_equal(as_array(result[[1]]), 3)

  result <- pjrt_execute(executable, simplify = TRUE)
  expect_class(result, "PJRTBuffer")
  expect_equal(as_array(result), 3)
})

test_that("can execute empty constant", {
  path <- system.file("programs/stablehlo-empty-constant.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_equal(as_array(result), array(integer(), 0L))
})

test_that("print works", {
  path <- system.file("programs/stablehlo-empty-constant.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  expect_snapshot(executable)
})

test_that("multiple inputs work correctly", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  src <- r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  x <- pjrt_buffer(matrix(1:4, 2, 2), dtype = "f32")
  y <- pjrt_buffer(matrix(5:8, 2, 2), dtype = "f32")

  result <- pjrt_execute(executable, x, y)

  arr <- as_array(result)
  expect_equal(as.vector(arr), as.vector(matrix(1:4, 2, 2) + matrix(5:8, 2, 2)), tolerance = 1e-6)
})

test_that("wrong shape input raises error", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  wrong_input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # CPU says "size", Metal says "shape"
  expect_error(
    pjrt_execute(executable, wrong_input),
    "size|shape"
  )
})

# is_ready / await / value -------------------------------------------------

test_that("is_ready works on buffers and array promises", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)

  ready <- is_ready(result)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)

  arr_promise <- as_array_async(result)
  ready <- is_ready(arr_promise)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("await blocks until buffer is ready", {
  src <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(matrix(runif(1e6), 1000, 1000), dtype = "f32")
  result <- pjrt_execute(executable, input)
  buf <- await(result)
  expect_class(buf, "PJRTBuffer")
  expect_true(is_ready(buf))
})

test_that("value() materializes R array from promise", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute(executable)
  arr <- value(as_array_async(result))
  expect_equal(arr, 3)
})

test_that("value() caches result on repeated calls", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr_promise <- as_array_async(result)

  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)

  # Call value again - should return cached result
  arr2 <- value(arr_promise)
  expect_identical(arr, arr2)
})

# Chaining ------------------------------------------------------------------

test_that("chained execution produces correct results", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%0): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute(executable, input)
  arr <- value(as_array_async(result))
  expect_equal(as.vector(arr), c(2.0, 4.0, 6.0), tolerance = 1e-6)
})

test_that("longer chains produce correct results", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  exec1 <- pjrt_compile(pjrt_program(src))
  exec2 <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  result1 <- pjrt_execute(exec1, input)
  result2 <- pjrt_execute(exec2, result1)
  arr <- value(as_array_async(result2))
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("execution with inputs chained to buffer-to-host", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  path <- system.file("programs/jax-stablehlo-subset-2d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  x_buf <- pjrt_buffer(x)
  i1_buf <- pjrt_scalar(0L, "i32")
  i2_buf <- pjrt_scalar(1L, "i32")

  async_result <- pjrt_execute(executable, x_buf, i1_buf, i2_buf)
  result <- value(as_array_async(async_result))
  expect_equal(result, x[1, 2]) # 0-indexed: x[0+1, 1+1] = x[1, 2] = 3
})

test_that("print works on not-yet-ready buffer", {
  src <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer(matrix(runif(1e6), 1000, 1000), dtype = "f32")
  result <- pjrt_execute(executable, input)
  expect_output(print(result), "PJRTBuffer")
})

test_that("cpu devices work", {
  src <- r"(
func.func @main(%x: tensor<1xf32>) -> tensor<1xf32> {
  "func.return"(%x): (tensor<1xf32>) -> ()
}
)"
  dev0 <- as_pjrt_device("cpu:0")
  dev1 <- as_pjrt_device("cpu:1")
  progr <- pjrt_program(src)
  exec0 <- pjrt_compile(progr, device = dev0)
  exec1 <- pjrt_compile(progr, device = dev1)
  expect_equal(device(exec0), dev0)
  expect_equal(device(exec1), dev1)
  expect_equal(
    device(pjrt_execute(exec0, pjrt_buffer(1, device = dev0))),
    dev0
  )
  expect_equal(
    device(pjrt_execute(exec1, pjrt_buffer(1, device = dev1))),
    dev1
  )
  expect_error(
    pjrt_execute(exec1, pjrt_buffer(1, device = dev0)),
    "is on device"
  )
  expect_error(
    pjrt_execute(exec0, pjrt_buffer(1, device = dev1)),
    "is on device"
  )
})

# The input/output donation aliases declared in a program are parsed at compile
# time and cached on the loaded executable (exposed via
# impl_loaded_executable_aliases). These tests pin down the parser's correctness;
# the runtime donation/keepalive behaviour they drive lives in the describe block
# below.
describe("input/output alias parsing", {
  it("exposes input_output_alias entries on the compiled executable", {
    mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    aliases <- impl_loaded_executable_aliases(exec)
    expect_equal(aliases$input, 0L)
    expect_equal(aliases$output, 0L)
  })

  it("parses input_output_alias entries regardless of whitespace", {
    single <- function(open, attr) {
      sprintf(
        '
module @m {
  func.func @main%s(%%arg0: tensor<4xf32> {%s}) -> tensor<4xf32> {
    %%out = stablehlo.multiply %%arg0, %%arg0 : tensor<4xf32>
    return %%out : tensor<4xf32>
  }
}',
        open,
        attr
      )
    }
    variants <- list(
      space_before_paren = single(" ", "tf.aliasing_output = 0 : i32"),
      tab_before_paren = single("\t", "tf.aliasing_output = 0 : i32"),
      no_space_eq = single("", "tf.aliasing_output=0 : i32"),
      extra_spaces_eq = single("", "tf.aliasing_output   =   0 : i32"),
      no_space_colon = single("", "tf.aliasing_output=0:i32"),
      space_after_colon_only = single("", "tf.aliasing_output = 0: i32"),
      # A same-prefix symbol declared before @main must not be mistaken for it.
      lookalike_symbol_first = '
module @m {
  func.func private @mainhelper(%a: tensor<4xf32>) -> tensor<4xf32> {
    return %a : tensor<4xf32>
  }
  func.func @main (%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %out = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}'
    )
    for (nm in names(variants)) {
      exec <- pjrt_compile(
        pjrt_program(src = variants[[nm]], format = "mlir"),
        device = "cpu"
      )
      aliases <- impl_loaded_executable_aliases(exec)
      expect_equal(aliases$input, 0L, info = nm)
      expect_equal(aliases$output, 0L, info = nm)
    }

    multi <- '
module @m {
  func.func @main(
      %arg0: tensor<3xf32> {tf.aliasing_output = 0 : i32},
      %arg1: tensor<3xf32> {tf.aliasing_output = 1 : i32}
  ) -> (tensor<3xf32>, tensor<3xf32>) {
    %a = stablehlo.add %arg0, %arg0 : tensor<3xf32>
    %b = stablehlo.multiply %arg1, %arg1 : tensor<3xf32>
    return %a, %b : tensor<3xf32>, tensor<3xf32>
  }
}'
    exec <- pjrt_compile(pjrt_program(src = multi, format = "mlir"), device = "cpu")
    aliases <- impl_loaded_executable_aliases(exec)
    expect_equal(aliases$input, c(0L, 1L))
    expect_equal(aliases$output, c(0L, 1L))
  })

  # The output index M is read digit-by-digit and the parameter index is recovered
  # by counting argument commas, so both must keep working past a single digit. Use
  # 12 arguments/outputs and alias the last (index 11 -> output 11).
  it("parses double-digit alias indices", {
    n <- 12L
    idx <- 0:(n - 1L)
    args <- paste0(
      sprintf("      %%arg%d: tensor<2xf32>", idx),
      ifelse(idx == n - 1L, sprintf(" {tf.aliasing_output = %d : i32}", n - 1L), ""),
      collapse = ",\n"
    )
    types <- paste(rep("tensor<2xf32>", n), collapse = ", ")
    body <- paste0(
      sprintf("    %%out%d = stablehlo.add %%arg%d, %%arg%d : tensor<2xf32>", idx, idx, idx),
      collapse = "\n"
    )
    rets <- paste0("%out", idx, collapse = ", ")
    mlir <- sprintf(
      "\nmodule @m {\n  func.func @main(\n%s\n  ) -> (%s) {\n%s\n    return %s : %s\n  }\n}",
      args,
      types,
      body,
      rets,
      types
    )
    exec <- pjrt_compile(pjrt_program(src = mlir, format = "mlir"), device = "cpu")
    aliases <- impl_loaded_executable_aliases(exec)
    expect_equal(aliases$input, 11L)
    expect_equal(aliases$output, 11L)
  })

  # A non-identity mapping (arg0 -> output 1, arg1 -> output 0) confirms each input
  # is paired with its *declared* output index rather than with its own position.
  it("parses non-identity input/output alias mappings", {
    mlir <- '
module @m {
  func.func @main(
      %arg0: tensor<3xf32> {tf.aliasing_output = 1 : i32},
      %arg1: tensor<3xf32> {tf.aliasing_output = 0 : i32}
  ) -> (tensor<3xf32>, tensor<3xf32>) {
    %a = stablehlo.add %arg0, %arg0 : tensor<3xf32>
    %b = stablehlo.multiply %arg1, %arg1 : tensor<3xf32>
    return %a, %b : tensor<3xf32>, tensor<3xf32>
  }
}'
    exec <- pjrt_compile(pjrt_program(src = mlir, format = "mlir"), device = "cpu")
    aliases <- impl_loaded_executable_aliases(exec)
    expect_equal(aliases$input, c(0L, 1L))
    expect_equal(aliases$output, c(1L, 0L))
  })
})

# Runtime behaviour of input/output donation: on execute, PJRT donates the
# aliased input and pjrt_execute migrates the input's keepalive (the backing
# RAWSXP) to the aliased output, leaving the input invalidated.
describe("input/output donation keepalive", {
  it("transfers keepalive from donated input to aliased output", {
    skip_if(!is_cpu())
    mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    x <- pjrt_buffer(c(10, 20, 30, 40), dtype = "f32")
    x_prot <- impl_test_xptr_prot(x)
    expect_true(is.raw(x_prot))

    out <- pjrt_execute(exec, x)

    # Output's prot slot now holds the input's RAWSXP; input's is cleared.
    expect_identical(impl_test_xptr_prot(out), x_prot)
    expect_null(impl_test_xptr_prot(x))

    expect_equal(as.numeric(as_array(out)), c(20, 40, 60, 80), tolerance = 1e-6)
  })

  it("raises a clean R-level error on operations on a donated input", {
    skip_if(!is_cpu())
    mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    x <- pjrt_buffer(c(1, 2, 3, 4), dtype = "f32")
    pjrt_execute(exec, x)
    expect_error(as_array(x), "called on deleted or donated buffer")
  })

  # tf.aliasing_output is a *may*-alias: PJRT donates the input only if it is
  # donatable at runtime, otherwise it copies and leaves the input valid.
  it("leaves the input untouched when a may-alias is not donated", {
    skip_if(!is_cpu())
    mlir <- '
module @double_inplace {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    x <- pjrt_buffer(c(10, 20, 30, 40), dtype = "f32")
    x_prot <- impl_test_xptr_prot(x)

    opts <- pjrt_execution_options(non_donatable_input_indices = 0L)
    out <- pjrt_execute(exec, x, execution_options = opts)

    # Input was copied, not donated: its keepalive stays put and input remains readable.
    expect_identical(impl_test_xptr_prot(x), x_prot)
    expect_equal(as.numeric(as_array(x)), c(10, 20, 30, 40), tolerance = 1e-6)

    expect_equal(as.numeric(as_array(out)), c(20, 40, 60, 80), tolerance = 1e-6)
    expect_equal(as.numeric(as_array(x)), c(10, 20, 30, 40), tolerance = 1e-6)
  })

  # Donating the same buffer to two parameters is rejected by PJRT before any
  # donation happens, so the input must remain valid after the error.
  it("rejects donating the same buffer twice and leaves it valid", {
    skip_if(!is_cpu())
    mlir <- '
module @two_donations {
  func.func @main(%arg0: tensor<3xf32> {tf.aliasing_output = 0 : i32},
                  %arg1: tensor<3xf32> {tf.aliasing_output = 1 : i32})
      -> (tensor<3xf32>, tensor<3xf32>) {
    %a = stablehlo.add %arg0, %arg0 : tensor<3xf32>
    %b = stablehlo.multiply %arg1, %arg1 : tensor<3xf32>
    return %a, %b : tensor<3xf32>, tensor<3xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    x <- pjrt_buffer(c(1, 2, 3), dtype = "f32")
    expect_error(pjrt_execute(exec, x, x, simplify = FALSE), "donate the same buffer twice")
  })

  it("migrates each donated input to its own aliased output", {
    skip_if(!is_cpu())
    mlir <- '
module @two_donations {
  func.func @main(%arg0: tensor<3xf32> {tf.aliasing_output = 0 : i32},
                  %arg1: tensor<3xf32> {tf.aliasing_output = 1 : i32})
      -> (tensor<3xf32>, tensor<3xf32>) {
    %a = stablehlo.add %arg0, %arg0 : tensor<3xf32>
    %b = stablehlo.multiply %arg1, %arg1 : tensor<3xf32>
    return %a, %b : tensor<3xf32>, tensor<3xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    a <- pjrt_buffer(c(1, 2, 3), dtype = "f32")
    b <- pjrt_buffer(c(4, 5, 6), dtype = "f32")
    a_prot <- impl_test_xptr_prot(a)
    b_prot <- impl_test_xptr_prot(b)

    outs <- pjrt_execute(exec, a, b, simplify = FALSE)

    expect_identical(impl_test_xptr_prot(outs[[1]]), a_prot)
    expect_identical(impl_test_xptr_prot(outs[[2]]), b_prot)
    expect_null(impl_test_xptr_prot(a))
    expect_null(impl_test_xptr_prot(b))

    expect_equal(as.numeric(outs[[1]]), c(2, 4, 6), tolerance = 1e-6)
    expect_equal(as.numeric(outs[[2]]), c(16, 25, 36), tolerance = 1e-6)
  })

  it("pjrt_execute drains the deferred-release queue", {
    skip_if(!is_cpu())
    impl_process_pending_releases()
    impl_test_enqueue_release(raw(8L))
    expect_gte(impl_pending_release_count(), 1L)

    mlir <- '
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<4xf32>
    %out = stablehlo.multiply %arg0, %two : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
'
    prog <- pjrt_program(src = mlir, format = "mlir")
    exec <- pjrt_compile(prog, device = "cpu")
    pjrt_execute(exec, pjrt_buffer(c(1, 2, 3, 4), dtype = "f32"))

    expect_equal(impl_pending_release_count(), 0L)
  })
})

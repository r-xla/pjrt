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

test_that("print handler prints input tensors", {
  program <- pjrt_program(
    r"(
func.func @main(
  %f: tensor<4xf32>,
  %i: tensor<4xi32>,
  %b: tensor<4xi1>
) -> (tensor<4xf32>, tensor<4xi32>, tensor<4xi1>) {
  stablehlo.custom_call @print_tensor(%f) {
    call_target_name = "print_tensor",
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<4xf32>) -> ()
  stablehlo.custom_call @print_tensor(%i) {
    call_target_name = "print_tensor",
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<4xi32>) -> ()
  stablehlo.custom_call @print_tensor(%b) {
    call_target_name = "print_tensor",
    backend_config = {
      print_header = "TestBuffer",
      print_footer = "CustomFooter"
    },
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<4xi1>) -> ()
  "func.return" (%f, %i, %b) : (tensor<4xf32>, tensor<4xi32>, tensor<4xi1>) -> ()
}
)"
  )

  program <- pjrt_compile(program)

  buf_f32 <- pjrt_buffer(as.double(1:4), dtype = "f32")
  buf_i32 <- pjrt_buffer(5:8, dtype = "i32")
  buf_pred <- pjrt_buffer(c(TRUE, FALSE, TRUE, FALSE))

  expect_snapshot({
    invisible(pjrt_execute(program, buf_f32, buf_i32, buf_pred))
  })
})

test_that("the ordered print handler prints in the order of its token chain", {
  # Each print takes the token the one before it returns, so the prints run in
  # the order of the chain.
  program <- pjrt_program(
    r"(
func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
  %t0 = stablehlo.after_all : !stablehlo.token
  %t1 = stablehlo.custom_call @print_tensor_ordered(%x, %t0) {
    call_target_name = "print_tensor_ordered",
    backend_config = {print_header = "", print_footer = "1"},
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<2xf32>, !stablehlo.token) -> !stablehlo.token
  %t2 = stablehlo.custom_call @print_tensor_ordered(%x, %t1) {
    call_target_name = "print_tensor_ordered",
    backend_config = {print_header = "", print_footer = "2"},
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<2xf32>, !stablehlo.token) -> !stablehlo.token
  %t3 = stablehlo.custom_call @print_tensor_ordered(%x, %t2) {
    call_target_name = "print_tensor_ordered",
    backend_config = {print_header = "", print_footer = "3"},
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<2xf32>, !stablehlo.token) -> !stablehlo.token
  "func.return" (%x) : (tensor<2xf32>) -> ()
}
)"
  )
  program <- pjrt_compile(program)
  out <- capture.output(invisible(pjrt_execute(program, pjrt_buffer(c(1, 2), dtype = "f32"))))
  expect_identical(grep("^[123]$", out, value = TRUE), c("1", "2", "3"))
})

test_that("print handler default footer prints bool not pred", {
  program <- pjrt_program(
    r"(
func.func @main(
  %b: tensor<3xi1>
) -> tensor<3xi1> {
  stablehlo.custom_call @print_tensor(%b) {
    call_target_name = "print_tensor",
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<3xi1>) -> ()
  "func.return" (%b) : (tensor<3xi1>) -> ()
}
)"
  )

  program <- pjrt_compile(program)
  buf <- pjrt_buffer(c(TRUE, FALSE, TRUE))

  expect_snapshot({
    invisible(pjrt_execute(program, buf))
  })
})

test_that("print handler supports empty header", {
  program <- pjrt_program(
    r"(
func.func @main(
  %x: tensor<3xf32>
) -> tensor<3xf32> {
  stablehlo.custom_call @print_tensor(%x) {
    call_target_name = "print_tensor",
    backend_config = {
      print_header = ""
    },
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<3xf32>) -> ()
  "func.return" (%x) : (tensor<3xf32>) -> ()
}
)"
  )

  program <- pjrt_compile(program)
  buf <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  expect_snapshot({
    invisible(pjrt_execute(program, buf))
  })
})

test_that("print handler supports custom footer", {
  program <- pjrt_program(
    r"(
func.func @main(
  %x: tensor<3xi32>
) -> tensor<3xi32> {
  stablehlo.custom_call @print_tensor(%x) {
    call_target_name = "print_tensor",
    backend_config = {
      print_footer = "[my custom footer]"
    },
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<3xi32>) -> ()
  "func.return" (%x) : (tensor<3xi32>) -> ()
}
)"
  )

  program <- pjrt_compile(program)
  buf <- pjrt_buffer(1:3, dtype = "i32")

  expect_snapshot({
    invisible(pjrt_execute(program, buf))
  })
})

test_that("print handler supports no head and no footer", {
  program <- pjrt_program(
    r"(
func.func @main(
  %x: tensor<3xi32>
) -> tensor<3xi32> {
  stablehlo.custom_call @print_tensor(%x) {
    call_target_name = "print_tensor",
    backend_config = {
      print_header = "",
      print_footer = ""
    },
    has_side_effect = true,
    api_version = 4 : i32
  } : (tensor<3xi32>) -> ()
  "func.return" (%x) : (tensor<3xi32>) -> ()
}
)"
  )

  program <- pjrt_compile(program)
  buf <- pjrt_buffer(1:3, dtype = "i32")

  expect_snapshot({
    invisible(pjrt_execute(program, buf))
  })
})

describe("print_tensor's reverse_dims", {
  # A program printing its operand of type `ty`, with the operand layout
  # `layout` (minor to major) and the backend config entries `config`.
  print_program <- function(ty, layout, config = "") {
    pjrt_compile(pjrt_program(sprintf(
      r"(
func.func @main(%%x: %s) -> %s {
  stablehlo.custom_call @print_tensor(%%x) {
    call_target_name = "print_tensor",
    backend_config = {%s},
    has_side_effect = true,
    api_version = 4 : i32,
    operand_layouts = [dense<[%s]> : tensor<%dxindex>],
    result_layouts = []
  } : (%s) -> ()
  "func.return"(%%x) : (%s) -> ()
}
)",
      ty,
      ty,
      config,
      paste(layout, collapse = ", "),
      length(layout),
      ty,
      ty
    )))
  }
  tensor_ty <- function(shape) sprintf("tensor<%sxf32>", paste(shape, collapse = "x"))
  printed <- function(program, buf) capture.output(invisible(as_array(pjrt_execute(program, buf))))

  it("prints a column-major operand as the array of its dims reversed", {
    for (shape in list(c(2L, 3L), c(2L, 3L, 4L))) {
      x <- array(seq_len(prod(shape)) / 4, shape)
      n <- length(shape)
      # The R array stored with its shape reversed, its operand declared
      # column-major: the handler receives the bytes row-major over `shape`.
      reversed <- printed(
        print_program(tensor_ty(rev(shape)), seq_len(n) - 1L, "reverse_dims = true"),
        pjrt_buffer(x, dtype = "f32", shape = rev(shape), row_major = TRUE)
      )
      ordinary <- printed(
        print_program(tensor_ty(shape), rev(seq_len(n) - 1L)),
        pjrt_buffer(x, dtype = "f32")
      )
      expect_identical(reversed, ordinary)
      expect_identical(
        reversed[[length(reversed)]],
        sprintf("[ f32{%s} ]", paste(shape, collapse = ","))
      )
    }
  })

  it("prints the operand as it is when false", {
    x <- matrix(1:6, 2L, 3L)
    expect_identical(
      printed(print_program(tensor_ty(c(2L, 3L)), c(1L, 0L), "reverse_dims = false"), pjrt_buffer(x, dtype = "f32")),
      printed(print_program(tensor_ty(c(2L, 3L)), c(1L, 0L)), pjrt_buffer(x, dtype = "f32"))
    )
  })
})

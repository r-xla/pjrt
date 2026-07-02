# A two-op program (add then multiply) so that "before" and "after"
# optimization differ visibly (the backend fuses the two ops).
dump_test_program <- function() {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "stablehlo.multiply"(%0, %x) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%1) : (tensor<2x2xf32>) -> ()
}
)"
  pjrt_program(src)
}

test_that("xla_dump_dir_from_flags extracts the dump directory", {
  expect_null(xla_dump_dir_from_flags(""))
  expect_null(xla_dump_dir_from_flags("--xla_dump_hlo_as_text"))
  expect_equal(
    xla_dump_dir_from_flags("--xla_dump_to=/tmp/x --xla_dump_hlo_as_text"),
    "/tmp/x"
  )
})

test_that("parse_hlo_dump keys stages and orders passes", {
  dir <- withr::local_tempdir()
  files <- c(
    "module_0000.main.before_optimizations.txt",
    "module_0000.main.cpu_after_optimizations.txt",
    "module_0000.main.cpu_after_optimizations-buffer-assignment.txt",
    "module_0000.main.0002.passC.after_x.before_y.txt",
    "module_0000.main.0000.passA.after_x.before_y.txt",
    "module_0000.main.0001.passB.after_x.before_y.txt"
  )
  for (i in seq_along(files)) {
    writeLines(paste0("content ", i), file.path(dir, files[[i]]))
  }

  dump <- parse_hlo_dump(dir, files)

  expect_s3_class(dump, "PJRTHLODump")
  expect_equal(attr(dump, "module"), "main")
  # backend-prefixed optimized file is picked; the buffer-assignment sibling is not.
  expect_true("after_optimizations" %in% names(dump))
  expect_false(any(grepl("buffer-assignment", names(dump))))
  # input first, passes in index order, optimized last.
  expect_equal(
    names(dump),
    c(
      "before_optimizations",
      "0000.passA.after_x.before_y",
      "0001.passB.after_x.before_y",
      "0002.passC.after_x.before_y",
      "after_optimizations"
    )
  )
})

test_that("pjrt_dump_hlo returns the input and optimized HLO", {
  dump <- pjrt_dump_hlo(dump_test_program())

  expect_s3_class(dump, "PJRTHLODump")
  expect_true(all(
    c("before_optimizations", "after_optimizations") %in% names(dump)
  ))
  expect_true(nzchar(dump[["before_optimizations"]]))
  expect_true(nzchar(dump[["after_optimizations"]]))
  expect_match(dump[["before_optimizations"]], "add")
  expect_match(dump[["before_optimizations"]], "multiply")
})

test_that("pjrt_dump_hlo shows the effect of optimization", {
  dump <- pjrt_dump_hlo(dump_test_program())
  # The optimizer fuses the two ops, so the optimized HLO differs from the input.
  expect_false(
    identical(dump[["before_optimizations"]], dump[["after_optimizations"]])
  )
  expect_match(dump[["after_optimizations"]], "fusion")
})

test_that("as.character returns the optimized HLO and print works", {
  dump <- pjrt_dump_hlo(dump_test_program())
  expect_identical(as.character(dump), dump[["after_optimizations"]])
  expect_output(print(dump), "PJRTHLODump")
})

test_that("pjrt_dump_hlo validates arguments", {
  expect_error(pjrt_dump_hlo(dump_test_program(), passes = "yes"), "passes")
})

test_that("pjrt_dump_hlo works on CUDA", {
  skip_if(!is_cuda(), "Not running on CUDA platform")
  dump <- pjrt_dump_hlo(dump_test_program(), device = "cuda")

  expect_s3_class(dump, "PJRTHLODump")
  expect_true(all(
    c("before_optimizations", "after_optimizations") %in% names(dump)
  ))
  expect_true(nzchar(dump[["after_optimizations"]]))
  # Optimization changes the module on GPU as well.
  expect_false(
    identical(dump[["before_optimizations"]], dump[["after_optimizations"]])
  )
})

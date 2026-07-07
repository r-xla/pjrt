# A two-op program (add then multiply) so that "before" and "after"
# optimization differ visibly (the backend fuses the two ops).
dump_test_src <- function() {
  r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "stablehlo.multiply"(%0, %x) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%1) : (tensor<2x2xf32>) -> ()
}
)"
}

dump_test_program <- function() {
  pjrt_program(dump_test_src())
}

# XLA reads XLA_FLAGS once, before the first compilation in a process, so real
# dumping cannot be exercised inside the shared test session (many compiles have
# already happened). Run it in a fresh R process instead, with XLA_FLAGS set from
# the start -- no callr needed, just Rscript. Returns the parsed dump stages.
dump_in_fresh_r <- function(device = NULL) {
  rscript <- file.path(R.home("bin"), "Rscript")
  skip_if(!file.exists(rscript), "Rscript not available")

  progfile <- withr::local_tempfile(fileext = ".mlir")
  writeLines(dump_test_src(), progfile)
  dumpdir <- withr::local_tempdir()
  out <- withr::local_tempfile(fileext = ".rds")
  script <- withr::local_tempfile(fileext = ".R")

  dev_arg <- if (is.null(device)) "NULL" else deparse(device)
  writeLines(
    c(
      "library(pjrt)",
      sprintf('prog <- pjrt_program(path = %s, format = "mlir")', deparse(progfile)),
      sprintf("d <- pjrt_dump_hlo(prog, device = %s)", dev_arg),
      sprintf(
        "saveRDS(list(stages = names(d), before = d[['before_optimizations']], after = d[['after_optimizations']]), %s)",
        deparse(out)
      )
    ),
    script
  )

  # Set env in this process (harmless -- XLA is already initialised here) and let
  # the child inherit it; passing via system2(env=) would not quote the space in
  # XLA_FLAGS. R_TESTS is dropped because it breaks a nested R started under R CMD
  # check.
  output <- withr::with_envvar(
    c(
      R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep),
      XLA_FLAGS = sprintf("--xla_dump_to=%s --xla_dump_hlo_as_text", dumpdir),
      R_TESTS = NA
    ),
    suppressWarnings(system2(
      rscript,
      shQuote(script),
      stdout = TRUE,
      stderr = TRUE
    ))
  )
  if (!file.exists(out)) {
    stop("dump subprocess failed:\n", paste(output, collapse = "\n"))
  }
  readRDS(out)
}

test_that("parse_hlo_dump keys the input and optimized stages", {
  dir <- withr::local_tempdir()
  files <- c(
    "module_0000.main.before_optimizations.txt",
    "module_0000.main.cpu_after_optimizations.txt",
    "module_0000.main.cpu_after_optimizations-buffer-assignment.txt"
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
  expect_equal(names(dump), c("before_optimizations", "after_optimizations"))
})

test_that("session_dump_dir detects usable dump flags", {
  # No XLA_FLAGS at all -> not enabled.
  withr::local_envvar(XLA_FLAGS = NA)
  expect_null(session_dump_dir())

  # Text dumping enabled -> returns the dump directory.
  withr::local_envvar(
    XLA_FLAGS = "--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text"
  )
  expect_equal(session_dump_dir(), "/tmp/hlo")

  # A dump dir without --xla_dump_hlo_as_text is not usable.
  withr::local_envvar(XLA_FLAGS = "--xla_dump_to=/tmp/hlo")
  expect_null(session_dump_dir())
})

test_that("pjrt_dump_hlo errors informatively when XLA_FLAGS is not set", {
  withr::local_envvar(XLA_FLAGS = NA)
  expect_error(pjrt_dump_hlo(dump_test_program()), "XLA_FLAGS")
})

test_that("pjrt_dump_hlo returns the input and optimized HLO, showing optimization", {
  res <- dump_in_fresh_r()

  expect_setequal(res$stages, c("before_optimizations", "after_optimizations"))
  # Input HLO contains the two ops the user wrote.
  expect_match(res$before, "add")
  expect_match(res$before, "multiply")
  # Optimization fuses them, so the optimized HLO differs and mentions a fusion.
  expect_false(identical(res$before, res$after))
  expect_match(res$after, "fusion")
})

test_that("pjrt_dump_hlo works on CUDA", {
  skip_if(!is_cuda(), "Not running on CUDA platform")
  res <- dump_in_fresh_r(device = "cuda")

  expect_setequal(res$stages, c("before_optimizations", "after_optimizations"))
  expect_true(nzchar(res$after))
  # Optimization changes the module on GPU as well.
  expect_false(identical(res$before, res$after))
})

# A two-op program (add then multiply) exercised by the dump tests.
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

# A second, content-distinct program (a single subtract) so two dumps can be told
# apart by their HLO text: this one mentions "subtract" and never "multiply".
dump_test_src_sub <- function() {
  r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.subtract"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()
}
)"
}

# XLA reads XLA_FLAGS once, before the first compilation in a process, so real
# dumping cannot be exercised inside the shared test session (many compiles have
# already happened). Run it in a fresh R process via callr instead, with XLA_FLAGS
# set from the start. `dumpdir` is exposed so a caller can reuse one directory
# across several runs. Returns the parsed dump stages.
dump_in_fresh_r <- function(src = dump_test_src(), dumpdir = withr::local_tempdir()) {
  skip_if_not_installed("callr")

  callr::r(
    function(src) {
      prog <- pjrt::pjrt_program(src, format = "mlir")
      d <- pjrt::inspect_hlo(prog)
      list(
        stages = names(d),
        before = d[["before_optimizations"]],
        after = d[["after_optimizations"]]
      )
    },
    args = list(src = src),
    env = c(
      callr::rcmd_safe_env(),
      XLA_FLAGS = sprintf("--xla_dump_to=%s --xla_dump_hlo_as_text", dumpdir)
    )
  )
}

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

# XLA_FLAGS misconfigured (not set) -> a clear error, in-process, no callr needed.
test_that("inspect_hlo errors informatively when XLA_FLAGS is not set", {
  withr::local_envvar(XLA_FLAGS = NA)
  expect_error(inspect_hlo(dump_test_program()), "XLA_FLAGS")
})

# XLA_FLAGS correctly configured -> both optimization stages exist, and reusing
# one dump dir across runs still yields each run's own program. The two runs share
# a dump dir, so the second overwrites the first's identically named module_0000
# files in place; inspect_hlo detects the rewrite by mtime (a name-only diff would
# miss it and wrongly report "no HLO"). Checking the HLO text confirms each run
# gets its own program, not stale output from the other.
test_that("inspect_hlo returns each program's HLO, even when the dump dir is reused", {
  skip_if_not_installed("callr")
  dumpdir <- withr::local_tempdir()

  res_mul <- dump_in_fresh_r(src = dump_test_src(), dumpdir = dumpdir)
  res_sub <- dump_in_fresh_r(src = dump_test_src_sub(), dumpdir = dumpdir)

  # First run: the add/multiply program, with both stages populated.
  expect_setequal(res_mul$stages, c("before_optimizations", "after_optimizations"))
  expect_true(nzchar(res_mul$after))
  expect_match(res_mul$before, "multiply")
  expect_false(grepl("subtract", res_mul$before, fixed = TRUE))

  # Second run into the same dir: the subtract program's HLO, not stale content.
  expect_match(res_sub$before, "subtract")
  expect_false(grepl("multiply", res_sub$before, fixed = TRUE))
})

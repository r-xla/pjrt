# XLA reads XLA_FLAGS once, before the first compilation in the process. Set a
# dump directory here (before any test triggers a compilation) so the
# pjrt_dump_hlo() integration tests can read back the HLO IR regardless of the
# order in which test files run. Only text dumps are enabled to keep this cheap.
local({
  if (!grepl("--xla_dump_to", Sys.getenv("XLA_FLAGS"))) {
    dir <- file.path(tempdir(), "pjrt_hlo_dump_tests")
    dir.create(dir, showWarnings = FALSE, recursive = TRUE)
    Sys.setenv(
      XLA_FLAGS = trimws(paste(
        Sys.getenv("XLA_FLAGS"),
        sprintf("--xla_dump_to=%s --xla_dump_hlo_as_text", dir)
      ))
    )
  }
})

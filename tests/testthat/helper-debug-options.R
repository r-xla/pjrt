# The DebugOptions fields that differ from XLA's built-in defaults in a compilation of
# `program` with `xla_options`, as XLA dumps them: one "name: value" string per field, or per
# entry of a repeated or map field. `xla_options` must not set xla_dump_to.
non_default_debug_options <- function(program, xla_options = list()) {
  dir <- withr::local_tempdir()
  pjrt::pjrt_compile(program, xla_options = c(list(xla_dump_to = dir), xla_options))
  lines <- readLines(list.files(dir, pattern = "\\.debug_options$", full.names = TRUE))
  # A map entry spans several lines: "name: {", its indented fields, "}".
  starts <- !grepl("^(\\s|\\})", lines)
  entries <- vapply(split(trimws(lines), cumsum(starts)), paste, character(1L), collapse = " ")
  unname(entries[!startsWith(entries, "xla_dump_to:")])
}

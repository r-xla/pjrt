# Parse the HLO dump files produced by a single compilation into a PJRTHLODump.
# `files` are paths relative to `dir`.
parse_hlo_dump <- function(dir, files) {
  txt <- files[grepl("\\.txt$", files)]
  module <- NA_character_
  if (length(txt)) {
    module <- sub("^module_\\d+\\.([^.]+)\\..*$", "\\1", txt[[1]])
  }

  read_file <- function(f) {
    paste(readLines(file.path(dir, f), warn = FALSE), collapse = "\n")
  }

  # The optimized file is backend-prefixed (e.g. cpu_after_optimizations.txt);
  # match on the suffix. Sibling reports end differently (-buffer-assignment.txt).
  before <- txt[grepl("before_optimizations\\.txt$", txt)]
  after <- txt[grepl("after_optimizations\\.txt$", txt)]

  stages <- list()
  if (length(before)) {
    stages[["before_optimizations"]] <- read_file(before[[1]])
  }
  if (length(after)) {
    stages[["after_optimizations"]] <- read_file(sort(after)[[1]])
  }

  structure(stages, class = "PJRTHLODump", dir = dir, module = module)
}

# If the running session's XLA_FLAGS enables text HLO dumping, return the
# directory XLA dumps into; otherwise NULL. NB: XLA_FLAGS is only actually
# honoured if it was set before the session's first compilation (XLA parses it
# once); the caller confirms by checking that new files appear.
session_dump_dir <- function() {
  xla_flags <- Sys.getenv("XLA_FLAGS")
  if (!nzchar(xla_flags)) {
    return(NULL)
  }
  m <- regmatches(xla_flags, regexpr("--xla_dump_to=\\S+", xla_flags))
  if (!length(m) || !grepl("--xla_dump_hlo_as_text", xla_flags, fixed = TRUE)) {
    return(NULL)
  }
  sub("--xla_dump_to=", "", m)
}

#' Inspect the HLO intermediate representations of a program
#'
#' Compiles `program` and returns the HLO intermediate representations the XLA
#' compiler produced: the input HLO (`before_optimizations`) and the optimized
#' HLO (`after_optimizations`).
#'
#' @section Enabling HLO dumping:
#'
#' Dumping is driven by the XLA compiler's own dump mechanism, enabled through
#' the `XLA_FLAGS` environment variable. XLA reads `XLA_FLAGS` **once, before the
#' first compilation in an R process**, so it must be set at the very start of a
#' fresh session, before any [pjrt_compile()] or [pjrt_execute()] call:
#'
#' ```r
#' Sys.setenv(XLA_FLAGS = "--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text")
#' library(pjrt)
#' # ... build a program ...
#' pjrt_dump_hlo(prog)
#' ```
#'
#' `pjrt_dump_hlo()` compiles `program` in-process and reads the files XLA writes
#' into the `--xla_dump_to` directory. If the required flags are absent, or were
#' set too late to take effect, it raises an error explaining what to set rather
#' than returning nothing.
#'
#' @param program (`PJRTProgram`)\cr The program to inspect.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr Device or platform to
#'   compile for, as in [pjrt_compile()].
#'
#' @return A `PJRTHLODump` object: a named list of HLO text keyed by stage
#'   (`before_optimizations`, `after_optimizations`). Use `[[` to extract a stage
#'   (e.g. `dump[["after_optimizations"]]` for the optimized HLO). Carries the
#'   dump directory and module name as attributes.
#' @export
pjrt_dump_hlo <- function(program, device = NULL) {
  check_program(program)

  device_spec <- if (inherits(device, "PJRTDevice")) {
    as.character(device)
  } else {
    device
  }

  dir <- session_dump_dir()
  if (is.null(dir)) {
    cli_abort(
      c(
        "HLO dumping is not enabled in this session.",
        "i" = "Set {.envvar XLA_FLAGS} to enable text HLO dumping {.strong before
               the first compilation}, then start a fresh R session:",
        "*" = '{.code Sys.setenv(XLA_FLAGS = "--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text")}'
      ),
      call = NULL
    )
  }

  before <- list.files(dir)
  pjrt_compile(program, device = device_spec)
  new_files <- setdiff(list.files(dir), before)
  dump <- parse_hlo_dump(dir, new_files)

  if (is.null(dump[["after_optimizations"]])) {
    cli_abort(
      c(
        "The compiler dumped no HLO for this program.",
        "i" = "{.envvar XLA_FLAGS} is read only {.strong once}, before the first
               compilation in an R process -- it looks like something was compiled
               in this session before the dump flags took effect.",
        "i" = "Set {.envvar XLA_FLAGS} at the very start of a fresh R session,
               before any {.fn pjrt_compile} or {.fn pjrt_execute} call."
      ),
      call = NULL
    )
  }

  dump
}

#' @export
format.PJRTHLODump <- function(x, ...) {
  n_lines <- vapply(
    x,
    function(s) length(strsplit(s, "\n", fixed = TRUE)[[1]]),
    integer(1)
  )
  header <- sprintf(
    "<PJRTHLODump: module '%s', %d stage(s)>",
    attr(x, "module") %||% "?",
    length(x)
  )
  c(header, sprintf("  %s (%d lines)", names(x), n_lines))
}

#' @export
print.PJRTHLODump <- function(x, ...) {
  cat(format(x), sep = "\n")
  cat("\n")
  invisible(x)
}

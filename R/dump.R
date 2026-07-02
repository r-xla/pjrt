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

  # Per-pass dumps carry a zero-padded `.NNNN.` sequence index; the input and
  # optimized snapshots do not.
  pass_files <- txt[grepl("\\.[0-9]{4}\\.", txt)]
  main_files <- setdiff(txt, pass_files)

  # The optimized file is backend-prefixed (e.g. cpu_after_optimizations.txt);
  # match on the suffix. Sibling reports end differently (-buffer-assignment.txt).
  before <- main_files[grepl("before_optimizations\\.txt$", main_files)]
  after <- main_files[grepl("after_optimizations\\.txt$", main_files)]

  stages <- list()
  if (length(before)) {
    stages[["before_optimizations"]] <- read_file(before[[1]])
  }
  if (length(pass_files)) {
    idx <- as.integer(regmatches(
      pass_files,
      regexpr("(?<=\\.)[0-9]{4}(?=\\.)", pass_files, perl = TRUE)
    ))
    pass_files <- pass_files[order(idx)]
    keys <- sub("\\.txt$", "", sub("^module_\\d+\\.[^.]+\\.", "", pass_files))
    for (i in seq_along(pass_files)) {
      stages[[keys[[i]]]] <- read_file(pass_files[[i]])
    }
  }
  if (length(after)) {
    stages[["after_optimizations"]] <- read_file(sort(after)[[1]])
  }

  structure(stages, class = "PJRTHLODump", dir = dir, module = module)
}

# Compile `progfile` (a program dumped to disk) in a fresh R process with
# XLA_FLAGS enabling HLO dumping into `dir`. A subprocess is used because XLA
# reads XLA_FLAGS once, before the first compilation in a process -- so an
# in-process attempt would silently do nothing once anything else has compiled.
run_dump_subprocess <- function(progfile, format, device, dir, passes) {
  flags <- sprintf("--xla_dump_to=%s --xla_dump_hlo_as_text", dir)
  if (passes) {
    flags <- paste(flags, "--xla_dump_hlo_pass_re=.*")
  }
  # Reproduce the current environment (so the child finds the plugin, platform,
  # libraries) but force XLA_FLAGS and drop R_TESTS (which breaks nested R).
  child_env <- Sys.getenv()
  child_env <- child_env[names(child_env) != "R_TESTS"]
  child_env[["XLA_FLAGS"]] <- flags

  callr::r(
    function(progfile, format, device) {
      library(pjrt)
      prog <- pjrt_program(path = progfile, format = format)
      pjrt_compile(prog, device = device)
      invisible(NULL)
    },
    args = list(progfile = progfile, format = format, device = device),
    env = child_env
  )
}

#' Inspect the HLO intermediate representations of a program
#'
#' Compiles `program` and returns the HLO intermediate representations the XLA
#' compiler produced: the input HLO (`before_optimizations`) and the optimized
#' HLO (`after_optimizations`) by default, and one entry per compiler pass when
#' `passes = TRUE`.
#'
#' The compilation is run in a fresh R subprocess. This is necessary because
#' dumping is enabled via the `XLA_FLAGS` environment variable, which XLA reads
#' only once, before the first compilation in a process. Running in a subprocess
#' makes `pjrt_dump_hlo()` work reliably regardless of what has already been
#' compiled or executed in the current session, at the cost of a one-off process
#' startup (roughly a second).
#'
#' @param program (`PJRTProgram`)\cr The program to inspect.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr Device or platform to
#'   compile for, as in [pjrt_compile()].
#' @param passes (`logical(1)`)\cr If `TRUE`, additionally dump the HLO after
#'   every compiler pass.
#'
#' @return A `PJRTHLODump` object: a named list of HLO text keyed by stage
#'   (`before_optimizations`, `after_optimizations`, and one entry per pass when
#'   `passes = TRUE`). Use `[[` to extract a stage and [as.character()] for the
#'   optimized HLO. Carries the dump directory and module name as attributes.
#' @export
pjrt_dump_hlo <- function(program, device = NULL, passes = FALSE) {
  check_program(program)
  if (!is.logical(passes) || length(passes) != 1L || is.na(passes)) {
    stop("`passes` must be a single TRUE/FALSE value.")
  }

  progfile <- tempfile("pjrt_prog_")
  writeBin(impl_program_code(program), progfile)
  on.exit(unlink(progfile), add = TRUE)

  dir <- tempfile("pjrt_hlo_")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  device_spec <- if (inherits(device, "PJRTDevice")) {
    as.character(device)
  } else {
    device
  }

  run_dump_subprocess(
    progfile,
    impl_program_format(program),
    device_spec,
    dir,
    passes
  )

  dump <- parse_hlo_dump(dir, list.files(dir))
  if (is.null(dump[["after_optimizations"]])) {
    stop(
      "The XLA compiler produced no HLO dump; the backend may not support ",
      "HLO dumping.",
      call. = FALSE
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

#' @export
as.character.PJRTHLODump <- function(x, ...) {
  x[["after_optimizations"]] %||% NA_character_
}

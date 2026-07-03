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

# The XLA flags that enable text HLO dumping into `dir` (plus per-pass dumps
# when `passes`). These are what both the in-process and subprocess paths need.
dump_flags <- function(dir, passes) {
  flags <- c(
    sprintf("--xla_dump_to=%s", dir),
    "--xla_dump_hlo_as_text"
  )
  if (passes) {
    flags <- c(flags, "--xla_dump_hlo_pass_re=.*")
  }
  flags
}

# If the running session's XLA_FLAGS already enables text HLO dumping (and, when
# `passes`, per-pass dumping), return the directory XLA dumps into -- so the
# caller can compile in-process instead of spawning a subprocess. Returns NULL
# when the required flags are absent. NB: this only actually dumps if those
# flags were set before the session's first compilation (XLA parses XLA_FLAGS
# once); the caller confirms by checking that new files appear.
session_dump_dir <- function(passes) {
  xla_flags <- Sys.getenv("XLA_FLAGS")
  if (!nzchar(xla_flags)) {
    return(NULL)
  }
  m <- regmatches(xla_flags, regexpr("--xla_dump_to=\\S+", xla_flags))
  if (!length(m) || !grepl("--xla_dump_hlo_as_text", xla_flags, fixed = TRUE)) {
    return(NULL)
  }
  if (passes && !grepl("--xla_dump_hlo_pass_re", xla_flags, fixed = TRUE)) {
    return(NULL)
  }
  sub("--xla_dump_to=", "", m)
}

# Compile `progfile` (a program dumped to disk) in a fresh R process with
# XLA_FLAGS enabling HLO dumping into `dir`. A subprocess is used because XLA
# reads XLA_FLAGS once, before the first compilation in a process -- so an
# in-process attempt would silently do nothing once anything else has compiled.
run_dump_subprocess <- function(progfile, format, device, dir, passes, flags) {
  # Preserve any XLA_FLAGS already set in the session plus the user's `flags`,
  # then append our dump flags LAST so that --xla_dump_to wins (XLA honours the
  # last occurrence of a flag) and points at the directory we read back.
  existing <- Sys.getenv("XLA_FLAGS")
  combined <- paste(
    c(if (nzchar(existing)) existing, flags, dump_flags(dir, passes)),
    collapse = " "
  )

  # Reproduce the current environment (so the child finds the plugin, platform,
  # libraries) but force XLA_FLAGS and drop R_TESTS (which breaks nested R).
  child_env <- Sys.getenv()
  child_env <- child_env[names(child_env) != "R_TESTS"]
  child_env[["XLA_FLAGS"]] <- combined

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
#' @section Subprocess vs. in-process:
#'
#' HLO dumping is enabled via the `XLA_FLAGS` environment variable, which XLA
#' reads only **once, before the first compilation in a process**. So that
#' `pjrt_dump_hlo()` works regardless of what has already been compiled or
#' executed in the session, it compiles `program` in a fresh R subprocess with
#' `XLA_FLAGS` set from the start. This costs a one-off process startup and
#' plugin load (roughly a second) per call.
#'
#' To inspect many programs in a session without that per-call cost, set the
#' dump flags in `XLA_FLAGS` **before your first compilation**. `pjrt_dump_hlo()`
#' then detects them and compiles in-process, skipping the subprocess:
#'
#' ```r
#' # Before any pjrt_compile()/pjrt_execute() in the session:
#' Sys.setenv(XLA_FLAGS = paste(
#'   "--xla_dump_to=/tmp/hlo",
#'   "--xla_dump_hlo_as_text",
#'   "--xla_dump_hlo_pass_re=.*" # only needed for passes = TRUE
#' ))
#' # ... then, as often as you like, with no per-call subprocess cost:
#' pjrt_dump_hlo(prog1)
#' pjrt_dump_hlo(prog2)
#' ```
#'
#' The in-process fast path is taken only when no extra `flags` are requested and
#' the flags are actually active (set before the first compile); otherwise
#' `pjrt_dump_hlo()` transparently falls back to the subprocess.
#'
#' @param program (`PJRTProgram`)\cr The program to inspect.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr Device or platform to
#'   compile for, as in [pjrt_compile()].
#' @param passes (`logical(1)`)\cr If `TRUE`, additionally dump the HLO after
#'   every compiler pass.
#' @param flags (`character()`)\cr Additional XLA compiler flags for the dump
#'   compilation (the same strings you would put in `XLA_FLAGS`, e.g.
#'   `"--xla_dump_hlo_as_proto"`). Appended to any `XLA_FLAGS` already set in the
#'   session. The dump destination is controlled by `pjrt_dump_hlo()`, so a
#'   `--xla_dump_to` here is ignored. Supplying any `flags` forces the subprocess
#'   path.
#'
#' @return A `PJRTHLODump` object: a named list of HLO text keyed by stage
#'   (`before_optimizations`, `after_optimizations`, and one entry per pass when
#'   `passes = TRUE`). Use `[[` to extract a stage and [as.character()] for the
#'   optimized HLO. Carries the dump directory and module name as attributes.
#' @export
pjrt_dump_hlo <- function(program, device = NULL, passes = FALSE, flags = character()) {
  check_program(program)
  checkmate::assert_flag(passes)
  checkmate::assert_character(flags, any.missing = FALSE)

  device_spec <- if (inherits(device, "PJRTDevice")) {
    as.character(device)
  } else {
    device
  }

  # Fast path: if the session was configured (before its first compile) to dump
  # text HLO, compile in-process and read the freshly written files -- avoiding
  # a subprocess + plugin reload. Only when no extra `flags` are requested,
  # since those cannot be applied to an already-initialised process.
  if (length(flags) == 0L) {
    dir <- session_dump_dir(passes)
    if (!is.null(dir)) {
      before <- list.files(dir)
      pjrt_compile(program, device = device_spec)
      new_files <- setdiff(list.files(dir), before)
      dump <- parse_hlo_dump(dir, new_files)
      if (!is.null(dump[["after_optimizations"]])) {
        return(dump)
      }
      # Flags were not actually active (set after the first compile) -- fall
      # through to the subprocess, which sets XLA_FLAGS from the start.
    }
  }

  progfile <- tempfile("pjrt_prog_")
  writeBin(impl_program_code(program), progfile)
  on.exit(unlink(progfile), add = TRUE)

  dir <- tempfile("pjrt_hlo_")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  run_dump_subprocess(
    progfile,
    impl_program_format(program),
    device_spec,
    dir,
    passes,
    flags
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

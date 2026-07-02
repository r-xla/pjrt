# Extract the `--xla_dump_to=<dir>` directory from an XLA_FLAGS string, or NULL.
xla_dump_dir_from_flags <- function(flags = Sys.getenv("XLA_FLAGS")) {
  m <- regmatches(flags, regexpr("--xla_dump_to=\\S+", flags))
  if (length(m) == 0L || !nzchar(m)) {
    return(NULL)
  }
  sub("--xla_dump_to=", "", m, fixed = TRUE)
}

# Parse the HLO dump files produced by a single compilation into a PJRTHLODump.
# `files` are paths relative to `dir` (typically the files newly created by the
# compilation).
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

dump_missing_msg <- function(dir) {
  paste0(
    "No HLO was dumped to '",
    dir,
    "'.\n",
    "XLA reads XLA_FLAGS once, before the first compilation in the process. ",
    "For reliable dumping, set e.g.\n",
    "  XLA_FLAGS=\"--xla_dump_to=/path --xla_dump_hlo_as_text\"\n",
    "before starting R, or call pjrt_dump_hlo() before any other compilation."
  )
}

#' Inspect the HLO intermediate representations of a program
#'
#' Compiles `program` and returns the HLO intermediate representations the XLA
#' compiler produced: the input HLO (`before_optimizations`) and the optimized
#' HLO (`after_optimizations`) by default, and one entry per compiler pass when
#' `passes = TRUE`.
#'
#' Dumping is performed by the XLA compiler itself and controlled through the
#' `XLA_FLAGS` environment variable (`--xla_dump_to`). XLA reads `XLA_FLAGS`
#' **once, before the first compilation** in the process. Therefore:
#'
#' * If `XLA_FLAGS=--xla_dump_to=...` is already set (ideally before R starts),
#'   `pjrt_dump_hlo()` reads back what the compiler dumps into that directory.
#' * Otherwise `pjrt_dump_hlo()` sets `XLA_FLAGS` to a temporary directory, which
#'   only takes effect if this is the first compilation in the session.
#'
#' If no IR is produced (because a different compilation already fixed
#' `XLA_FLAGS`), an error explains how to enable dumping.
#'
#' @param program (`PJRTProgram`)\cr The program to inspect.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr Device or platform to
#'   compile for, as in [pjrt_compile()].
#' @param passes (`logical(1)`)\cr If `TRUE`, additionally dump the HLO after
#'   every compiler pass. Requires `--xla_dump_hlo_pass_re` to be part of
#'   `XLA_FLAGS` when the directory is taken from a pre-set `XLA_FLAGS`.
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

  flags <- Sys.getenv("XLA_FLAGS")
  dir <- xla_dump_dir_from_flags(flags)
  if (is.null(dir)) {
    dir <- tempfile("pjrt_hlo_")
    added <- sprintf("--xla_dump_to=%s --xla_dump_hlo_as_text", dir)
    if (passes) {
      added <- paste(added, "--xla_dump_hlo_pass_re=.*")
    }
    Sys.setenv(XLA_FLAGS = trimws(paste(flags, added)))
  } else if (passes && !grepl("--xla_dump_hlo_pass_re", flags)) {
    warning(
      "`passes = TRUE` but XLA_FLAGS lacks --xla_dump_hlo_pass_re; ",
      "per-pass dumps require it to be set before the first compilation."
    )
  }
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  before <- list.files(dir)
  pjrt_compile(program, device = device)
  new_files <- setdiff(list.files(dir), before)

  dump <- parse_hlo_dump(dir, new_files)
  if (is.null(dump[["after_optimizations"]])) {
    stop(dump_missing_msg(dir), call. = FALSE)
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

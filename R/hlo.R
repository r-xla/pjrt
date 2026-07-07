dump_mtimes <- function(dir) {
  files <- list.files(dir)
  mtimes <- file.mtime(file.path(dir, files))
  names(mtimes) <- files
  mtimes
}

parse_hlo_dump <- function(dir, files) {
  txt <- files[grepl("\\.txt$", files)]
  module <- NA_character_
  if (length(txt)) {
    module <- sub("^module_\\d+\\.([^.]+)\\..*$", "\\1", txt[[1]])
  }

  read_file <- function(f) {
    path <- file.path(dir, f)
    readChar(path, file.size(path))
  }

  before <- txt[grepl("before_optimizations\\.txt$", txt)]
  after <- txt[grepl("after_optimizations\\.txt$", txt)]

  stages <- list()
  if (length(before)) {
    stages[["before_optimizations"]] <- new_hlo_module(
      read_file(before[[1]]),
      "PJRTHloBeforeOptimizations"
    )
  }
  if (length(after)) {
    stages[["after_optimizations"]] <- new_hlo_module(
      read_file(sort(after)[[1]]),
      "PJRTHloAfterOptimizations"
    )
  }

  structure(stages, class = "PJRTHloModuleSrcs", dir = dir, module = module)
}

new_hlo_module <- function(text, subclass) {
  structure(text, class = c(subclass, "PJRTHloModuleSrc"))
}

#' @export
print.PJRTHloModuleSrc <- function(x, ...) {
  cat(unclass(x))
  if (!endsWith(x, "\n")) {
    cat("\n")
  }
  invisible(x)
}

# If xla_dump_to is not set, the Hlo is written to Stdout and we can't retrieve it
session_dump_dir <- function() {
  xla_flags <- Sys.getenv("XLA_FLAGS")
  if (!nzchar(xla_flags)) {
    return(NULL)
  }
  # Need both the text-format flag and a --xla_dump_to directory to read from.
  m <- regmatches(xla_flags, regexpr("--xla_dump_to=\\S+", xla_flags))
  if (!length(m) || !grepl("--xla_dump_hlo_as_text", xla_flags, fixed = TRUE)) {
    return(NULL)
  }
  sub("--xla_dump_to=", "", m)
}

#' Inspect the HLO Source of a Program
#'
#' The result contains the program before and after optimization.
#'
#' @section Enabling HLO dumping:
#'
#' Dumping is driven by the XLA compiler's own dump mechanism, enabled through
#' the `XLA_FLAGS` environment variable. **Two** flags are required:
#'
#' * `--xla_dump_hlo_as_text` -- dump the HLO in text form.
#' * `--xla_dump_to=<dir>` -- write the dump into `<dir>`. `inspect_hlo()` reads
#'   the HLO back from the files XLA writes here.
#'
#' XLA reads `XLA_FLAGS` **once, before the first compilation in an R process**,
#' so both flags must be set at the very start of a fresh session, before any
#' [pjrt_compile()] or [pjrt_execute()] call:
#'
#' ```r
#' Sys.setenv(XLA_FLAGS = "--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text")
#' library(pjrt)
#' # ... build a program ...
#' inspect_hlo(prog)
#' ```
#'
#' @param program (`PJRTProgram`)\cr The program to inspect.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr Device to compile for.
#'
#' @return A `PJRTHloModuleSrcs`
#' @export
inspect_hlo <- function(program, device = NULL) {
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
        "i" = "Both {.code --xla_dump_hlo_as_text} (dump as text) and
               {.code --xla_dump_to=<dir>} (a directory to read the HLO back from)
               must be set in {.envvar XLA_FLAGS} {.strong before the first
               compilation}. Start a fresh R session with:",
        "*" = '{.code Sys.setenv(XLA_FLAGS = "--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text")}'
      ),
      call = NULL
    )
  }

  # Detect the files this compile produces by mtime, not by name: XLA numbers HLO
  # modules per process from 0, so a reused dump dir already holds identically
  # named files from a previous session that this compile overwrites in place. A
  # plain name diff would see nothing new; comparing mtimes catches the rewrite.
  before <- dump_mtimes(dir)
  pjrt_compile(program, device = device_spec)
  after <- dump_mtimes(dir)
  new_files <- names(after)[vapply(
    names(after),
    function(f) is.na(before[f]) || after[[f]] > before[[f]],
    logical(1)
  )]
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
format.PJRTHloModuleSrcs <- function(x, ...) {
  n_lines <- vapply(
    x,
    function(s) length(strsplit(s, "\n", fixed = TRUE)[[1]]),
    integer(1)
  )
  header <- sprintf(
    "<PJRTHloModuleSrcs: module '%s', %d stage(s)>",
    attr(x, "module") %||% "?",
    length(x)
  )
  c(header, sprintf("  %s (%d lines)", names(x), n_lines))
}

#' @export
print.PJRTHloModuleSrcs <- function(x, ...) {
  cat(format(x), sep = "\n")
  cat("\n")
  invisible(x)
}

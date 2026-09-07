pjrt_debug <- function(..., .envir = parent.frame()) {
  if (nzchar(Sys.getenv("PJRT_DEBUG", ""))) {
    cli::cli_inform(..., .envir = .envir)
  }
}

get_dims <- function(data) {
  if (is.null(dim(data))) {
    if (length(data) == 1) {
      return(1L)
    } else if (length(data) == 0) {
      return(integer())
    } else {
      return(length(data))
    }
  }
  dim(data)
}

default_platform <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu")
}

# Ask the user for permission before downloading a large binary artifact
# (a PJRT plugin or the stablehlo-opt tool), mirroring the behaviour of torch's
# auto-install prompt. The `PJRT_INSTALL` environment variable overrides the
# prompt:
#   - PJRT_INSTALL=1  download without asking (e.g. CI, scripts, Docker builds)
#   - PJRT_INSTALL=0  never download; abort with instructions instead
# When `PJRT_INSTALL` is unset we ask in an interactive session and abort in a
# non-interactive one (where there is no terminal to ask on), so a batch job or
# script never triggers a surprise download. This is never reached during
# `R CMD check` because examples are guarded behind `plugins_downloaded()` /
# `stablehlo_opt_available()` and the test suite only runs when `PJRT_TEST=1`
# (see tests/testthat.R).
#
# `what` and `override` are inserted verbatim into the messages, so callers
# that want cli markup there have to pre-format it with `cli::format_inline()`.
confirm_install <- function(what, url, dest, override) {
  install <- Sys.getenv("PJRT_INSTALL", unset = "")

  if (install == "1") {
    return(invisible(TRUE))
  }

  if (install == "0") {
    cli_abort(c(
      "{what} needs to be downloaded but automatic downloads are disabled.",
      i = "{.envvar PJRT_INSTALL} is set to {.val 0}.",
      i = "Set {.envvar PJRT_INSTALL} to {.val 1} to allow the download, or set {override}."
    ))
  }

  # PJRT_INSTALL unset: only download if we can ask and the user agrees.
  if (!interactive()) {
    cli_abort(c(
      "{what} needs to be downloaded for this to work.",
      i = "Automatic downloads are not performed in non-interactive sessions.",
      i = "Set {.envvar PJRT_INSTALL} to {.val 1} to allow the download, or set {override}."
    ))
  }

  cli::cli_inform(c(
    "{what} needs to be {.strong downloaded}.",
    i = "It will be downloaded from {.url {url}} and cached in {.path {dest}}.",
    i = "Set {.envvar PJRT_INSTALL} to {.val 1} to skip this prompt in the future."
  ))
  response <- utils::askYesNo("Do you want to download it now?")
  if (is.na(response) || !response) {
    cli_abort("Download of {what} was declined.")
  }

  invisible(TRUE)
}

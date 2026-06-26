#' @title Install PJRT Plugins
#' @description
#' Download and cache the PJRT plugins needed to run `pjrt`.
#' The CPU plugin is always installed. The CUDA plugin is installed in
#' addition when a CUDA-capable GPU is detected, or when `cuda = TRUE` is
#' passed explicitly.
#'
#' Plugins are otherwise downloaded lazily the first time a client is
#' created. Calling `install_pjrt()` performs this download eagerly, for
#' example to warm a cache during a Docker build or before going offline.
#'
#' @details
#' The call to `install_pjrt()` is treated as consent to download, so the
#' interactive confirmation prompt (and the `PJRT_INSTALL` environment
#' variable that controls it) is bypassed.
#'
#' @param cuda (`logical(1)` | `NULL`)\cr
#'   Whether to also install the CUDA plugin. When `NULL` (the default),
#'   CUDA support is auto-detected: the CUDA plugin is installed when an
#'   NVIDIA GPU is available on a Linux x86_64 machine.
#' @return (`character()`)\cr
#'   The platforms that were installed, invisibly.
#' @examplesIf interactive()
#' install_pjrt()
#' @export
install_pjrt <- function(cuda = NULL) {
  if (is.null(cuda)) {
    cuda <- cuda_available()
  }
  checkmate::assert_flag(cuda)

  platforms <- if (cuda) c("cpu", "cuda") else "cpu"

  withr::local_envvar(PJRT_INSTALL = "1")

  for (platform in platforms) {
    cli::cli_inform("Installing the {.val {platform}} PJRT plugin.")
    plugin_path(platform)
  }

  cli::cli_inform(c(v = "Installed PJRT plugin{?s}: {.val {platforms}}."))
  invisible(platforms)
}

# Detect whether a CUDA-capable GPU is usable on this machine. The CUDA PJRT
# plugin only ships for Linux x86_64, so we short-circuit on other platforms
# before shelling out to `nvidia-smi` to confirm a GPU is actually present.
cuda_available <- function() {
  if (plugin_os() != "linux" || plugin_arch() != "amd64") {
    return(FALSE)
  }

  nvidia_smi <- Sys.which("nvidia-smi")
  if (nvidia_smi == "") {
    return(FALSE)
  }

  status <- tryCatch(
    suppressWarnings(system2(nvidia_smi, stdout = FALSE, stderr = FALSE)),
    error = function(e) 1L
  )
  identical(status, 0L)
}

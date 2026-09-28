#' @title Platform Support
#' @description
#' Which PJRT backends are available on which operating system and CPU
#' architecture.
#'
#' The possible support levels are:
#' * `"yes"`: The plugin is available and fully supported.
#' * `"wsl2"`: Only available via the Windows Subsystem for Linux (WSL2),
#'   i.e. by running R inside Linux.
#' * `"no"`: No plugin is available.
#'
#' @return (`data.frame()`)\cr
#'   One row per operating system and architecture, with character columns
#'   `os`, `arch`, and one column per backend (`cpu`, `cuda`)
#'   holding the support level.
#' @examples
#' platform_support()
#' @export
platform_support <- function() {
  data.frame(
    os = c("Linux", "Linux", "Windows", "Windows", "macOS", "macOS"),
    arch = c("x86_64", "arm64", "x86_64", "arm64", "x86_64", "arm64"),
    cpu = c("yes", "yes", "yes", "no", "yes", "yes"),
    cuda = c("yes", "yes", "wsl2", "no", "no", "no")
  )
}

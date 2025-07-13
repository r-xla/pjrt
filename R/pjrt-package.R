#' R interface to PJRT
#'
#' The package provides an R interface to PJRT, which allows you to run XLA or stableHLO
#' programs on a variety of hardware backends.
#'
#' @section Package Configuration:
#'
#' The package uses the following environment variables to configure its behavior:
#'
#' - `PJRT_PLUGIN_URL_<PLATFORM>`: The URL to download the PJRT plugin for the given platform.
#'   The platform can e.g. be CPU, CUDA, or METAL.
#' - `PJRT_ZML_ARTIFACT_VERSION`: The version of the PJRT artifacts to use.
#'
NULL
"_PACKAGE"

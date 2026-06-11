## usethis namespace: start
#' @useDynLib pjrt, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom tengen as_array as_dtype as_raw device dtype shape
#' @import checkmate
#' @importFrom safetensors safe_tensor_buffer safe_tensor_meta
#' @importFrom utils hashtab
#' @importFrom cli cli_abort
#' @importFrom bit64 integer64
## usethis namespace: end
NULL

#' @section Environment Variables:
#' **Configuration options provided by XLA**
#'
#' XLA provides various configuration options, but their documentation is scattered across
#' various websites. The options include:
#' * `TF_CPP_MIN_LOG_LEVEL`: Logging level for PJRT C++ API:
#'   * 0: shows info, warnings and errors
#'   * 1: shows warnings and errors
#'   * 2: shows errors
#'   * 3: shows nothing
#' * `XLA_FLAGS`: See the [openxla website](https://openxla.org/xla/flags_guidance) for
#'   more information.
#'
#' **Configuration options provided by this package**
#'
#' * `PJRT_PLATFORM`: Default platform to use, falls back to `"cpu"`.
#' * `PJRT_PLUGIN_PATH_<PLATFORM>`: Path to custom plugin library file for a specific
#'   platform (e.g., `PJRT_PLUGIN_PATH_CPU`, `PJRT_PLUGIN_PATH_CUDA`,
#'   `PJRT_PLUGIN_PATH_METAL`). If set, the package will use this path instead
#'   of downloading the plugin.
#' * `PJRT_PLUGIN_URL_<PLATFORM>`: URL to download plugin from for a specific
#'   platform (e.g., `PJRT_PLUGIN_URL_CPU`, `PJRT_PLUGIN_URL_CUDA`,
#'   `PJRT_PLUGIN_URL_METAL`). If set, overrides the default plugin download URL.
#' * `PJRT_INSTALL`: Controls whether plugins may be downloaded automatically.
#'   Set this to `"1"` to always download without asking (e.g. in CI, scripts,
#'   or Docker builds), or to `"0"` to never download (the call errors with
#'   instructions instead). When unset, the package asks for confirmation in an
#'   interactive session and errors in a non-interactive one, so a script never
#'   triggers a surprise download.
#' * `PJRT_ZML_ARTIFACT_VERSION`: Version of ZML artifacts to download.
#'   Only used when downloading plugins from zml/pjrt-artifacts.
#' * `PJRT_CPU_DEVICE_COUNT`: The number of CPU devices to use. Defaults to 1.
#'    This is primarily intended for testing purposes.
#' * `PJRT_CUDA_R_PACKAGE`: Name of the R package providing CUDA libraries.
#'   Defaults to the value of `r cuda_r_package()`.
#'   Set this to use a different CUDA toolkit package, but note that other
#'   versions may not work with the XLA plugin.
#'
#' * `PJRT_DEBUG`: If set (to any non-empty value), enables verbose debug output
#'   via `cli::cli_inform()`.
#'
#' @section Third-Party Licenses:
#' The `pjrt` package itself is MIT-licensed. The CUDA backend dynamically
#' loads NVIDIA software which is not bundled with `pjrt`, but downloaded
#' from NVIDIA's official redistributable channels by the CUDA toolkit R
#' package (e.g. `cuda12.8`) at install time. Its use is governed by the
#' [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), with the
#' exception of cuDNN, which is covered by the
#' [NVIDIA cuDNN SLA](https://docs.nvidia.com/deeplearning/cudnn/sla/index.html),
#' and NCCL, which is covered by its [own license](https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt).
#' By installing or using the CUDA backend you accept those terms.
"_PACKAGE"

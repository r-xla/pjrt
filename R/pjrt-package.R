#' @section Environment Variables:
#' * `TF_CPP_MIN_LOG_LEVEL`: Logging level for PJRT C++ API:
#'   * 0: shows info, warnings and errors
#'   * 1: shows warnings and errors
#'   * 2: shows errors
#'   * 3: shows nothing
#' * `PJRT_PLATFORM`: Default platform to use, falls back to `"cpu"`.
#' * `PJRT_CPU_DEVICE_COUNT`: The number of CPU devices to use. Defaults to 1.
"_PACKAGE"

## usethis namespace: start
#' @importFrom tengen as_array
#' @importFrom tengen as_raw
#' @importFrom tengen device
#' @importFrom tengen dtype
#' @importFrom tengen shape
#' @import checkmate
#' @importFrom safetensors safe_tensor_buffer safe_tensor_meta
#' @importFrom S7 method<- new_generic class_logical class_raw class_integer class_double
#' @importFrom tengen shape dtype
#' @importFrom utils hashtab
#' @importFrom cli cli_abort
## usethis namespace: end
NULL

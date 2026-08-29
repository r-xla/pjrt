#' @title Are pjrt's CUDA Kernels Available?
#' @description
#' Whether this installation of `pjrt` carries the CUDA kernels it ships in
#' `src/cuda/`. They are compiled during installation, which needs `nvcc` --
#' from a CUDA toolkit on the machine, or from the `cuda12.8` R package. When
#' none was found the package still installs and everything else still works,
#' but the custom calls backed by those kernels have no CUDA implementation.
#'
#' Downstream packages can use this to pick a fallback lowering rather than
#' emit a custom call that would fail once it reaches the device.
#'
#' This says nothing about whether a GPU is present: it is a fact about how the
#' package was built, fixed at install time.
#'
#' @return `logical(1)`
#' @examples
#' pjrt_cuda_kernels_available()
#' @export
pjrt_cuda_kernels_available <- function() {
  impl_cuda_kernels_available()
}

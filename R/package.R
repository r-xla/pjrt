## usethis namespace: start
#' @useDynLib pjrt, .registration = TRUE
## usethis namespace: end
NULL

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL


.onLoad <- function(libname, pkgname) {
  # this allows for tests without as_array() conversion
  register_s3_method("waldo", "compare_proxy", "PJRTBuffer")
}

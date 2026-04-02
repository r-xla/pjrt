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

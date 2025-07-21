get_dims <- function(data) {
  if (is.null(dim(data))) {
    if (length(data) == 1) {
      return(integer())
    }
    return(length(data))
  }
  dim(data)
}

default_platform <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu")
}

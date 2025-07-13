get_dims <- function(data) {
  if (is.null(dim(data))) {
    if (length(data) == 1) {
      return(integer())
    }
    return(length(data))
  }
  dim(data)
}

default_client <- function(platform_name = NULL) {
  get_client(platform_name)
}

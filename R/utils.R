get_dims <- function(data) {
  if (is.null(dim(data))) {
    return(length(data))
  }
  dim(data)
}

default_client <- function() {
  plugin_client_create(plugin_load())
}

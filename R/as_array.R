#' @export
as_array <- function(x, ..., client = default_client()) {
  client_buffer_to_host(x, client = client, ...)
}

# TODO: Better name? This might return a vector if it is a PJRT array
# with no dimension ()
as_array <- function(x) {
  client_buffer_to_host(x)
}

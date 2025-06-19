client_program_compile <- function(
  client,
  program,
  compile_options = new_compile_options()
) {
  check_client(client)
  check_program(program)
  check_compile_options(compile_options)

  impl_client_program_compile(client, program, compile_options)
}

client_scalar_buffer_from_host <- function(client, data) {
  check_client(client)
  impl_client_scalar_buffer_from_host(client, data)
}

client_buffer_from_host <- function(client, data) {
  check_client(client)
  impl_client_buffer_from_host(client, data)
}

client_buffer_to_host <- function(client, buffer) {
  check_client(client)
  check_buffer(buffer)

  impl_client_buffer_to_host(client, buffer)
}

client_platform_name <- function(client) {
  check_client(client)
  tolower(impl_client_platform_name(client))
}

check_client <- function(client) {
  stopifnot(inherits(client, "PJRTClient"))
  invisible(NULL)
}

new_compile_options <- function(
  build_options = new_build_options(
    num_replicas = 1L,
    num_partitions = 1L,
    device_ordinal = -1
  )
) {
  check_build_options(build_options)
  impl_compile_options_create(build_options)
}

check_compile_options <- function(compile_options) {
  stopifnot(inherits(compile_options, "PJRTCompileOptions"))
  invisible(NULL)
}

new_build_options <- function(
  num_replicas = 1L,
  num_partitions = 1L,
  device_ordinal = -1
) {
  impl_build_options_create(num_replicas, num_partitions, device_ordinal)
}

check_build_options <- function(build_options) {
  stopifnot(inherits(build_options, "PJRTBuildOptions"))
  invisible(NULL)
}

#' Compile a program
#' @description
#' Compile a program into a [`PJRTExecutable`].
#' @param program (`character(1)`)\cr
#'   A program to compile.
#' @param compile_options ([`PJRTCompileOptions`])\cr
#'   Compile options.
#' @param client ([`PJRTClient`])\cr
#'   A client to use for compilation.
#' @return A compiled program.
#' @export
pjrt_compile <- function(
  program,
  compile_options = new_compile_options(),
  client = default_client()
) {
  check_client(client)
  check_program(program)
  check_compile_options(compile_options)

  impl_client_program_compile(client, program, compile_options)
}

client_scalar_buffer_from_host <- function(data, client = default_client()) {
  check_client(client)
  impl_client_scalar_buffer_from_host(client, data)
}

client_buffer_from_integer <- function(
  data,
  precision = 32L,
  signed = TRUE,
  dims,
  client = default_client()
) {
  check_client(client)
  impl_client_buffer_from_integer(
    client,
    data,
    dims,
    precision,
    signed
  )
}

client_buffer_from_logical <- function(
  data,
  dims,
  client = default_client()
) {
  check_client(client)
  impl_client_buffer_from_logical(client, data, dims)
}

client_buffer_from_double <- function(
  data,
  precision = 32L,
  dims,
  client = default_client()
) {
  check_client(client)
  impl_client_buffer_from_double(
    client,
    data,
    dims,
    precision
  )
}

# TODO: Rename to buffer_to_host
client_buffer_to_host <- function(buffer, client = default_client()) {
  check_client(client)
  check_buffer(buffer)

  impl_client_buffer_to_host(client, buffer)
}

client_buffer_from_host <- function(data, client = default_client()) {
  check_client(client)

  # For scalars, use the scalar function
  if (length(data) == 1) {
    return(client_scalar_buffer_from_host(data, client))
  }

  # For vectors/arrays, detect type and use appropriate function
  if (is.logical(data)) {
    return(client_buffer_from_logical(data, client))
  } else if (is.double(data) || is.numeric(data)) {
    return(client_buffer_from_double(data, client))
  } else if (is.integer(data)) {
    return(client_buffer_from_integer(data, client))
  } else {
    stop("Unsupported data type: ", class(data)[1])
  }
}

client_platform_name <- function(client = default_client()) {
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

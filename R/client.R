#' @title Compile a Program
#' @description
#' Compile a `PJRTProgram` program into a `PJRTExecutable`.
#'
#' @param program (`character(1)`)\cr
#'   A program to compile.
#' @param compile_options (`PJRTCompileOptions`)\cr
#'   Compile options.
#' @param client (`PJRTClient` or `character(1)`)\cr
#'   A PJRT client object or a platform name (e.g., "cpu", "gpu", "metal").
#'   If a platform name is provided, the appropriate client will be created
#'   or retrieved automatically.
#' @return `PJRTExecutable`
#' @export
pjrt_compile <- function(
  program,
  compile_options = new_compile_options(),
  client = default_client()
) {
  # Handle case where client is a string (platform name)
  if (is.character(client)) {
    client <- default_client(client)
  }

  check_client(client)
  check_program(program)
  check_compile_options(compile_options)

  impl_client_program_compile(client, program, compile_options)
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

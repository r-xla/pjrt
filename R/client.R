#' @title Compile a Program
#' @description
#' Compile a `PJRTProgram` program into a `PJRTExecutable`.
#'
#' @param program (`character(1)`)\cr
#'   A program to compile.
#' @param compile_options (`PJRTCompileOptions`)\cr
#'   Compile options.
#' @template param_client
#' @return `PJRTExecutable`
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

#' @title Create a Client
#' @description
#' Create a PJRT client for a specific device.
#'
#' @param platform (`character(1)`)\cr
#'   Platform name (e.g., "cpu", "cuda", "metal").
#' @return `PJRTClient`
#' @export
pjrt_client <- function(platform) {
  if (platform %in% names(the$clients)) {
    return(the$clients[[platform]])
  }
  plugin_client_create(plugin_load(platform), platform)
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

#' @title Default Client
#' @description
#' Respects environment variable `PJRT_DEVICE` and otherwise defaults to "cpu".
#'
#' @return `PJRTClient`
#' @export
default_client <- function() {
  platform <- Sys.getenv("PJRT_DEVICE", "cpu")
  plugin_client_create(plugin_load(platform), platform)
}

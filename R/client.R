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
  client = pjrt_client()
) {
  client <- as_pjrt_client(client)
  check_program(program)
  check_compile_options(compile_options)

  impl_client_program_compile(client, program, compile_options)
}

#' @title Create a Client
#' @description
#' Create a PJRT client for a specific device.
#'
#' @section Extractors:
#' * [`platform_name()`] for a `character(1)` representation of the platform.
#'
#' @param platform (`character(1)` | `NULL`)\cr
#'   Platform name (e.g., "cpu", "cuda", "metal").
#'   If `NULL`, use `PJRT_PLATFORM` environment variable or default to "cpu".
#' @return `PJRTClient`
#' @export
#' @examplesIf Sys.info()["sysname"] != "Windows"
#' pjrt_client("cpu")
pjrt_client <- function(platform = NULL) {
  if (is.null(platform)) {
    platform <- default_platform()
  }

  if (platform %in% names(the$clients)) {
    return(the$clients[[platform]])
  }
  plugin_client_create(pjrt_plugin(platform), platform)
}

#' @title Convert to PJRT Client
#' @description
#' Convert a platform name to a PJRT client or verify that an object is already a client.
#'
#' @param x (`PJRTClient` | `character(1)`)\cr
#'   Either a PJRT client object or a platform name (e.g., "cpu", "cuda", "metal").
#' @return `PJRTClient`
#' @export
as_pjrt_client <- function(x) {
  if (inherits(x, "PJRTClient")) {
    return(x)
  }

  if (is.character(x) && length(x) == 1 && nchar(x) > 0) {
    return(pjrt_client(x))
  }

  stop("Must be a PJRTClient or a platform name")
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

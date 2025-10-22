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
#' * [`platform()`] for a `character(1)` representation of the platform.
#' * [`devices()`] for a `list` of `PJRTDevice` objects.
#'
#' @param platform (`character(1)` | `NULL`)\cr
#'   Platform name (e.g., "cpu", "cuda", "metal").
#'   If `NULL`, use `PJRT_PLATFORM` environment variable or default to "cpu".
#' @param ... Additional options passed to the PJRT client creation.
#'   For CPU clients, you can pass `cpu_device_count` to specify the number of CPU devices (default: number of logical cores).
#' @return `PJRTClient`
#' @export
pjrt_client <- function(platform = NULL, ...) {
  if (is.null(platform)) {
    platform <- default_platform()
  }

  plugin_client_create(pjrt_plugin(platform), platform, options = list(...))
}

default_client_options <- function(platform) {
  switch(platform, cpu = list(cpu_device_count = pjrt_config()$cpu_device_count), list())
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

  if (is.null(x)) {
    return(pjrt_client())
  }

  stop("Must be a PJRTClient or a platform name")
}

#' @title Platform Name
#' @description
#' Get the platform name of a PJRT buffer.
#' @param x (`PJRTBuffer`)\cr
#'   The buffer.
#' @param ... Additional arguments (unused).
#' @return `character(1)`
#' @export
platform <- S7::new_generic("platform", "x")
S7::method(platform, S7::new_S3_class("PJRTClient")) <- function(x) {
  impl_client_platform(x)
}

#' @title Devices
#' @description
#' Get the addressable devices for a PJRT client.
#' @param client ([`PJRTClient`][pjrt_client])\cr
#'   Object convertible to a `PJRTClient`.
#' @return `list` of `PJRTDevice`
#' @export
devices <- function(client = NULL) {
  impl_client_devices(as_pjrt_client(client))
}

#' @title Convert to PJRT Device
#' @description
#' Convert a platform name or device to a PJRT device object.
#'
#' @param x (`PJRTDevice` | `character(1)` | `NULL`)\cr
#'   Either a PJRT device object, a platform name (e.g., "cpu", "cuda", "metal"),
#'   a device specification with index (e.g., "cpu:0", "cuda:1" for 0-based indexing),
#'   or NULL (defaults to first CPU device).
#' @return `PJRTDevice`
#' @keywords internal
as_pjrt_device <- function(x) {
  if (inherits(x, "PJRTDevice")) {
    return(x)
  }

  if (is.character(x) && length(x) == 1 && nchar(x) > 0) {
    # Parse device specification (e.g., "cpu:0" or just "cpu")
    parts <- strsplit(x, ":", fixed = TRUE)[[1]]
    platform_name <- parts[1]
    device_index <- if (length(parts) > 1) {
      x <- as.integer(parts[2L])
      assert_int(x, lower = 0L, coerce = TRUE)
    } else {
      0L
    }

    # Get devices for this platform
    client <- pjrt_client(platform_name)
    devs <- devices(client)
    if (!length(devs)) {
      stop("No devices available for platform: ", platform_name)
    }
    if (device_index >= length(devs)) {
      cli_abort("Device index {device_index} out of range for platform {platform_name}")
    }
    return(devs[[device_index + 1]])
  }

  if (is.null(x)) {
    # Default to first device of default platform
    client <- pjrt_client()
    devs <- devices(client)
    if (length(devs) == 0) {
      stop("No devices available")
    }
    return(devs[[1]])
  }

  stop("Must be a PJRTDevice, a PJRTClient, a platform name, or NULL")
}

#' @title Get Client from Device
#' @description
#' Get the client associated with a device by looking it up from the global cache.
#'
#' @param device (`PJRTDevice`)\cr
#'   A PJRT device object.
#' @return `PJRTClient`
#' @keywords internal
client_from_device <- function(device) {
  if (!inherits(device, "PJRTDevice")) {
    stop("Must be a PJRTDevice")
  }
  the[["clients"]][[impl_device_platform(device)]]
}

#' @export
as.character.PJRTClient <- function(x, ...) {
  platform(x)
}

#' @export
format.PJRTClient <- function(x, ...) {
  as.character(x)
}

#' @export
print.PJRTClient <- function(x, ...) {
  cat(sprintf("<PJRTClient:%s>\n", platform(x)))
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

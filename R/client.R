#' @title Compile a Program
#' @description
#' Compile a `PJRTProgram` program into a `PJRTExecutable`.
#'
#' @param program (`character(1)`)\cr
#'   A program to compile.
#' @param compile_options (`PJRTCompileOptions`)\cr
#'   Compile options.
#' @param device (`NULL` | `PJRTDevice` | `character(1)`)\cr
#'   A `PJRTDevice` object or the name of the platform to use ("cpu", "cuda", ...),
#'   in which case the first device for that platform is used.
#'   The default is to use the CPU platform, but this can be configured via the `PJRT_PLATFORM`
#'   environment variable.
#' @param xla_options (named `list()`)\cr
#'   XLA compiler options for this compilation only. Each is named
#'   after a field of XLA's `DebugOptions` message (`xla.proto`), the names `XLA_FLAGS` takes
#'   without the leading `--`, and XLA applies it on top of its defaults and of `XLA_FLAGS`.
#'   The type of a value decides how XLA reads it: a `logical(1)` sets a boolean option, an
#'   `integer(1)` an integer option (or an enum by number), a `double(1)` a floating-point
#'   option, and a `character(1)` is parsed by the option's type -- `"True"` or `"False"`, a
#'   number, the name of an enum value, or a comma-separated list for a repeated option, which
#'   replaces its whole value. So an integer option takes `3L` or `"3"`, but not `3`, and a
#'   single-precision option takes a `double(1)` only if it is positive, so `0` or a negative
#'   value has to be given as a string, e.g. `"0"`. XLA reports an unknown name or a value of the
#'   wrong type when compiling.
#'
#'   An option of map type, such as `xla_backend_extra_options`, takes a `character(1)` in
#'   `XLA_FLAGS` syntax: comma-separated entries `key=value`, or `key` for an empty value, e.g.
#'   `"xla_cpu_use_multi_output_fusion"`. Its entries are added to those the plugin applies by
#'   default, so entries set through `XLA_FLAGS` are kept. XLA does not check the entries: it
#'   ignores a key it does not know, and some keys count by their presence alone, so
#'   `"xla_cpu_use_multi_output_fusion=false"` turns that option on as well. On the CPU, XLA hands
#'   a key that does not start with `xla_` to LLVM, which ends the R process if it does not know
#'   the key.
#'
#'   XLA's overrides cannot set a map, so pjrt reads the plugin's `DebugOptions`, with
#'   `XLA_FLAGS` applied, back from a small program it compiles once per client, adds the entries
#'   and sends the whole set. That differs from a compilation without map options in a few ways:
#'   XLA reads some options only when a compilation sends such a set (with `xla_dump_to`, it then
#'   also writes the compilation's inputs, and it dumps the small program too), and with
#'   `--xla_flags_reset=true` a later change to `XLA_FLAGS` does not reach compilations with map
#'   options. Reading the set back needs a plugin that reports the compile options it applies,
#'   which the CPU plugin does and the CUDA plugin does not; there, set the option through
#'   `XLA_FLAGS` before the plugin is loaded.
#' @return `PJRTExecutable`
#' @examplesIf plugins_downloaded()
#' # Create a simple program
#' src <- r"(
#' func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
#'   return %arg0 : tensor<2xf32>
#' }
#' )"
#' prog <- pjrt_program(src = src)
#' exec <- pjrt_compile(prog)
#'
#' # Write the optimized HLO of this compilation to a directory
#' dir <- tempfile()
#' exec <- pjrt_compile(prog, xla_options = list(xla_dump_to = dir))
#' list.files(dir)
#'
#' # Keep XLA:CPU from duplicating cheap operations into the fusions that use them
#' exec <- pjrt_compile(
#'   prog,
#'   device = "cpu",
#'   xla_options = list(xla_backend_extra_options = "xla_cpu_use_multi_output_fusion")
#' )
#' @export
pjrt_compile <- function(
  program,
  compile_options = new_compile_options(),
  device = NULL,
  xla_options = list()
) {
  device <- as_pjrt_device(device)
  client <- client_from_device(device)
  check_program(program)
  check_compile_options(compile_options)
  check_xla_options(xla_options)
  is_map <- names(xla_options) %in% impl_xla_map_options()

  impl_client_program_compile(
    client,
    device,
    program,
    compile_options,
    xla_options[!is_map],
    Map(parse_xla_map_option, xla_options[is_map], names(xla_options)[is_map])
  )
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
#'   Platform name (e.g., "cpu", "cuda").
#'   If `NULL`, use `PJRT_PLATFORM` environment variable or default to "cpu".
#' @param ... Additional options passed to the PJRT client creation.
#'   For CPU clients, you can pass `cpu_device_count` to specify the number of CPU devices.
#'   You can also configure this via `PJRT_CPU_DEVICE_COUNT` environment variable.
#' @return `PJRTClient`
#' @examplesIf plugins_downloaded()
#' # Create a client (defaults to CPU)
#' client <- pjrt_client()
#' client
#' @export
pjrt_client <- function(platform = NULL, ...) {
  if (is.null(platform)) {
    platform <- default_platform()
  }

  plugin_client_create(pjrt_plugin(platform), platform, options = list(...))
}

default_client_options <- function(platform) {
  switch(platform, cpu = list(cpu_device_count = as.integer(Sys.getenv("PJRT_CPU_DEVICE_COUNT", 1L))), list())
}

#' @title Convert to PJRT Client
#' @description
#' Convert a platform name to a PJRT client or verify that an object is already a client.
#'
#' @param x (`PJRTClient` | `character(1)`)\cr
#'   Either a PJRT client object or a platform name (e.g., "cpu", "cuda").
#' @return `PJRTClient`
#' @examplesIf plugins_downloaded()
#' # Convert from platform name
#' client <- as_pjrt_client("cpu")
#' client
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

  cli_abort("Must be a PJRTClient or a platform name")
}

#' @title Platform Name
#' @description
#' Get the platform name of a PJRT buffer.
#' @param x (`PJRTBuffer`)\cr
#'   The buffer.
#' @param ... Additional arguments (unused).
#' @return `character(1)`
#' @examplesIf plugins_downloaded()
#' buf <- pjrt_buffer(c(1, 2, 3))
#' platform(buf)
#' @export
platform <- function(x, ...) {
  UseMethod("platform")
}

#' @export
platform.PJRTClient <- function(x, ...) {
  impl_client_platform(x)
}

#' @title Devices
#' @description
#' Get the addressable devices.
#' @param x An object to get devices from: a [`PJRTClient`][pjrt_client],
#'   a [`PJRTLoadedExecutable`][pjrt_compile], or `NULL` (default client).
#' @param ... Additional arguments (currently unused).
#' @return `list` of `PJRTDevice`
#' @examplesIf plugins_downloaded()
#' # Create client (defaults to CPU)
#' client <- pjrt_client()
#' devices(client)
#' @export
devices <- function(x = NULL, ...) {
  UseMethod("devices")
}

#' @export
devices.default <- function(x = NULL, ...) {
  devices(as_pjrt_client(x))
}

#' @export
devices.PJRTClient <- function(x, ...) {
  plat <- platform(x)
  if (!exists(plat, envir = the[["devices"]], inherits = FALSE)) {
    the[["devices"]][[plat]] <- impl_client_devices(x)
  }
  the[["devices"]][[plat]]
}

#' @title Convert to PJRT Device
#' @description
#' Convert a platform name or device to a PJRT device object.
#'
#' @param x (`PJRTDevice` | `character(1)` | `NULL`)\cr
#'   Either a PJRT device object, a platform name (e.g., "cpu", "cuda"),
#'   a device specification with index (e.g., "cpu:0", "cuda:1" for 0-based indexing),
#'   or NULL (defaults to first CPU device).
#' @return `PJRTDevice`
#' @export
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

    client <- pjrt_client(platform_name)
    devs <- devices(client)
    if (!length(devs)) {
      cli_abort("No devices available for platform: ", platform_name)
    }
    if (device_index >= length(devs)) {
      cli_abort("Device index {device_index} out of range for platform {platform_name}")
    }
    return(devs[[device_index + 1]])
  }

  if (is.null(x)) {
    client <- pjrt_client()
    devs <- devices(client)
    if (length(devs) == 0) {
      cli_abort("No devices available")
    }
    return(devs[[1]])
  }

  cli_abort(c(
    "Must be a PJRTDevice, a PJRTClient, a platform name, or NULL",
    x = "Got {.cls {class(x)}}"
  ))
}

client_from_device <- function(device) {
  if (!inherits(device, "PJRTDevice")) {
    cli_abort("Must be a PJRTDevice")
  }
  the[["clients"]][[platform(device)]]
}

# Map a freshly-created PJRTDevice xptr to the cached device with the same
# string representation, so equal devices share one external pointer and can
# be used as stable hashtab keys (and compared by identity).
#
# Memoized by the device string: after the first lookup the canonical xptr is
# returned via a single hashtab read, avoiding the per-call rescan of the
# client's device list (each iteration of which stringified a device).
cached_device <- function(device) {
  target <- as.character(device)
  canonical <- the[["canonical_devices"]][[target]]
  if (!is.null(canonical)) {
    return(canonical)
  }
  client <- client_from_device(device)
  if (is.null(client)) {
    # Client not yet created: don't cache, the platform may not be set up.
    return(device)
  }
  for (d in devices(client)) {
    if (identical(as.character(d), target)) {
      the[["canonical_devices"]][[target]] <- d
      return(d)
    }
  }
  device
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

check_xla_options <- function(xla_options) {
  assert_list(xla_options, names = "unique")
  is_scalar <- vapply(
    xla_options,
    function(x) {
      (is.logical(x) || is.integer(x) || is.double(x) || is.character(x)) &&
        length(x) == 1L &&
        !is.na(x) &&
        is.null(attributes(x))
    },
    logical(1L)
  )
  if (!all(is_scalar)) {
    bad <- names(xla_options)[!is_scalar]
    cli_abort(c(
      "Each of {.arg xla_options} must be a non-missing logical, integer, double or character scalar.",
      x = "Not one: {.val {bad}}."
    ))
  }
  invisible(NULL)
}

# A map-valued XLA option in XLA_FLAGS syntax ("a,b=1") as a named character vector.
parse_xla_map_option <- function(value, name) {
  if (!is.character(value) || !grepl("^[^,=]+(=[^,]*)?(,[^,=]+(=[^,]*)?)*$", value)) {
    cli_abort(c(
      "{.arg xla_options${name}} sets a map, so it must be a string of comma-separated entries {.code key=value} or {.code key}.",
      x = "Got {.val {value}}."
    ))
  }
  entries <- strsplit(value, ",", fixed = TRUE)[[1L]]
  eq <- regexpr("=", entries, fixed = TRUE)
  values <- ifelse(eq > 0L, substring(entries, eq + 1L), "")
  names(values) <- ifelse(eq > 0L, substr(entries, 1L, eq - 1L), entries)
  values
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

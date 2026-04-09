pjrt_platform_name <- function(platform) {
  switch(platform, cpu = "host", host = "host", cuda = "cuda", platform)
}

#' @title Register a Custom Call Handler
#' @description
#' Register an XLA FFI handler for use with `stablehlo.custom_call`.
#'
#' Handlers are C/C++ functions defined using the XLA FFI API
#' (see `xla/ffi/api/ffi.h` shipped in pjrt's `inst/include/`).
#' They are passed to this function as external pointers.
#'
#' Registration is deferred: if the PJRT plugin for a given platform
#' is not yet loaded, the handler is queued and registered automatically
#' when [`pjrt_plugin()`] loads it.
#'
#' @param target_name (`character(1)`)\cr
#'   The target name used in `stablehlo.custom_call @target_name(...)`.
#' @param handler A named list of external pointers (`externalptr`) to
#'   `XLA_FFI_Handler`s, keyed by PJRT platform name
#'   (e.g., `list(host = ptr)` or `list(host = cpu_ptr, cuda = cuda_ptr)`).
#' @param .package (`character(1)` or `NULL`)\cr
#'   The package registering this handler. When provided, handlers are
#'   automatically removed from the registry when the package unloads.
#' @return `NULL` (invisibly).
#' @export
pjrt_register_custom_call <- function(target_name, handler, .package = NULL) {
  checkmate::assert_string(target_name)

  if (!is.list(handler) || is.null(names(handler)) || !all(nzchar(names(handler)))) {
    cli_abort(
      "{.arg handler} must be a named list of external pointers keyed by platform (e.g. {.val host}, {.val cuda})."
    )
  }
  if (!all(vapply(handler, is, logical(1), "externalptr"))) {
    cli_abort("All elements of {.arg handler} must be external pointers.")
  }
  names(handler) <- vapply(names(handler), pjrt_platform_name, character(1))

  if (target_name %in% names(the[["custom_calls"]])) {
    existing <- the[["custom_calls"]][[target_name]]
    pjrt_debug(
      "Overwriting custom call {.val {target_name}} (previously from package {.pkg {existing$package %||% 'unknown'}})"
    )
  }

  the[["custom_calls"]][[target_name]] <- list(
    handler = handler,
    package = .package
  )

  for (platform in ls(the[["plugins"]])) {
    plugin <- the[["plugins"]][[platform]]
    pjrt_platform <- pjrt_platform_name(platform)
    register_custom_call_for_plugin(target_name, handler, plugin, pjrt_platform)
  }

  if (!is.null(.package)) {
    setHook(
      packageEvent(.package, "onUnload"),
      function(...) pjrt_unregister_custom_calls(.package),
      action = "append"
    )
  }

  invisible(NULL)
}

register_custom_call_for_plugin <- function(target_name, handler, plugin, pjrt_platform) {
  ptr <- handler[[pjrt_platform]]
  if (is.null(ptr)) {
    pjrt_debug(
      "Custom call {.val {target_name}}: no handler for platform {.val {pjrt_platform}}, skipping."
    )
    return(invisible(FALSE))
  }

  tryCatch(
    {
      impl_register_custom_call(plugin, target_name, ptr, pjrt_platform)
      pjrt_debug("Registered custom call {.val {target_name}} for platform {.val {pjrt_platform}}")
      invisible(TRUE)
    },
    error = function(e) {
      cli::cli_warn(c(
        x = "Failed to register custom call {.val {target_name}} for platform {.val {pjrt_platform}}.",
        i = conditionMessage(e)
      ))
      invisible(FALSE)
    }
  )
}

drain_custom_calls <- function(plugin, platform) {
  pjrt_platform <- pjrt_platform_name(platform)
  for (target_name in names(the[["custom_calls"]])) {
    entry <- the[["custom_calls"]][[target_name]]
    register_custom_call_for_plugin(target_name, entry$handler, plugin, pjrt_platform)
  }
}

pjrt_unregister_custom_calls <- function(package) {
  calls <- the[["custom_calls"]]
  to_remove <- vapply(calls, function(x) identical(x$package, package), logical(1))
  if (any(to_remove)) {
    pjrt_debug("Removing {sum(to_remove)} custom call(s) from package {.pkg {package}}")
    the[["custom_calls"]][to_remove] <- NULL
  }
}

setup_logging <- function() {
  # TF_CPP_MIN_LOG_LEVEL: 0 = info, 1 = warn (default), 2 = error, 3 = off
  # Users can set this env var directly; we default to 1 to suppress startup info.
  if (Sys.getenv("TF_CPP_MIN_LOG_LEVEL", "") == "") {
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "1")
  }
}

register_namespace_callback <- function(pkgname, namespace, callback) {
  # nocov start
  assert_string(pkgname)
  assert_string(namespace)
  assert_function(callback)

  remove_hook <- function(event) {
    hooks <- getHook(event)
    pkgnames <- vapply(
      hooks,
      function(x) {
        ee <- environment(x)
        if (isNamespace(ee)) environmentName(ee) else ee$pkgname %||% NA_character_
      },
      NA_character_
    )
    setHook(event, hooks[pkgnames != pkgname], action = "replace")
  }

  remove_hooks <- function(...) {
    remove_hook(packageEvent(namespace, "onLoad"))
    remove_hook(packageEvent(pkgname, "onUnload"))
  }

  if (isNamespaceLoaded(namespace)) {
    callback()
  }

  setHook(packageEvent(namespace, "onLoad"), callback, action = "append")
  setHook(packageEvent(pkgname, "onUnload"), remove_hooks, action = "append")
  # nocov end
}

.onLoad <- function(libname, pkgname) {
  # nocov start
  setup_logging()
  # this allows for tests without as_array() conversion
  register_s3_method("waldo", "compare_proxy", "PJRTBuffer")
  # pjrt's own custom calls (print_tensor and the LAPACK / cuSOLVER linear
  # algebra), so that downstream packages can invoke them via
  # `stablehlo.custom_call @<target>(...)` without shipping their own kernels.
  handlers <- builtin_custom_calls()
  for (target in names(handlers)) {
    pjrt_register_custom_call(target, handlers[[target]], .package = pkgname)
  }

  # make safetensors work with pjrt
  register_namespace_callback(pkgname, "safetensors", function(...) {
    frameworks <- utils::getFromNamespace(
      "safetensors_frameworks",
      ns = "safetensors"
    )
    frameworks[["pjrt"]] <- list(
      constructor = pjrt_tensor_from_raw,
      packages = "pjrt"
    )
  })
  # nocov end
}

# silence rcmd check
withr::local_options

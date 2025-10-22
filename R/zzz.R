register_namespace_callback <- function(pkgname, namespace, callback) {
  assert_string(pkgname)
  assert_string(namespace)
  assert_function(callback)

  remove_hook <- function(event) {
    hooks <- getHook(event)
    pkgnames <- vapply(
      hooks,
      function(x) {
        ee <- environment(x)
        if (isNamespace(ee)) environmentName(ee) else environment(x)$pkgname
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
}

.onLoad <- function(libname, pkgname) {
  # this allows for tests without as_array() conversion
  register_s3_method("waldo", "compare_proxy", "PJRTBuffer")
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
  S7::methods_register()
}

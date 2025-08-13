#' @import checkmate
#' @importFrom safetensors safe_tensor_buffer safe_tensor_meta
NULL

register_namespace_callback = function(pkgname, namespace, callback) {
  assert_string(pkgname)
  assert_string(namespace)
  assert_function(callback)

  remove_hook = function(event) {
    hooks = getHook(event)
    pkgnames = vapply(hooks, function(x) {
      ee = environment(x)
      if (isNamespace(ee)) environmentName(ee) else environment(x)$pkgname
    }, NA_character_)
    setHook(event, hooks[pkgnames != pkgname], action = "replace")
  }

  remove_hooks = function(...) {
    remove_hook(packageEvent(namespace, "onLoad"))
    remove_hook(packageEvent(pkgname, "onUnload"))
  }

  if (isNamespaceLoaded(namespace)) {
    callback()
  }

  setHook(packageEvent(namespace, "onLoad"), callback, action = "append")
  setHook(packageEvent(pkgname, "onUnload"), remove_hooks, action = "append")
}

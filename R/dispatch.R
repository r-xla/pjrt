#' @title Native eager-dispatch fast path
#' @name pjrt_dispatch
#' @description
#' A native fast path for repeatedly executing a compiled program on inputs of
#' the same signature, intended for a code-transformation frontend's eager
#' primitive dispatch (e.g. anvl). It owns an executable cache keyed on the
#' inputs' structure and abstract values, calling back into R to compile only on
#' a cache miss.
#'
#' [`pjrt_dispatcher()`] creates a dispatcher for one program family;
#' [`pjrt_dispatch()`] runs a call through it.
#'
#' @details
#' On each [`pjrt_dispatch()`] call the inputs are flattened, classified, and
#' turned into a cache key built from each leaf's dtype/shape/device (read
#' natively off the buffers) plus an `ambiguous` flag. On a hit the cached
#' executable runs immediately; on a miss `compile` is invoked.
#'
#' A leaf is dispatchable if it is a [`pjrt_buffer`] or an `"AnvlArray"` list
#' with `$backend == "xla"` (from which `$data` and `$ambiguous` are read).
#' Any other input -- a non-buffer, a non-xla backend, or inputs spread across
#' devices -- makes [`pjrt_dispatch()`] return [`pjrt_dispatch_sentinel()`] so
#' the caller can fall back to its own slow path.
#'
#' @param capacity (`integer(1)`)\cr
#'   Maximum number of compiled executables to cache.
#' @param compile (`function`)\cr
#'   Cache-miss callback. Called as `compile(args)` with the dispatched argument
#'   list and must return a named list with:
#'   * `exec`: a [`pjrt_compile`]d executable,
#'   * `const_arrays` (optional): buffers prepended to the inputs,
#'   * `phantom_specs` (optional): a list of `list(dtype = <string>, shape =
#'     <integer>)` donation-output buffers to allocate fresh per call,
#'   * `client`, `device` (required if `phantom_specs` is non-empty): the
#'     [`pjrt_client`] and device used to allocate phantom buffers,
#'   * `out_tree`, `ambiguous_out` (optional): opaque values returned verbatim
#'     for the caller to wrap the outputs.
#' @param static (`character()`)\cr
#'   Names of top-level arguments that are static (not arrays). Static values
#'   are part of the cache key (compared with [identical()]) and are excluded
#'   from execution. Defaults to none.
#' @return [`pjrt_dispatcher()`] returns a `PJRTDispatcher`.
#' @export
pjrt_dispatcher <- function(capacity, compile, static = character()) {
  checkmate::assert_count(capacity, positive = TRUE)
  checkmate::assert_function(compile)
  checkmate::assert_character(static, any.missing = FALSE)
  impl_dispatch_create(as.integer(capacity), compile, as.character(static))
}

#' @rdname pjrt_dispatch
#' @param dispatcher (`PJRTDispatcher`)\cr A dispatcher from [`pjrt_dispatcher()`].
#' @param args (`list`)\cr The (already evaluated) argument list of the call.
#' @return [`pjrt_dispatch()`] returns, on a handled call, a list with
#'   `buffers` (the raw output [`pjrt_buffer`]s), `out_tree`, and
#'   `ambiguous_out` (the opaque values from `compile`); otherwise the value of
#'   [`pjrt_dispatch_sentinel()`].
#' @export
pjrt_dispatch <- function(dispatcher, args) {
  impl_dispatch_run(dispatcher, args)
}

#' @rdname pjrt_dispatch
#' @return [`pjrt_dispatch_sentinel()`] returns the singleton sentinel value
#'   that [`pjrt_dispatch()`] yields for calls it does not handle natively.
#' @export
pjrt_dispatch_sentinel <- function() {
  impl_dispatch_sentinel()
}

#' @rdname pjrt_dispatch
#' @return [`pjrt_dispatch_size()`] returns the number of compiled executables
#'   the dispatcher currently caches.
#' @export
pjrt_dispatch_size <- function(dispatcher) {
  impl_dispatch_size(dispatcher)
}

#' @export
print.PJRTDispatcher <- function(x, ...) {
  cat("<PJRTDispatcher>\n")
  invisible(x)
}

#' @title Dispatcher
#' @description
#' A native fast path for repeatedly executing a compiled program on inputs of
#' the same signature, intended to be used in anvl. It owns an executable cache keyed on the
#' inputs' structure and abstract values, calling back into R to compile only on
#' a cache miss.
#'
#' [`pjrt_dispatcher()`] creates a pjrt_dispatcher for one program family;
#' [`pjrt_dispatch()`] runs a call through it.
#'
#' @details
#' On each [`pjrt_dispatch()`] call the inputs are flattened, classified, and
#' turned into a cache key built per leaf from either its dtype/shape (read
#' natively off buffers, or the R type/dim of a bare literal/array) plus an
#' `ambiguous` flag, or -- for static and other value-keyed leaves -- the value
#' itself (compared with [identical()]). On a hit the cached artifact runs
#' immediately; on a miss `compile` is invoked.
#'
#' With `engine = "pjrt"`, executable inputs are assembled natively: `"xla"`
#' `"AnvlArray"` leaves contribute their buffer (copied to the entry's device
#' under `move_inputs = TRUE`), and bare R literals/arrays are uploaded with
#' the same dtype defaults as [`pjrt_scalar()`]/[`pjrt_buffer()`]. With
#' `engine = "closure"`, the compiled R closure is called on the flat leaf
#' list, with R-array-backed `"AnvlArray"` leaves contributing their `$data`.
#'
#' A leaf no engine can execute (e.g. an `"AnvlArray"` of a foreign backend)
#' is keyed by value; `compile` is expected to reject it, so such calls error
#' through the callback. [`pjrt_dispatch_sentinel()`] is returned only for
#' inputs spread across devices when no target device is fixed -- the caller
#' should re-run its own input validation to raise its canonical error.
#'
#' @param capacity (`integer(1)`)\cr
#'   Maximum number of compiled executables to cache.
#' @param compile (`function`)\cr
#'   Cache-miss callback. Called as `compile(args)` with the dispatched argument
#'   list and must return a named list with:
#'   * `exec`: a [`pjrt_compile`]d executable (`engine = "pjrt"`),
#'   * `r_fun`: a function called on the flat leaf list (`engine = "closure"`),
#'   * `const_arrays` (optional): buffers prepended to the inputs,
#'   * `phantom_specs` (optional): a list of `list(dtype = <string>, shape =
#'     <integer>)` donation-output buffers to allocate fresh per call,
#'   * `client`, `device` (required for phantom allocation, R-data uploads, and
#'     device moves): the [`pjrt_client`] and target device,
#'   * `out_tree`, `ambiguous_out` (optional): opaque values returned verbatim
#'     for the caller to wrap the outputs.
#' @param static (`character()`)\cr
#'   Names of top-level arguments that are static (not arrays). Static values
#'   are part of the cache key (compared with [identical()]) and are excluded
#'   from execution. Defaults to none.
#' @param engine (`character(1)`)\cr
#'   `"pjrt"` (default) executes a compiled PJRT executable; `"closure"` calls
#'   the compiled R closure returned by `compile` on the flat leaves.
#' @param move_inputs (`logical(1)`)\cr
#'   If `TRUE`, a target device is fixed per cache entry (via the `compile`
#'   callback's `device`) and buffer inputs are copied to it at execute time;
#'   the cache key then carries no device. Default `FALSE`: the first buffer's
#'   device is the call's device and conflicting inputs yield the sentinel.
#' @return [`pjrt_dispatcher()`] returns a `PJRT_dispatcher`.
#' @export
pjrt_dispatcher <- function(
  capacity,
  compile,
  static = character(),
  engine = "pjrt",
  move_inputs = FALSE
) {
  checkmate::assert_count(capacity, positive = TRUE)
  checkmate::assert_function(compile)
  checkmate::assert_character(static, any.missing = FALSE)
  checkmate::assert_choice(engine, c("pjrt", "closure"))
  checkmate::assert_flag(move_inputs)
  impl_dispatch_create(
    as.integer(capacity),
    compile,
    as.character(static),
    engine,
    move_inputs
  )
}

#' @rdname pjrt_dispatch
#' @param pjrt_dispatcher (`PJRT_dispatcher`)\cr A pjrt_dispatcher from [`pjrt_dispatcher()`].
#' @param args (`list`)\cr The (already evaluated) argument list of the call.
#' @return [`pjrt_dispatch()`] returns, on a handled call with
#'   `engine = "pjrt"`, a list with `buffers` (the raw output
#'   [`pjrt_buffer`]s), `out_tree` and `ambiguous_out` (the opaque values from
#'   `compile`), `out_dtypes` and `out_shapes` (each output's dtype string and
#'   integer shape, read natively), and `device` (the `compile` callback's
#'   device object). With `engine = "closure"` it returns
#'   `list(value = <closure result>)`. Otherwise it returns the value of
#'   [`pjrt_dispatch_sentinel()`].
#' @export
# Bound directly to the generated native entry (RcppExports.R precedes this
# file in the Collate order): pjrt_dispatch() is on anvl's per-primitive hot
# path, and a plain forwarding wrapper would add an R call frame per dispatch.
pjrt_dispatch <- impl_dispatch_run

#' @rdname pjrt_dispatch
#' @return [`pjrt_dispatch_sentinel()`] returns the singleton sentinel value
#'   that [`pjrt_dispatch()`] yields for calls it does not handle natively.
#' @export
pjrt_dispatch_sentinel <- function() {
  impl_dispatch_sentinel()
}

#' @rdname pjrt_dispatch
#' @return [`pjrt_dispatch_size()`] returns the number of compiled executables
#'   the pjrt_dispatcher currently caches.
#' @export
pjrt_dispatch_size <- function(pjrt_dispatcher) {
  impl_dispatch_size(pjrt_dispatcher)
}

#' @export
print.PJRT_dispatcher <- function(x, ...) {
  cat("<PJRT_dispatcher>\n")
  invisible(x)
}

#' @title Dispatcher
#' @description
#' A native fast path for repeatedly executing a compiled program on inputs of
#' the same signature, intended to be used in anvl. It owns an executable cache
#' keyed on the inputs' structure and abstract values, calling back into R to
#' compile only on a cache miss.
#'
#' [`dispatcher()`] creates a dispatcher for one program family;
#' [`dispatch()`] runs a call through it and returns the call's finished
#' result.
#'
#' @details
#' On each [`dispatch()`] call the inputs are flattened, classified, and
#' turned into a cache key built per leaf from either its dtype/shape (read
#' natively off buffers, or the R type/dim of a bare literal/array) plus an
#' `ambiguous` flag, or -- for static and other value-keyed leaves -- the value
#' itself (compared with [identical()]). On a hit the cached artifact runs
#' immediately; on a miss `compile` is invoked.
#'
#' The dispatcher core is backend-agnostic; the `backend` selects one of two
#' execution engines, and everything backend-specific sits behind it:
#'
#' * `backend = "xla"` executes a compiled PJRT executable natively: array
#'   leaves contribute their `$data` buffer (copied to the entry's device
#'   under `move_inputs = TRUE`), bare R literals/arrays are uploaded with the
#'   same dtype defaults as [`pjrt_scalar()`]/[`pjrt_buffer()`], and the
#'   output buffers are wrapped back into arrays (see *Array leaves and
#'   output wrapping*) and re-nested via the `compile` callback's `out_tree`
#'   -- all without leaving C++.
#' * any other `backend` calls the compiled R closure returned by `compile`
#'   on the call's inputs, with array leaves contributing their `$data`. The
#'   closure returns the call's finished value, so execution, output wrapping,
#'   and -- under `move_inputs = TRUE` -- input placement stay entirely under
#'   the backend's control. This is the path for any non-PJRT backend (e.g.
#'   anvl's `"quickr"`).
#'
#' Inputs are validated natively, before the cache is probed, and a rejection
#' names the offending argument by its path in the argument tree. An input must
#' be an `"AnvlArray"` of the dispatcher's `backend` carrying a `$device`, a
#' length-1 atomic scalar, or an [`is.array()`] value; anything else errors, as
#' does a static argument that is an `"AnvlArray"`, an `"AnvlArray"` of the
#' `"plain"` backend (those capture trace-time constants and are never call
#' arguments), and -- when no target device is fixed -- inputs spread across
#' devices. `compile` is therefore never asked to validate: it is called only
#' on a cache miss, only for inputs already known to be executable, and only
#' to produce the cache entry.
#'
#' @section Array leaves and output wrapping:
#' An array input is an R list of class `"AnvlArray"` with fields `$data` (the
#' backing buffer or R array), `$dtype` (a tengen `DataType`), `$shape`
#' (`integer()`), `$device` (an interned device object), `$ambiguous`
#' (`logical(1)`), and `$backend` (`character(1)`). The dispatcher reads this
#' contract when classifying inputs -- `$dtype`/`$shape` for any backend, the
#' `PJRTBuffer` in `$data` directly under `backend = "xla"` -- and writes it
#' when wrapping outputs: with `backend = "xla"`, [`dispatch()`] returns the
#' output buffers wrapped in this very layout (`$dtype`/`$shape`/`$ambiguous`
#' from the `compile` callback's `out_avals`, `$device` the entry's device,
#' `$backend` the dispatcher's `backend`), re-nested via `out_tree`. The
#' wrappers are built once, when the entry is compiled, so a call only
#' shallow-copies one per output and drops the buffer into its `$data`.
#'
#' @param capacity (`integer(1)`)\cr
#'   Maximum number of compiled executables to cache.
#' @param compile (`function`)\cr
#'   Cache-miss callback. Called as `compile(info)`, where `info` carries
#'   everything the dispatch already derived from the call, so that the callback
#'   never re-classifies the inputs:
#'   * `args`: the dispatched argument list,
#'   * `in_tree`: its `RTree` (see [`build_tree`]),
#'   * `leaves`: the flat leaf list (see [`flatten`]),
#'   * `is_static`: a `logical()` mask over `leaves`,
#'   * `avals`: per leaf, `NULL` if static, else the `list(dtype, shape,
#'     ambiguous)` the cache key was built from. `dtype` is the canonical dtype
#'     string (`"f32"`, `"i64"`, ...), whichever backend the leaf came from;
#'     `shape` is an `integer()`, empty for a scalar,
#'   * `default_device`: the device resolved for this call because no array
#'     input named one -- the device the cache key was built on, so `compile`
#'     must compile for it rather than resolve a default of its own. `NULL` when
#'     an array named the device, or under `move_inputs`.
#'
#'   For `backend = "xla"` it must return a named list with:
#'   * `exec`: a [`pjrt_compile`]d executable,
#'   * `client`, `device`: the [`pjrt_client`] and the device the entry is
#'     compiled for (used for R-data uploads, device moves, phantom
#'     allocation, and stamped on the wrapped outputs),
#'   * `out_tree`: the `RTree` of the outputs (see [`build_tree`]), used to
#'     re-nest the wrapped output arrays,
#'   * `out_avals`: one aval per output leaf of `out_tree`, each a
#'     `list(dtype = <string>, shape = <integer>, ambiguous = <logical(1)>)` --
#'     the same shape as the input avals the callback receives in `info$avals`
#'     (`ambiguous` is optional and defaults to `FALSE`). The callback traced
#'     and compiled the program, so it is the one that knows these; pjrt uses
#'     them to build the output wrappers once, at compile time,
#'   * `const_arrays` (optional): buffers prepended to the inputs,
#'   * `phantom_specs` (optional): a list of `list(dtype = <string>, shape =
#'     <integer>)` donation-output buffers to allocate fresh per call.
#'
#'   For any other `backend` it must return a named list with:
#'   * `r_fun`: a function called on the call's inputs -- the dynamic leaves
#'     only, in order, with an array leaf contributing its `$data` -- returning
#'     the call's finished value. Static leaves are *not* passed: they are
#'     constants of the closure `compile` just built, and a cache hit already
#'     proves the call's statics are `identical()` to the ones it was built
#'     from.
#' @param static (`character()`)\cr
#'   Names of top-level arguments that are static (not arrays). Static values
#'   are part of the cache key and are excluded from execution. Defaults to
#'   none. They are compared with `identical(num.eq = FALSE)`, i.e. numbers
#'   compare bitwise: `0` and `-0` are distinct keys (a redundant compile of the
#'   same program), which is what keeps a `bit64::integer64` `NA` -- stored as
#'   the bit pattern of `-0` -- from sharing a cache entry with `0`.
#' @param backend (`character(1)`)\cr
#'   The `$backend` tag every `"AnvlArray"` input must carry, and the tag
#'   stamped on wrapped outputs. It also selects the execution engine:
#'   `"xla"` (the default) executes a compiled PJRT executable natively; any
#'   other value calls the compiled R closure returned by `compile` on the flat
#'   leaves (anvl's quickr backend passes `"quickr"`).
#' @param move_inputs (`logical(1)`)\cr
#'   If `TRUE`, a target device is fixed per cache entry and the engine places
#'   each input on it at execute time; the cache key then carries no device, so
#'   inputs may arrive from any device. Default `FALSE`: the first array's
#'   device is the call's device, and a conflicting input is an error.
#'
#'   It is a per-dispatcher *policy* flag, not a device: it selects only
#'   *whether* inputs are relocated. *Which* device an entry targets is the
#'   `compile` callback's returned `device`, read off the entry at execute time
#'   -- the dispatcher core never resolves or owns a device. That target need
#'   not be constant across entries: when the backend fixes it up front (anvl's
#'   `jit(device = "cuda")`) every entry compiles for the same device, but when
#'   the backend derives it from a static argument (anvl's `device_arg()`) each
#'   value keys a separate entry compiled for a different device. That is why
#'   the target lives per entry rather than as a single device on the
#'   dispatcher, and why this parameter is a boolean and not a device.
#'
#'   Placing an input is the engine's own business, since only it knows what
#'   `$data` holds. With `backend = "xla"` the entry's device is the `compile`
#'   callback's `device` and an input living elsewhere is copied to it. With
#'   any other backend pjrt does nothing: the entry's device is whichever one
#'   the backend compiled `r_fun` for, and **`r_fun` must ensure its inputs are
#'   on that device** -- it receives only their `$data`, not their `$device`, so
#'   the placing has to be idempotent. Forgetting it means the backend runs on
#'   inputs it did not place; nothing in pjrt can detect that, because the
#'   device deliberately left the cache key.
#' @param default_device (`function` | `NULL`)\cr
#'   Called with no arguments to get the backend's *current* default device,
#'   whenever a call has no array input to read a device from. Its result is part
#'   of the cache key, so an entry compiled under one default device is never
#'   served after the default changes. Required unless `move_inputs = TRUE`,
#'   which fixes the device per entry.
#'
#'   A device is identified by its object -- an array's `$device`, or whatever
#'   this returns -- canonicalized per dispatcher: object identity first, and
#'   [identical()] as a fallback, so equal-but-distinct device objects count
#'   as one device. Interning device objects (one object per device, alive for
#'   the session, as [`as_pjrt_device()`] does) is therefore not required, but
#'   it is recommended: an interned device resolves in one pointer comparison,
#'   and each distinct object a backend hands out stays alive for the
#'   dispatcher's lifetime.
#' @param extractor (`function` | `NULL`)\cr
#'   Reads a non-`"xla"` array leaf's metadata via the backend's accessors,
#'   called as `extractor(leaf)` and returning
#'   `list(aval = list(dtype, shape, ambiguous), device, backend)` -- `dtype` a
#'   tengen `DataType`, `shape` an `integer()`, `device` the interned device
#'   object, `backend` the leaf's tag. Required for any backend other than
#'   `"xla"`; ignored for `"xla"`, where the dtype/shape/device come off the
#'   `PJRTBuffer` in `$data` directly. Only `$data` is assumed on the leaf; every
#'   other property is obtained through this function, so a backend need not
#'   store them as fields (see the `AnvlBackend` contract in anvl).
#' @export
dispatcher <- function(
  capacity,
  compile,
  static = character(),
  backend = "xla",
  move_inputs = FALSE,
  default_device = NULL,
  extractor = NULL
) {
  checkmate::assert_count(capacity, positive = TRUE)
  checkmate::assert_function(compile)
  checkmate::assert_character(static, any.missing = FALSE)
  checkmate::assert_string(backend, min.chars = 1L)
  checkmate::assert_flag(move_inputs)
  checkmate::assert_function(default_device, null.ok = TRUE)
  checkmate::assert_function(extractor, null.ok = TRUE)
  if (!move_inputs && is.null(default_device)) {
    cli::cli_abort(
      "{.arg default_device} is required unless {.code move_inputs = TRUE}."
    )
  }
  # The backend fixes the execution engine: "xla" runs a compiled PJRT
  # executable natively; any other backend runs through the compiled R closure.
  engine <- if (backend == "xla") "pjrt" else "closure"
  if (engine == "closure" && is.null(extractor)) {
    cli::cli_abort(
      "{.arg extractor} is required for a non-{.val xla} backend: pass a
       function that reads a leaf's metadata via the backend's accessors."
    )
  }
  impl_dispatch_create(
    as.integer(capacity),
    compile,
    as.character(static),
    engine,
    backend,
    move_inputs,
    default_device,
    extractor
  )
}

#' @title Native eager-dispatch fast path
#' @description
#' Dispatch a call through a [`dispatcher()`]'s executable cache, compiling on a
#' miss. An input the dispatcher cannot execute is an error, named after the
#' offending argument; there is no fallback for the caller to take.
#' @rdname dispatch
#' @param dispatcher (`Dispatcher`)\cr A dispatcher from [`dispatcher()`].
#' @param args (`list`)\cr The (already evaluated) argument list of the call.
#' @return [`dispatch()`] returns the call's finished result. With
#'   `backend = "xla"` that is the output buffers wrapped into `"AnvlArray"`s
#'   and re-nested by the `compile` callback's `out_tree` (see
#'   [`dispatcher()`]); with any other backend it is whatever the compiled
#'   closure returned.
#' @export
# Bound directly to the generated native entry (RcppExports.R precedes this
# file in the Collate order): dispatch() is on anvl's per-primitive hot path,
# and a plain forwarding wrapper would add an R call frame per dispatch.
dispatch <- impl_dispatch_run

#' @rdname dispatch
#' @return [`dispatcher_size()`] returns the number of compiled executables
#'   the dispatcher currently caches.
#' @export
dispatcher_size <- impl_dispatcher_size

#' @export
print.Dispatcher <- function(x, ...) {
  cat("<Dispatcher>\n")
  invisible(x)
}

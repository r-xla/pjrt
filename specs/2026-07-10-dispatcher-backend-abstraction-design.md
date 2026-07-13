# Dispatcher backend abstraction and native output wrapping

Date: 2026-07-10
Status: implemented on `feat/native-dispatch-clean` (pjrt) and
`perf/dispatch-overhead` (anvl)

## Problem

The native dispatcher (`src/dispatch.cpp`) is the cache + dispatch hot path
behind anvl's `jit()`. It works, but:

1. **Backend-specific behaviour is scattered.** `impl_dispatch_run()` is one
   ~400-line function in which the two engines ("pjrt" executables vs. a
   compiled R closure) are `if (closure)` branches, and the closure engine
   hardcodes `"quickr"` as the only array backend it accepts. A third backend
   cannot plug in.
2. **Output wrapping lives in anvl.** For the pjrt engine, `dispatch()`
   returns raw buffers plus wrap material (`out_tree`, `out_dtypes`,
   `out_shapes`, `ambiguous_out`, `device`), and anvl loops over them in R
   (`jit_wrap_outputs_native()`) to build `AnvlArray`s. That is a second
   source of truth for the array-wrapper layout (pjrt reads the fields in
   `anvl_fields()`, anvl writes them) and R-loop cost on every call.

## Goals

* The dispatcher core — flatten, classify, key, cache, miss protocol — is
  backend-agnostic. The pjrt/XLA fast path is an implementation detail behind
  an abstraction; other backends (e.g. quickr) plug in and stay reasonably
  fast.
* A backend controls its own implementation details: how an aval is read off
  one of its arrays, what a cache entry holds, how a call executes, how
  outputs are wrapped.
* `dispatch()` returns the *finished* call result. Output wrapping happens
  natively, inside the dispatcher.
* Single source of truth: the array-wrapper field contract (`$data`,
  `$dtype`, `$shape`, `$device`, `$ambiguous`, `$backend`, class
  `"AnvlArray"`) is read **and** written in pjrt; anvl's R-level wrap code is
  deleted.

## Design

### Engine abstraction (C++)

A per-dispatcher `Engine` object (virtual class — the "C++ closure"
parametrization; one indirect call per hook, invisible next to the ~5us
dispatch cost) with the hooks a backend may want to own:

```
class Engine {
  // The engine's per-entry material, built from the compile callback's
  // result. Entry SEXPs are rooted via CacheEntry::preserve().
  virtual void build_entry(const Rcpp::List& res, CacheEntry& e) = 0;
  // Execute a call against a cached entry; returns the finished R value.
  virtual SEXP run(CacheEntry& e, const CacheKey& key,
                   const std::vector<SEXP>& exec_sexp) = 0;
  // The aval of an AnvlArray leaf of this dispatcher's backend. Default
  // implementation reads the $dtype / $shape fields every AnvlArray
  // carries; a native engine can read cheaper, authoritative metadata.
  virtual aval array_aval(const AnvlFields& af, ...);
  virtual bool supports_move_inputs() const;
};
```

Two implementations:

* **`PjrtEngine`** (the fast path, `engine = "pjrt"`): `$data` is a
  `PJRTBuffer`; avals come off the buffer's cached native metadata; `run()`
  assembles inputs natively (const arrays, device moves, R-data uploads,
  phantom donation buffers), executes, and wraps the outputs natively (below).
* **`ClosureEngine`** (`engine = "closure"`, the generic vehicle for any
  non-pjrt backend): the entry holds a compiled R closure; `run()` calls it
  on the flat leaves and returns its value verbatim. Wrapping — or not
  wrapping (quickr's `unwrap = TRUE`) — is the closure's business.

The dispatcher core never branches on the engine; it calls the hooks. A new
native backend is a new `Engine` subclass; a new R-level backend needs no C++
at all — it uses `ClosureEngine` with its own `backend` name.

### Device identity is an engine detail

The cache key's `DeviceToken` stays a `const void*` (compared and folded,
never dereferenced), but how a device object maps to it is the engine's
business: `Engine::canonical_device()` resolves by object identity first —
one pointer compare for a backend that interns its devices — and falls back
to `identical()`, preserving each distinct device object for the dispatcher's
lifetime. Interning is therefore a fast path, not a contract: a backend
handing out equal-but-distinct device objects collapses to one token instead
of being rejected. A native engine with its own identity scheme can override
the hook.

### The `backend` parameter

`dispatcher()` gains a `backend` argument: the `$backend` tag every
`AnvlArray` input must carry. It defaults to `"xla"` for `engine = "pjrt"`
and is required for `engine = "closure"` (quickr passes `"quickr"`); the
closure engine no longer hardcodes quickr. Because each dispatcher has its
own cache and its backend is fixed at creation, `CacheKey::backend` is
dropped from the key (it could never differ between two keys of one cache).

### Native output wrapping (pjrt engine)

For the pjrt engine, everything about an output except its buffer is fixed
per cache entry: dtype, shape, device, ambiguity, backend tag, and the output
tree. So:

* On an entry's **first** execution, `run()` reads each output's dtype/shape
  off the buffers and builds one *template* `AnvlArray` per output — a named
  VECSXP `(data = NULL, dtype = <tengen DataType>, shape, device, ambiguous,
  backend = "xla")` with class `"AnvlArray"`. The tengen dtype object is
  obtained by calling `tengen::as_dtype()` once per output (miss-path cost).
  Templates are preserved on the entry.
* On **every** call, each output is `Rf_shallow_duplicate()` of its template
  with the buffer written into `$data`, and the flat list is rebuilt into the
  caller's structure natively via `unflatten_rec()` over the entry's
  `out_tree` (an `RTree` xptr the compile callback supplies — anvl's
  `graph$out_tree`).

The compile-callback contract for `engine = "pjrt"` becomes: `exec`,
`client`, `device`, `out_tree` **required**; `const_arrays`,
`phantom_specs`, `ambiguous_out` optional. `out_dtypes` / `out_shapes` /
`buffers` disappear from the `dispatch()` result — `dispatch()` returns the
wrapped, unflattened outputs.

### File layout (src/)

* `dispatch_key.h` — key material: `AnvlDtype`, `aval`, `KeyLeaf`,
  `CacheKey`, hash/equality (unchanged apart from dropping
  `CacheKey::backend`).
* `dispatch_engine.h` / `dispatch_engine.cpp` — the array-wrapper contract
  (`AnvlFields`, `classify_rdata`), `CacheEntry`, the `Engine` interface, and
  the `PjrtEngine` / `ClosureEngine` implementations.
* `dispatch.cpp` — the backend-agnostic core: `Dispatcher` (cache + engine +
  policies), flatten/classify/key, the miss protocol, and the Rcpp entry
  points.

### anvl side (perf/dispatch-overhead)

* `run <- function(args) dispatch(dispatcher, args)` for both backends —
  `jit_wrap_outputs_native()`, the memoised dtype table, and the `$value`
  indirection are deleted.
* `jit_quickr_impl()` passes `backend = "quickr"`.

### Testing

* Existing `tests/testthat/test-dispatch.R` and `src/test-dispatch.cpp`
  updated to the new result contract (wrapped outputs / direct closure
  value).
* anvl moves into pjrt's `Suggests`, and a new
  `tests/testthat/test-dispatch-anvl.R` exercises the dispatcher indirectly
  through `anvl::jit()` end-to-end (wrapping, nested output trees, statics,
  literals, caching, quickr when installed). Skipped when anvl is not
  installed.

## Alternatives considered

* **Struct of `std::function` hooks** instead of a virtual class: same
  performance, but no natural home for engine-specific entry data and noisier
  ownership.
* **Reorganize the `if (closure)` branches into helpers** without an
  interface: least change, but leaves the closure engine hardcoded to quickr
  and the wrap in anvl — misses the point.
* **R-callback output wrapping** (dispatcher calls an anvl-supplied wrap
  function): keeps the layout defined in anvl, but pays an R call per
  dispatch and keeps two sources of truth for the field contract.

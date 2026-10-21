@../claude-config/CLAUDE.md

## Package Overview

`pjrt` is the runtime layer of the r-xla stack. It compiles StableHLO/MLIR programs to hardware-specific executables and runs them via the PJRT C API. It supports CPU and CUDA backends through dynamically loaded plugins.

Beyond the runtime, pjrt also owns the **Rtree module** (`build_tree()`/`flatten()`/`unflatten()` and the structural tree ops in `src/tree.h`/`src/tree.cpp`/`R/tree.R`); trees are opaque `RTree` external pointers. The Rtree is pjrt's R analog of [JAX's pytree](https://docs.jax.dev/en/latest/pytrees.html), which is where the idea comes from.

It also owns the **Dispatcher** (`dispatcher()`/`dispatch()`), the native eager-dispatch engine behind anvl's `jit()`: an executable cache keyed on the inputs' structure and abstract values, which calls back into R to compile only on a miss.

The dispatcher's C++ names anvl's data model -- the `"AnvlArray"` class, its `$data`/`$backend` fields, the `"plain"` backend tag, and the `AnvlDtype` vocabulary. That is a contract pjrt defines and anvl produces; it is deliberately *not* a package dependency. **pjrt must not depend on anvl, in `Suggests` or anywhere else.** `tests/testthat/test-dispatch.R` therefore drives the engine with its own fixtures (`parr()`, `qarr()`, `pjrt_entry()`), and the integration test that anvl's real callback matches this engine lives in anvl's `test-jit-dispatch.R`, which is the side of the dependency that can hold it.

## Core Design

### Object Hierarchy

All core objects are S3 classes backed by C++ external pointers (via Rcpp):

```
PJRTPlugin       – loaded shared library (.so/.dylib) providing the PJRT C API
  └─ PJRTClient  – owns devices, creates buffers, compiles programs
       └─ PJRTDevice  – a single device (e.g. "cpu:0", "cuda:1")
```

```
PJRTProgram            – MLIR or HLO source code ready for compilation
  └─ PJRTLoadedExecutable  – compiled, device-specific executable
```

```
PJRTBuffer         – device-resident tensor (the main data object)
PJRTElementType    – dtype enum (bool, i8–i64, ui8–ui64, f32, f64)
```

A `PJRTBuffer` is an external pointer whose protected slot holds an ALTREP *carrier* (`src/buffer_sexp.{h,cpp}`). The carrier holds the CPU keepalive and gives buffers a custom serialization (host bytes plus dtype and shape, re-uploaded on load to `the[["unserialize_device"]]`, else the default device -- the device is deliberately not saved; anvl writes that entry from its `.onLoad()` so that buffers load onto its own default device; see `buffer_serialized_state()`/`buffer_unserialize()` in `buffer.R`). Since R tracks external pointers by reference, a buffer referenced several times is written once and loads as one object. A loaded buffer is a NULL pointer until its first use, when `rpjrt::buffer_xptr()` moves the uploaded buffer into it, so C++ must reach a buffer only through `rpjrt::buffer_xptr()`/`rpjrt::as_buffer()`, and its keepalive only through `rpjrt::buffer_keepalive()`/`rpjrt::set_buffer_keepalive()` -- never `R_ExternalPtrAddr()`/`R_ExternalPtrProtected()` directly. C++ creates buffer objects with `rpjrt::wrap_buffer()`.

### Plugin and Client Lifecycle

Plugins and clients are singletons cached in a global environment (`the` in plugin.R):

1. `pjrt_client(platform)` checks the cache, otherwise loads the plugin and creates a client.
2. Plugin loading: checks `PJRT_PLUGIN_PATH_<PLATFORM>` env var, falls back to downloading from zml/pjrt-artifacts into `R_user_dir("pjrt", "cache")`.
3. One client per platform — calling `pjrt_client()` again returns the cached instance.

### Execution Pipeline

The typical workflow is: create program → compile → create buffers → execute → read results.

```r
prog <- pjrt_program(src, format = "mlir")       # PJRTProgram
exec <- pjrt_compile(prog, device = "cpu")        # PJRTLoadedExecutable
buf  <- pjrt_buffer(data, dtype = "f32")           # PJRTBuffer (host → device)
out  <- pjrt_execute(exec, buf)                    # PJRTBuffer (on device, may not be ready)
arr  <- as_array(out)                              # R array (device → host, blocks)
```

### Async Model

Both execution and buffer transfers are asynchronous:

- `pjrt_execute()` returns `PJRTBuffer`s that may not be ready yet. PJRT tracks dependencies internally, so unready buffers can be passed directly as inputs to the next execution.
- `as_array_async()` returns a plain-looking R array that is an ALTREP *host array* (`src/host_array.cpp`): it has its final type, `dim` and class at once, and the first read of its data waits for the device-to-host transfer and materializes it.
- `is_ready()` polls without blocking; `await()` blocks until complete (and materializes a host array). On any other R object, `is_ready()` is `TRUE` and `await()` is a no-op.

### Column-Major Convention

R uses column-major (Fortran) order. The C++ layer handles row-to-column-major conversion when transferring between R and device buffers.

## Key Source Files

- `plugin.R` – plugin loading, client caching, global state (`the`)
- `client.R` – `pjrt_client()`, `pjrt_compile()`
- `buffer.R` – `pjrt_buffer()`, `pjrt_scalar()`, `pjrt_empty()`, type dispatch
- `loaded_executable.R` – `pjrt_execute()`
- `async.R` – `is_ready()`, `await()`
- `device.R` – `pjrt_device()`, device spec parsing ("cpu:0")
- `program.R` – `pjrt_program()` (MLIR/HLO loading)
- `format.R` – buffer pretty-printing
- `tree.R` – Rtree API over the native `RTree` (`build_tree()`, `flatten()`, `unflatten()`, `map_tree()`, ...)
- `dispatch.R` – `dispatcher()`, `dispatch()`; the engine itself is C++ (`src/dispatch*.{h,cpp}`)
- `safetensors.R` – safetensors read/write integration
- `reexports.R` – xlamisc re-exports
- `src/` – Rcpp C++ layer wrapping the PJRT C API, plus protobuf for compile options

**Important:** Do not call `devtools::load_all()` and `devtools::test()` in the same R process. The protobuf descriptors get registered twice, causing a fatal `CHECK failed: GeneratedDatabase()->Add(...)` crash. Use separate `Rscript -e` calls instead.

## Memory management

An object's finalizer (e.g., `reg.finalizer()`) runs only when the object is **garbage collected**, not when its binding goes out of scope or is `rm()`-ed. A test that relies on a finalizer must call `gc()` explicitly to trigger it.

In C++, root R objects with Rcpp rather than by hand:

- Use Rcpp types (`Rcpp::List`, `Rcpp::CharacterVector`, ...) or `Rcpp::Shield<SEXP>` instead of `PROTECT`/`UNPROTECT`. They unwind correctly when a C++ exception is thrown, where a skipped `UNPROTECT` would not. The exception is ALTREP methods, which R calls from C and which must not hold objects with destructors (see `call_pjrt()` in `src/buffer_sexp.h`).
- Keep an object alive beyond the current call with `rpjrt::preserve()` and `rpjrt::release()`, or `rpjrt::queue_release()` from a PJRT callback thread (`src/deferred_release.h`). These use Rcpp's precious list; don't call `R_PreserveObject()`/`R_ReleaseObject()`.

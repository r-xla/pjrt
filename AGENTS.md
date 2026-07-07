@../claude-config/CLAUDE.md

## Package Overview

`pjrt` is the runtime layer of the r-xla stack. It compiles StableHLO/MLIR programs to hardware-specific executables and runs them via the PJRT C API. It supports CPU, CUDA, and Metal backends through dynamically loaded plugins.

Beyond the runtime, pjrt also owns the **Rtree module** (`build_tree()`/`flatten()`/`unflatten()` and the structural tree ops in `src/tree.h`/`src/tree.cpp`/`R/tree.R`); trees are opaque `RTree` external pointers. The Rtree is pjrt's R analog of [JAX's pytree](https://docs.jax.dev/en/latest/pytrees.html), which is where the idea comes from.

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
PJRTArrayPromise   – async device-to-host transfer result (environment-based)
PJRTElementType    – dtype enum (pred, i8–i64, ui8–ui64, f32, f64)
```

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
- `as_array_async()` returns a `PJRTArrayPromise` (non-blocking device-to-host transfer).
- `is_ready()` polls without blocking; `await()` and `value()` block until complete.

### Column-Major Convention

R uses column-major (Fortran) order. The C++ layer handles row-to-column-major conversion when transferring between R and device buffers.

## Key Source Files

- `plugin.R` – plugin loading, client caching, global state (`the`)
- `client.R` – `pjrt_client()`, `pjrt_compile()`
- `buffer.R` – `pjrt_buffer()`, `pjrt_scalar()`, `pjrt_empty()`, type dispatch
- `loaded_executable.R` – `pjrt_execute()`
- `async.R` – `value()`, `is_ready()`, `await()`, `PJRTArrayPromise`
- `device.R` – `pjrt_device()`, device spec parsing ("cpu:0")
- `program.R` – `pjrt_program()` (MLIR/HLO loading)
- `format.R` – buffer pretty-printing
- `tree.R` – Rtree API over the native `RTree` (`build_tree()`, `flatten()`, `unflatten()`, `map_tree()`, ...)
- `safetensors.R` – safetensors read/write integration
- `reexports.R` – tengen re-exports
- `src/` – Rcpp C++ layer wrapping the PJRT C API, plus protobuf for compile options

**Important:** Do not call `devtools::load_all()` and `devtools::test()` in the same R process. The protobuf descriptors get registered twice, causing a fatal `CHECK failed: GeneratedDatabase()->Add(...)` crash. Use separate `Rscript -e` calls instead.

## Memory management

An object's finalizer (e.g., `reg.finalizer()`) runs only when the object is **garbage collected**, not when its binding goes out of scope or is `rm()`-ed. A test that relies on a finalizer must call `gc()` explicitly to trigger it.

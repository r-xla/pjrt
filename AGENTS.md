@../claude-config/AGENTS.md

## Package Overview

`pjrt` is the runtime layer of the r-xla stack. It compiles StableHLO/MLIR programs to hardware-specific executables and runs them via the PJRT C API. It supports CPU, CUDA, and Metal backends through dynamically loaded plugins.

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

Plugins and clients are singletons cached in a global hashtable (`the` in plugin.R):

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

### Tengen Integration

`pjrt` implements S3 methods for tengen generics on `PJRTBuffer`: `shape()`, `dtype()`, `device()`, `as_array()`, `as_raw()`. These are re-exported so users don't need to load tengen directly.

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
- `safetensors.R` – safetensors read/write integration
- `reexports.R` – tengen re-exports
- `src/` – Rcpp C++ layer wrapping the PJRT C API, plus protobuf for compile options

## Configuration

Environment variables:
- `PJRT_PLATFORM` – default platform ("cpu", "cuda", "metal"), defaults to "cpu"
- `PJRT_PLUGIN_PATH_<PLATFORM>` – override plugin shared library path
- `PJRT_CPU_DEVICE_COUNT` – number of CPU devices (default: 1)

R options:
- `pjrt.print_max_rows`, `pjrt.print_max_width` – buffer display limits

## Testing

Tests require the CPU plugin to be downloaded. Most tests skip otherwise via `skip_if_not_downloaded_pjrt()` in `helper-skip.R`. The test setup (`setup.R`) sets `PJRT_CPU_DEVICE_COUNT=2` for multi-device testing.

Test files mirror the object hierarchy: `test-plugin.R`, `test-client.R`, `test-buffer.R`, `test-device.R`, `test-loaded_executable.R`, etc. Snapshot tests cover buffer formatting output.

```r
devtools::test()
testthat::test_file("tests/testthat/test-buffer.R")
```

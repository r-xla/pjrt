@../claude-config/CLAUDE.md

## Package Overview

`pjrt` is the runtime layer of the r-xla stack. It compiles StableHLO/MLIR programs to hardware-specific executables and runs them via the PJRT C API. It supports CPU, CUDA, and Metal backends through dynamically loaded plugins.

Beyond the runtime, pjrt also owns the **Rtree module** (`build_tree()`/`flatten()`/`unflatten()` and the structural tree ops in `src/tree.h`/`src/tree.cpp`/`R/tree.R`); trees are opaque `RTree` external pointers. The Rtree is pjrt's R analog of [JAX's pytree](https://docs.jax.dev/en/latest/pytrees.html), which is where the idea comes from.

It also owns the **Dispatcher** (`dispatcher()`/`dispatch()`), the native eager-dispatch engine behind anvl's `jit()`: an executable cache keyed on the inputs' structure and abstract values, which calls back into R to compile only on a miss.

The dispatcher's C++ names anvl's data model -- the `"AnvlArray"` class, its `$data`/`$backend`/`$device` fields, the `"plain"` backend tag, and the `AnvlDtype` vocabulary. That is a contract pjrt defines and anvl produces; it is deliberately *not* a package dependency. **pjrt must not depend on anvl, in `Suggests` or anywhere else.** `tests/testthat/test-dispatch.R` therefore drives the engine with its own fixtures (`parr()`, `qarr()`, `pjrt_entry()`), and the integration test that anvl's real callback matches this engine lives in anvl's `test-jit-dispatch.R`, which is the side of the dependency that can hold it.

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
- `dispatch.R` – `dispatcher()`, `dispatch()`; the engine itself is C++ (`src/dispatch*.{h,cpp}`)
- `safetensors.R` – safetensors read/write integration
- `reexports.R` – tengen re-exports
- `cuda_kernels.R` – `pjrt_cuda_kernels_available()`
- `src/` – Rcpp C++ layer wrapping the PJRT C API, plus protobuf for compile options
- `src/cuda/` – CUDA kernels pjrt ships itself (see below)

**Important:** Do not call `devtools::load_all()` and `devtools::test()` in the same R process. The protobuf descriptors get registered twice, causing a fatal `CHECK failed: GeneratedDatabase()->Add(...)` crash. Use separate `Rscript -e` calls instead.

## Shipping a CUDA kernel

Most of pjrt's GPU work is cuSOLVER, reached by `dlopen` (`src/ffi_cuda.h`).
`src/cuda/` is for the cases where that is not enough and pjrt needs device
code of its own.

To add one, write three things; `lu_pivots_to_permutation` is the worked
example of each.

1. `src/cuda/<name>.h` -- one declaration of the entry point, spelling
   `__global__` as `PJRT_CUDA_KERNEL` (from `src/cuda/kernel.h`) so the host
   compiler can read it too.
2. `src/cuda/<name>.cu` -- includes that header and defines the kernel. nvcc
   then checks the definition against the declaration. The build picks the
   file up on its own.
3. The launch, in the platform's FFI handler: include the header and
   instantiate `Kernel<decltype(<symbol>)>` with the `extern "C"` symbol name
   (`src/cuda_kernels.h`), then call it with the grid, block, stream and the
   kernel's arguments.

The reason for the header, rather than just calling `cuda_kernel()` and
`cuda_launch()` directly: the device code is compiled separately into a fatbin
and an `extern "C"` symbol carries no argument types, so a handler that
disagrees with its kernel produces neither a compile nor a link error -- just a
`cuLaunchKernel` reading the wrong bytes. Deriving the launch signature from a
declaration that the `.cu` is also held to puts both ends back under the
compiler.

What happens underneath:

- `configure` finds an `nvcc` -- from `PJRT_NVCC`, the `cuda12.8` R package,
  `CUDA_HOME`, or the `PATH` -- and compiles each `.cu` into a fatbin holding
  cubins for every architecture the PJRT CUDA plugin supports, plus PTX so a
  newer GPU still runs. The architecture list is copied from JAX's `.bazelrc`;
  `PJRT_CUDA_SM_ARCHS` / `PJRT_CUDA_PTX_ARCH` override it.
- `tools/embed_fatbin.R` turns those into `src/cuda_fatbin.cpp`, so the device
  code lives in `pjrt.so` and nothing has to find a file at run time.
- `src/cuda_kernels.cpp` hands a fatbin to `cuModuleLoadData` on first use and
  caches the module per CUDA context.

Two constraints worth keeping:

- **No link-time CUDA.** Everything goes through `libcuda.so.1` via `dlopen`,
  never `libcudart`, so the package builds with no CUDA installed and loads on
  machines with no GPU. Writing a `.cu` that uses the CUDA *runtime* API or the
  `<<<>>>` launch syntax would break that; use the driver API from the handler
  instead.
- **A missing `nvcc` is not a build failure.** The package installs without the
  kernels and `pjrt_cuda_kernels_available()` returns `FALSE`. A handler that
  needs one then fails with an explanation, and callers that can lower the same
  computation another way should check first -- anvl's `pivots_to_permutation()`
  does.

## Memory management

An object's finalizer (e.g., `reg.finalizer()`) runs only when the object is **garbage collected**, not when its binding goes out of scope or is `rm()`-ed. A test that relies on a finalizer must call `gc()` explicitly to trigger it.

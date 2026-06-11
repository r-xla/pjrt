@../claude-config/CLAUDE.md

## Package Overview

`pjrt` is the runtime layer of the r-xla stack. It compiles StableHLO/MLIR programs to hardware-specific executables and runs them via the PJRT C API. It supports CPU, CUDA, and Metal backends through dynamically loaded plugins.

## Design Documentation

Core design elements are documented in `specs/design/`:

- `specs/design/core-design/` — object hierarchy, execution pipeline, async model, and the column-major convention.
- `specs/design/memory-management/` — host/device buffer memory management and layout-aware readback.

Each folder has a markdown document and an accompanying `.svg` diagram. **Consult the relevant design docs before making big architectural changes, and update them afterwards** so they stay in sync with the implementation.
Don't read the SVG documents, only the markdown documents, but update both.
The SVGs are for the humans to understand the code.

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

**Important:** Do not call `devtools::load_all()` and `devtools::test()` in the same R process. The protobuf descriptors get registered twice, causing a fatal `CHECK failed: GeneratedDatabase()->Add(...)` crash. Use separate `Rscript -e` calls instead.

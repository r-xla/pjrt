# XLA IR Inspection — Design

**Date:** 2026-07-02
**Branch:** `feat/xla-ir-inspection`
**Package:** `pjrt`
**Related issue:** [r-xla/pjrt#194](https://github.com/r-xla/pjrt/issues/194)

## Goal

Let users inspect the intermediate representations the XLA compiler produces for a
program — primarily the **input HLO** (before optimization) and the **optimized
HLO** (after optimization), with the option to see **every compiler pass** — so
they can debug what the compiler did to their program.

Issue #194 also asks for **xprof** profiling. That is a substantially larger,
independent subsystem (see "Out of scope" / "Follow-up") and is **deferred to its
own spec**. This spec covers IR inspection only.

## Background: how the IR is actually obtained

We drive the XLA compiler's own **dump mechanism** (`DebugOptions.xla_dump_to`)
and read the files it writes. We do **not** reconstruct HLO on the client side.

Why: the PJRT C API's only in-memory route to the optimized program,
`PJRT_Executable_OptimizedProgram`, returns a *serialized* `HloModuleProtoWithConfig`.
Turning that into the readable HLO assembly syntax requires XLA's
`HloModule::CreateFromProtoWithConfig` + `HloModule::ToString()` — i.e. linking a
large slice of the XLA C++ compiler (HLO IR lib + abseil + tsl, or MLIR for the
`mlir` format). That is incompatible with our `dlopen`-a-plugin, link-nothing-of-XLA
model. The protobuf alone only yields `DebugString()` (verbose protobuf
text-format), not HLO syntax.

The dump mechanism sidesteps this entirely: the pretty-printing happens **inside
the plugin**, which already has XLA linked, so `xla_dump_to` writes the exact
`HloModule::ToString()` text JAX shows — for free, without us linking XLA. The
cost is that it goes via disk.

**Ecosystem precedent** (both are PJRT-C-API consumers with our exact constraint):
- **gomlx/gopjrt** (`pjrt/compile.go`): attaches an `xla.DebugOptions` proto to
  `ExecutableBuildOptions.DebugOptions` and relies on the plugin to dump; points
  the user at the dump dir. No client-side reconstruction, no file readback.
- **zml** (`zml/module.zig`): exposes `xla_dump_to`, `xla_dump_hlo_pass_re`, etc.
  as first-class compilation options; same dump-to-disk approach.

Neither reads the dumped IR back into the host language. Our high-level readback
helper (below) is a deliberate ergonomic step beyond both.

## Architecture

Two layers on top of one new native entry point.

### C++ (one new Rcpp export)

`impl_compile_options_set_dump(compile_options, dump_to, hlo_as_text,
hlo_as_proto, hlo_as_dot, hlo_as_html, pass_re, module_re)`

- Mutates
  `compile_options.mutable_executable_build_options()->mutable_debug_options()`,
  setting the corresponding `DebugOptions` fields (all confirmed present in
  `inst/proto/xla/xla.proto`): `xla_dump_to` (109), `xla_dump_hlo_module_re`
  (110), `xla_dump_hlo_pass_re` (111), `xla_dump_hlo_as_text` (112),
  `xla_dump_hlo_as_proto` (113), `xla_dump_hlo_as_dot` (114),
  `xla_dump_hlo_as_html` (116).
- String fields are only set when non-empty; bool fields set as given.
- No change to the serialize → `PJRT_Client_Compile` path; the debug options ride
  along in the already-serialized `CompileOptionsProto`.

This is the only new native code.

### R — low-level escape hatch

- `pjrt_dump_options(dir = NULL, hlo_as_text = TRUE, hlo_as_proto = FALSE,
  hlo_as_dot = FALSE, hlo_as_html = FALSE, pass_re = NULL, module_re = NULL)`
  → validated S3 object of class `PJRTDumpOptions` (a plain, checked list).
  - `dir`: destination directory (`xla_dump_to`). Created if it does not exist.
  - Format flags map to the `xla_dump_hlo_as_*` booleans.
  - `pass_re = ".*"` turns on per-pass dumping; `module_re` filters modules.
- New `dump = NULL` argument on `pjrt_compile()`:
  - When a `PJRTDumpOptions` is supplied, `pjrt_compile()` calls
    `impl_compile_options_set_dump()` on the (default or user-provided)
    compile-options object before compiling.
  - Chosen over exporting the currently-internal `new_compile_options()` /
    `new_build_options()` machinery: it keeps the surface small while letting any
    compile dump wherever/however the user wants.

### R — high-level helper

`pjrt_dump_hlo(program, device = NULL, passes = FALSE)` → `PJRTHLODump`

- Resolves `device` exactly as `pjrt_compile()` does (default device when `NULL`).
- Creates a fresh temp dir (`tempfile("pjrt_hlo_")`).
- Compiles `program` for `device` with
  `pjrt_dump_options(dir = <tmp>, hlo_as_text = TRUE, pass_re = if (passes) ".*")`.
  (`xla_dump_to` with no format flag already defaults to text; we set
  `hlo_as_text = TRUE` explicitly.)
- Reads the resulting `*.txt` files back and returns a `PJRTHLODump` object.
- The compiled executable is discarded — this function is for inspection only.

## Return object: `PJRTHLODump`

A named list keyed by **stage**, each value the file's text (character scalar):

- Always present when compilation dumps: `before_optimizations`,
  `after_optimizations`.
- When `passes = TRUE`: one entry per dumped pass, keyed by the pass label parsed
  from the filename.

XLA dump filenames look like
`module_0000.<name>.before_optimizations.txt`,
`module_0000.<name>.after_optimizations.txt`, and (with `pass_re`)
`module_0000.<name>.<NNNN>.<pass_name>.{before,after}.txt`. The helper parses the
stage/pass label out of the filename; the module name is captured as an attribute.

Attributes:
- `attr(x, "dir")`: the dump directory (so power users can inspect raw files).
- `attr(x, "module")`: the module name parsed from the filenames.

S3 methods (with `#' @export` + `devtools::document()`):
- `print.PJRTHLODump` / `format.PJRTHLODump`: summary — module name, ordered list
  of stages with per-stage line counts; not the full text.
- `[[.PJRTHLODump` / `$.PJRTHLODump`: return a stage's text (inherited list
  behaviour is sufficient; only override if needed for nicer errors on unknown
  stage).
- `as.character.PJRTHLODump`: the `after_optimizations` text (the single most
  useful artifact).

## Error handling

- `pjrt_dump_options()` validates types and that at least one format flag is
  `TRUE` when used directly; invalid `dir` (non-string) errors early.
- `pjrt_compile(dump=)` errors if `dump` is not a `PJRTDumpOptions`.
- `pjrt_dump_hlo()`: if, after compiling, the temp dir contains **no** dump files,
  raise an informative error pointing at the plugin-honours-proto risk (below) —
  this is the signal that the fallback is needed.

## Primary risk + spike

**Risk:** a plugin might honour `xla_dump_to` only via the `XLA_FLAGS` env var and
ignore the `DebugOptions` proto passed through compile options.

**Spike (first implementation step):** compile a trivial program on the CPU
plugin with `xla_dump_to` set via the proto and confirm files appear. gomlx sets
`DebugOptions` on the proto and it works for it, so this is expected to pass.

**Fallback (same public API):** if the proto is ignored, have `pjrt_dump_hlo()`
set `XLA_FLAGS=--xla_dump_to=<tmp> --xla_dump_hlo_as_text[...]` around the compile
(save/restore the env var). The R API is unchanged either way. The low-level
`pjrt_dump_options()` + `pjrt_compile(dump=)` path still sets the proto (correct
if/when plugins honour it).

## Testing (CPU backend)

- `pjrt_dump_hlo()` on a trivial program (e.g. `add`) returns a `PJRTHLODump`
  containing `before_optimizations` and `after_optimizations`; both texts
  non-empty and mention the expected op.
- `passes = TRUE` yields strictly more stages than the default.
- `pjrt_dump_options(dir = d)` + `pjrt_compile(dump = ...)` writes files into `d`.
- `pjrt_dump_options()` input validation (bad `dir`, no formats) errors.
- `print.PJRTHLODump` snapshot (module name + stage list; avoid embedding
  full, potentially version-dependent HLO text).

Build/test note (from CLAUDE.md): never `devtools::load_all()` and
`devtools::test()` in the same R process (protobuf double-registration crash);
use separate `Rscript -e` calls.

## Documentation

- Roxygen for `pjrt_dump_hlo()`, `pjrt_dump_options()`, the new `dump` arg on
  `pjrt_compile()`, and the `print`/`format`/`as.character` methods; document
  return values.
- `devtools::document()` to regenerate `.Rd` + `NAMESPACE`.
- Add new exported functions to `_pkgdown.yml`.

## Out of scope / follow-up

- **xprof profiling** (issue #194, second half). Its own spec/branch. Reference
  implementation: **zml** (`zml/profiling/`, `docs/howtos/profiling.md`) uses the
  **PJRT Profiler extension** → captures an `XSpace` protobuf → `profiling.xplane.pb`
  (xprof/TensorBoard format) + Perfetto JSON, with host-side `TraceMe` scopes.
  For us this means vendoring `pjrt_c_api_profiler_extension.h` plus the
  `XSpace`/`ProfileOptions` protos — sized as its own slice.
- Linking XLA to render optimized HLO in-memory from
  `PJRT_Executable_OptimizedProgram` (rejected: enormous dependency).
- Non-text formats (proto/dot/html) in the **high-level** helper: reachable via
  the low-level `pjrt_dump_options()` flags; the helper stays text-focused.

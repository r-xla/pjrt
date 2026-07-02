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

## Why not the `DebugOptions` proto (design pivot)

The original plan set dump fields on `ExecutableBuildOptions.DebugOptions` in the
serialized `CompileOptionsProto`. Implemented and tested, this **does not work**:

- `xla.proto` is **proto3 with no custom defaults**, so a freshly
  `mutable_debug_options()`-constructed message is all-zero. The plugin takes it
  literally, overriding XLA's real defaults (`xla_backend_optimization_level` → 0,
  `xla_cpu_parallel_codegen_split_count` → 0, ...). Compilation then errors
  (*"Too many extra compilation parts ... parallel_codegen_split_count (0)"*), and
  even patched, the "optimized" HLO would not be genuinely optimized.
- XLA's real defaults live in the compiler's `DefaultDebugOptionsIgnoringFlags()`.
  There is no PJRT C-API way to fetch them, and reconstructing ~50
  version/backend-specific fields ourselves is a non-starter.
- **gomlx/gopjrt confirms this**: it sets `DebugOptions` only when the user
  supplies a *complete* `XLA_DEBUG_OPTIONS` prototext (else leaves it nil so the
  plugin defaults apply), and for dumping its docs tell users to set `XLA_FLAGS`.

So dumping must go through the flag path, which *merges* onto the compiler
defaults instead of replacing them.

## Architecture

Pure-R, no native code. The compiler dumps; we configure it via `XLA_FLAGS` and
read the files back.

### The `XLA_FLAGS` constraint

XLA parses `XLA_FLAGS` **once, before the first compilation** in the process
(measured: setting it after a prior compile dumps nothing). The design works with
this rather than against it.

### R — the single entry point

`pjrt_dump_hlo(program, device = NULL, passes = FALSE)` → `PJRTHLODump`

- Determines the dump dir from the current `XLA_FLAGS` (`--xla_dump_to=`); if
  absent, best-effort sets `XLA_FLAGS` to a fresh temp dir (with
  `--xla_dump_hlo_as_text`, plus `--xla_dump_hlo_pass_re=.*` when `passes`), which
  takes effect only if this is the process's first compilation.
- Snapshots the dir, compiles `program` for `device` (via `pjrt_compile()`), and
  diffs for the files this compilation produced.
- Parses those files into a `PJRTHLODump`. If nothing was dumped (a prior compile
  already fixed `XLA_FLAGS`), raises an actionable error telling the user to set
  `XLA_FLAGS=--xla_dump_to=...` before starting R.
- The compiled executable is discarded — this function is for inspection only.

Internal helpers (unexported): `xla_dump_dir_from_flags()`, `parse_hlo_dump()`.

### Low-level escape hatch

Just `XLA_FLAGS` itself (documented), exactly like gomlx/zml. No wrapper object —
the proto route it would have wrapped is unusable, so wrapping it would be a
footgun.

## Return object: `PJRTHLODump`

A named list keyed by **stage**, each value the file's text (character scalar):

- Always present when compilation dumps: `before_optimizations`,
  `after_optimizations`.
- When `passes = TRUE`: one entry per dumped pass, keyed by the numeric pass
  index + label parsed from the filename.

**Real filenames** (verified by spike on the CPU plugin — see "Spike results"):

- Input HLO: `module_<NNNN>.<name>.before_optimizations.txt`.
- Optimized HLO: **backend-prefixed** — `module_<NNNN>.<name>.<backend>_after_optimizations.txt`
  (e.g. `cpu_after_optimizations.txt`). The helper matches on the
  `after_optimizations.txt` *suffix* (so it works across backends) and maps it to
  the canonical `after_optimizations` key.
- Per-pass (with `pass_re`): `module_<NNNN>.<name>.<IIII>.<pipeline>.after_<prev>.before_<next>.txt`,
  where `<IIII>` is a zero-padded sequence index. The helper keys these by
  `<IIII>` + the descriptive label, preserving compiler order.

The `after_optimizations.txt` suffix match must **exclude** sibling artifacts that
share the `cpu_after_optimizations` prefix but add a hyphenated suffix
(`-buffer-assignment.txt`, `-memory-usage-report.txt`) — match files *ending in*
`after_optimizations.txt` exactly. The module name is captured as an attribute.

Attributes:
- `attr(x, "dir")`: the dump directory (so power users can inspect raw files).
- `attr(x, "module")`: the module name parsed from the filenames.

S3 methods (with `#' @export` + `devtools::document()`):
- `print.PJRTHLODump` / `format.PJRTHLODump`: summary — module name, ordered list
  of stages with per-stage line counts; not the full text.
- `[[` / `$`: return a stage's text (inherited list behaviour; not overridden).
- `as.character.PJRTHLODump`: the `after_optimizations` text (the single most
  useful artifact).

## Error handling

- `pjrt_dump_hlo()` validates `passes` is a single `TRUE`/`FALSE`.
- If, after compiling, no HLO was dumped (missing `after_optimizations` stage),
  raise an actionable error pointing at the `XLA_FLAGS`-before-first-compile
  requirement.

## Spike results (2026-07-02, CPU plugin)

Ran real compiles of a two-op program (`add` then `multiply`) against the
installed `pjrt` 0.4.0.9000 + cached CPU plugin. Findings that shaped the design:

- **Proto route is honoured but unusable.** Setting the dump fields on
  `DebugOptions` via the serialized `CompileOptionsProto` took effect, but the
  all-zero proto3 message clobbered XLA's defaults and the compile errored
  (*parallel_codegen_split_count (0)*). → pivoted to `XLA_FLAGS` (see "Why not the
  proto").
- **`XLA_FLAGS` is once-per-process.** A compile-then-set-`XLA_FLAGS`-then-compile
  sequence dumped **0 files**. → `pjrt_dump_hlo()` best-effort sets the flag only
  when it can be the first compile, and errors clearly otherwise.
- **Dumping content is exactly the readable HLO** we want. `before_optimizations`
  shows the two ops; `cpu_after_optimizations` shows them fused into a single
  `kLoop` fusion (`%add_multiply_fusion` calling `%fused_computation`) with
  `is_scheduled=true`.
- CPU emits extra artifacts (LLVM `.ll`, `*.mlir`, `.o`, `.debug_options`,
  buffer-assignment / memory-usage reports); the helper filters to HLO `.txt`.
- `--xla_dump_hlo_pass_re=.*` produced 28 files (one per pass) → drove the
  per-pass parser and its filename format.

## Testing

- Unit (no compile, deterministic): `xla_dump_dir_from_flags()` parsing;
  `parse_hlo_dump()` on fabricated files — stage keys, pass ordering by index,
  backend-prefixed `after_optimizations`, exclusion of `-buffer-assignment`.
- Integration (CPU): `pjrt_dump_hlo()` returns non-empty `before_optimizations`
  (mentions `add`/`multiply`) and `after_optimizations` (mentions `fusion`);
  optimized differs from input; `as.character()` == optimized; `print()` works;
  `passes` argument validated.
- CUDA: same shape assertions, gated with `skip_if(!is_cuda())` — runs later with
  `PJRT_PLATFORM=cuda`; kept lenient (no exact fusion assertion) since GPU output
  differs.
- `tests/testthat/setup-dump.R` sets `XLA_FLAGS=--xla_dump_to=<tmpdir>
  --xla_dump_hlo_as_text` before any test compiles, so the integration/CUDA tests
  read back reliably regardless of test-file ordering.

Build/test note (from CLAUDE.md): never `devtools::load_all()` and
`devtools::test()` in the same R process (protobuf double-registration crash);
use separate `Rscript -e` calls. Verified: full suite 655 pass / 0 fail / 13 skip.

## Documentation

- Roxygen for `pjrt_dump_hlo()` and the `print`/`format`/`as.character` methods,
  including the return value and the `XLA_FLAGS` behaviour.
- `devtools::document()` regenerates `.Rd` + `NAMESPACE`.
- `_pkgdown.yml` has no explicit `reference:` list, so exports are auto-included.

## Out of scope / follow-up

- **xprof profiling** (issue #194, second half). Its own spec/branch. Reference
  implementation: **zml** (`zml/profiling/`, `docs/howtos/profiling.md`) uses the
  **PJRT Profiler extension** → captures an `XSpace` protobuf → `profiling.xplane.pb`
  (xprof/TensorBoard format) + Perfetto JSON, with host-side `TraceMe` scopes.
  For us this means vendoring `pjrt_c_api_profiler_extension.h` plus the
  `XSpace`/`ProfileOptions` protos — sized as its own slice.
- Linking XLA to render optimized HLO in-memory from
  `PJRT_Executable_OptimizedProgram` (rejected: enormous dependency).
- Non-text formats (proto/dot/html): reachable by adding the corresponding
  `--xla_dump_hlo_as_*` flags to `XLA_FLAGS` directly; `pjrt_dump_hlo()` stays
  text-focused.

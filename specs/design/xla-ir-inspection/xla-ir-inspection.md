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

The compiler dumps; we enable that via `XLA_FLAGS` and read the files back. The
compile runs in a **subprocess** to make `XLA_FLAGS` reliable.

### The `XLA_FLAGS` constraint → subprocess

XLA parses `XLA_FLAGS` **once, before the first compilation** in the process
(measured: setting it after a prior compile dumps nothing). So an in-process
approach only works if `pjrt_dump_hlo()` happens to be the session's first
compile — useless in interactive debugging, where the user has already compiled.

The fix: run the dump-compile in a **fresh R process** (via `callr`) with
`XLA_FLAGS` set in its environment from the start. That process's first (and only)
compile always honours the flag, so `pjrt_dump_hlo()` works regardless of session
state — strictly friendlier than gomlx/zml, which just tell users to export
`XLA_FLAGS` before starting.

### In-process fast path (opt-in)

The subprocess costs ~1 s (process startup + plugin load) per call — noticeable
when inspecting many programs. So `pjrt_dump_hlo()` also offers the JAX-style
in-process route for users who *do* set `XLA_FLAGS` up front: if the session's
own `XLA_FLAGS` already enables text dumping (`--xla_dump_to=<dir>` +
`--xla_dump_hlo_as_text`, plus `--xla_dump_hlo_pass_re` when `passes`), it
compiles **in-process** and reads `<dir>` instead of spawning a child.

Correctness under the once-per-process constraint is preserved by *verification,
not trust*: it snapshots `<dir>`, compiles, and diffs (`setdiff`) to grab exactly
this compile's new files. If none appeared — i.e. the flags were set too late to
be parsed — it transparently **falls back to the subprocess**. Measured: ~0.06–0.12 s
per call on the fast path vs. ~1 s via subprocess. This is gated on `session_dump_dir()`
returning non-NULL and on no extra `flags` being requested (see below).

### R — the single entry point

`pjrt_dump_hlo(program, device = NULL, passes = FALSE, flags = character())` → `PJRTHLODump`

- **Fast path** (when `length(flags) == 0` and `session_dump_dir(passes)` is
  non-NULL): compile in-process via the exported `pjrt_compile()`, then parse the
  files that appeared in the session's dump dir. Falls back to the subprocess if
  nothing was dumped.
- **Subprocess path** (default): extracts the program's raw code
  (`impl_program_code`) + format (`impl_program_format`), writes the code to a
  temp file, and spawns a child R process (`callr::r`) whose environment
  reproduces the current one but sets `XLA_FLAGS` and drops `R_TESTS`. The child
  only calls the **exported** `pjrt_program(path=, format=)` + `pjrt_compile()`,
  so it works with any installed `pjrt` (no dev-only symbols needed). The parent
  parses the dumped files into a `PJRTHLODump`; the child's executable is
  discarded — this is inspection only.

The subprocess `XLA_FLAGS` is built as `existing session XLA_FLAGS` + user `flags`
+ our dump flags, in that order, so that: (a) the user's own flags (optimization
levels, etc.) still apply, (b) the user can request extra behaviour via `flags`
(e.g. `--xla_dump_hlo_as_proto`), and (c) our `--xla_dump_to=<tmp>` is **last** and
therefore wins (XLA honours the last occurrence of a flag), so the child always
dumps into the directory we read back. Supplying any `flags` forces the subprocess
path (extra flags cannot be injected into an already-initialised in-process XLA).

Native additions: two tiny accessors, `impl_program_code` (raw `program.code`
bytes) and `impl_program_format`. `parse_hlo_dump()`, `run_dump_subprocess()`,
`dump_flags()`, and `session_dump_dir()` are unexported R helpers. `callr` is
added to Imports.

### Low-level escape hatch

Two, both documented: (1) the `flags` argument for per-call XLA flags, and (2)
`XLA_FLAGS` itself, exactly like gomlx/zml — which additionally *enables the
in-process fast path* above. No wrapper object — the proto route it would have
wrapped is unusable, so wrapping it would be a footgun.

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
- A compile failure in the child surfaces as a `callr` error.
- If the child dumped nothing (missing `after_optimizations`), raise an error that
  the backend may not support HLO dumping (not expected for the CPU/CUDA plugins).

## Spike results (2026-07-02, CPU plugin)

Ran real compiles of a two-op program (`add` then `multiply`) against the
installed `pjrt` 0.4.0.9000 + cached CPU plugin. Findings that shaped the design:

- **Proto route is honoured but unusable.** Setting the dump fields on
  `DebugOptions` via the serialized `CompileOptionsProto` took effect, but the
  all-zero proto3 message clobbered XLA's defaults and the compile errored
  (*parallel_codegen_split_count (0)*). → pivoted to `XLA_FLAGS` (see "Why not the
  proto").
- **`XLA_FLAGS` is once-per-process.** A compile-then-set-`XLA_FLAGS`-then-compile
  sequence dumped **0 files**. → `pjrt_dump_hlo()` runs the compile in a fresh
  subprocess so the flag is always honoured (see Architecture).
- **Dumping content is exactly the readable HLO** we want. `before_optimizations`
  shows the two ops; `cpu_after_optimizations` shows them fused into a single
  `kLoop` fusion (`%add_multiply_fusion` calling `%fused_computation`) with
  `is_scheduled=true`.
- CPU emits extra artifacts (LLVM `.ll`, `*.mlir`, `.o`, `.debug_options`,
  buffer-assignment / memory-usage reports); the helper filters to HLO `.txt`.
- `--xla_dump_hlo_pass_re=.*` produced 28 files (one per pass) → drove the
  per-pass parser and its filename format.

## Testing

No test setup/`XLA_FLAGS` juggling is needed: because each `pjrt_dump_hlo()` call
compiles in its own subprocess, the tests are independent of session/ordering
state — including an explicit test that dumping still works **after a prior
in-session compile** (the exact case the subprocess exists to handle).

- Unit (no compile, deterministic): `parse_hlo_dump()` on fabricated files — stage
  keys, pass ordering by index, backend-prefixed `after_optimizations`, exclusion
  of `-buffer-assignment`; `session_dump_dir()` flag detection (returns the dir
  only when text dumping is enabled, requires `pass_re` for `passes`, NULL when
  `XLA_FLAGS` is unset or lacks `--xla_dump_hlo_as_text`).
- Integration (CPU): `pjrt_dump_hlo()` returns non-empty `before_optimizations`
  (mentions `add`/`multiply`) and `after_optimizations` (mentions `fusion`);
  optimized differs from input; `as.character()` == optimized; `print()` works;
  works after a prior compile; `passes` and `flags` arguments validated; `flags =
  "--xla_dump_hlo_as_proto"` reaches the compiler (a `.pb` appears in the dump
  dir). The in-process fast path and the too-late-flags fallback are verified
  manually — they cannot be forced deterministically inside the shared test
  process, since earlier tests have already initialised in-process XLA (the
  once-per-process constraint); `session_dump_dir()`'s unit test covers the
  decision logic they hinge on.
- CUDA: same shape assertions, gated with `skip_if(!is_cuda())` — runs later with
  `PJRT_PLATFORM=cuda`; kept lenient (no exact fusion assertion) since GPU output
  differs.

Build/test note (from CLAUDE.md): never `devtools::load_all()` and
`devtools::test()` in the same R process (protobuf double-registration crash);
use separate `Rscript -e` calls. Verified: full suite 658 pass / 0 fail / 13 skip.

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

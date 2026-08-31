---
name: upgrade-pjrt
description: >
  Upgrade the vendored XLA headers and proto files and the PJRT plugin artifact
  version to a new XLA commit / ZML release. Use when the user says "upgrade
  PJRT", "update XLA headers", "bump PJRT version", or wants to sync vendored
  files to a newer XLA revision.
user_invocable: true
tools: Read, Edit, Glob, Grep, Bash, Write, AskUserQuestion
---

# Upgrade PJRT

The `pjrt` package vendors headers (`inst/include/`) and proto files
(`inst/proto/`) from [openxla/xla](https://github.com/openxla/xla), and
downloads pre-built PJRT plugin binaries from
[zml/pjrt-artifacts](https://github.com/zml/pjrt-artifacts). An upgrade syncs
the vendored files to a new XLA commit and bumps the artifact version.

Both must move together: the vendored `pjrt_c_api.h` declares the ABI that the
downloaded plugin implements.

That includes **every** plugin, not just ZML's. ZML ships no Windows artifact,
so `plugin_url()` serves Windows from `r-xla/pjrt-builds` at a hand-pinned XLA
commit. A handler reports the FFI header version it was compiled against back to
the runtime (`api.h`, `PopulateMetadata`), and a plugin will not accept a handler
newer than itself -- so as soon as `XLA_FFI_API_MINOR` moves, an un-rebuilt
plugin silently drops every custom call:

```
No FFI handler registered for print_tensor on a platform Host (canonical host)
```

Registration itself still reports success, and `test-custom-call.R` still passes
(it only checks bookkeeping, never compiles a program that uses a custom call),
so this surfaces as `test-ffi.R` and `test-linalg.R` failing wholesale. Rebuild
those plugins as **part of** the upgrade: dispatch `r-xla/pjrt-builds` ->
*Build PJRT* with `commit_hash=<the new XLA commit>`, then repoint the pinned
URLs in `plugin_url()`. To reproduce the mismatch locally without Windows, point
`PJRT_PLUGIN_PATH_CPU` at a plugin from the previous ZML release.

## 1. Pick the target

Ask the user for the ZML release tag if not given. Then read the XLA commit
that release was built from — it is recorded in the release's own tree:

```bash
gh api repos/zml/pjrt-artifacts/contents/openxla/commits.env?ref=<tag> \
  --jq '.content' | base64 -d
```

`XLA_COMMIT` is the one to vendor. The CUDA / cuDNN / NCCL / NVSHMEM versions
and the supported GPU compute capabilities come from the same tree:

```bash
gh api repos/zml/pjrt-artifacts/contents/openxla/bazelrc/upstream/.bazelrc?ref=<tag> \
  --jq '.content' | base64 -d | grep -E 'HERMETIC_|COMPUTE_CAPABILITIES'
```

## 2. Clone XLA at that commit

```bash
mkdir xla && cd xla && git init -q
git remote add origin https://github.com/openxla/xla.git
git fetch --depth 1 origin <XLA_COMMIT> && git checkout -q FETCH_HEAD
```

## 3. Check for files that need to be added to the copy lists

- **Headers** (`HEADER_FILES` in `tools/copy-header.R`): check the `#include
  "xla/..."` directives of the copied headers.
- **Protos** (`PROTO_FILES` in `tools/copy-proto.R`): the list must be closed
  under `import` (ignoring `google/protobuf/*`). Compute the transitive closure
  from the current roots against the new XLA tree and add whatever is missing —
  a new transitive import is the most common reason an upgrade fails to build.

## 4. Copy and patch

```bash
XLA_SRC=<path-to-xla> Rscript tools/copy-header.R
XLA_SRC=<path-to-xla> Rscript tools/copy-proto.R
```

Each script copies the files verbatim and then applies the patches in
`tools/patch/` that belong to its destination tree (`inst-include-*` /
`inst-proto-*`). See `tools/patch.R` for the naming convention.

### What the patches fix

- **`inst-include-xla-pjrt-c-pjrt_c_api.h.patch`** — makes
  `_PJRT_API_STRUCT_FIELD` append `_` to the struct field name
  (`fn_type* fn_type##_`), so the field does not collide with the typedef of
  the same name in C. `PJRT_Api_STRUCT_SIZE` is updated to name the renamed
  last field.
- **`inst-include-xla-ffi-api-c_api.h.patch`** — the same collision, from the
  other side: every function typedef listed in `_XLA_FFI_API_STRUCT_FIELD`
  gets a `_` suffix and the macro becomes `fn_type##_* fn_type`.
- **`inst-include-xla-ffi-api-api.h.patch`** — adds `#include <stdexcept>` and
  a trailing `throw` to the `operator<<` switches that have no default case
  (fixes `-Wreturn-type`), and uses `&&` instead of `&` in a fold expression
  (fixes `-Wbitwise-instead-of-logical`).
- **`inst-include-xla-ffi-api-ffi.h.patch`** — the same trailing `throw` for
  `ByteWidth()` and the `XLA_FFI_ArgType` / `XLA_FFI_RetType` `operator<<`.
- **`inst-proto-xla-backends-autotuner-backends.proto.patch`** — rewrites
  `edition = "2023"` to `syntax = "proto3"` and quotes the `reserved` enum
  names, so the file compiles with protoc 3.21 (`protobuf@21`), the version
  available on the CI runners and assumed by CRAN.

### When a patch fails to apply

The upstream file changed in a patched region. Re-apply the same *logical*
change to the freshly copied file by hand, then regenerate the patch:

```bash
XLA_SRC=<path-to-xla> Rscript tools/regen-patch.R inst/include/xla/ffi/api/api.h
```

`tools/regen-patch.R` diffs the pristine XLA file against our copy and rewrites
the patch; if there is no difference it deletes the patch file. Afterwards,
re-run the copy scripts from a clean tree and confirm they reproduce byte-identical
files — that round-trip is the check that the patches are in sync.

## 5. Update the plugin version and CUDA dependencies

- `R/plugin.R`: `plugin_version()` returns the ZML tag without the leading `v`.
- `R/plugin.R`: `the[["config"]]$cuda_r_package` / `$cuda_r_repos` must name a
  CUDA R package matching the artifact's `HERMETIC_CUDA_VERSION`. This is not
  optional slack: the plugin has a versioned `NEEDED` entry on
  `libnvrtc-builtins.so.<major>.<minor>`, so a CUDA package one minor version
  behind will not load. Check with:

  ```bash
  readelf -d ~/.cache/R/pjrt/cuda/libpjrt_cuda.so | grep NEEDED
  ```

- `.github/workflows/R-CMD-check.yaml` and `.github/workflows/test-cuda.yaml`:
  the CUDA container image tag and the `cudaX.Y` R package reference.
- The CUDA R package must also ship `nvvm/libdevice/libdevice.10.bc`. On CUDA 12
  that came with the nvcc wheel; on CUDA 13 it moved to a separate
  `nvidia-nvvm` wheel. Without it pjrt's own tests still pass but anvl's math
  ops fail with `libdevice not found at ./libdevice.10.bc`.
- `src/ffi_cuda.cpp`: pjrt declares the cuSOLVER entry points itself rather
  than including the CUDA SDK, and `dlopen`s the library by SONAME. cuSOLVER's
  SONAME major does *not* track the CUDA major (`.11` on CUDA 12, `.12` on
  CUDA 13), so a CUDA major bump needs a new candidate in `kCusolverSonames`.
  Verify the declared signatures in `src/ffi_cuda.h` against the new
  `cusolverDn.h` — they are hand-written and nothing checks them at build time.

## 6. Build and test

```bash
R CMD INSTALL --no-multiarch --preclean .
cd tests && NOT_CRAN=true PJRT_TEST=1 PJRT_PLATFORM=cpu PJRT_INSTALL=1 Rscript testthat.R
```

Compare the compiler warnings from `inst/include/` against a pre-upgrade build;
new `-Wreturn-type` warnings mean a new switch statement needs a `throw` added
to the patch.

gcc is not enough on its own here: `PJRT_NO_DISCARD` expands to `[[nodiscard]]`
only under clang, so new `[[nodiscard]]` markings on `PJRT_Api` fields show up
first on the macOS runner, where `rcmdcheck`'s `error_on = "warning"` turns them
into a check failure. Any `PJRT_Api` call whose result is deliberately dropped
(destructors) should pass it to `destroy_error()` instead.

With a GPU available, repeat with `PJRT_PLATFORM=cuda`. Also run the {anvl}
test suite against the upgraded pjrt — anvl exercises code paths (e.g.
`libdevice`-backed math) that pjrt's own tests do not.

**Important**: do not call `devtools::load_all()` and `devtools::test()` in the
same R process (protobuf descriptor crash). Use separate `Rscript -e` calls.

## 7. Finish

Update `NEWS.md` and re-render `README.md` from `README.Rmd`. Windows CI is
expected to fail and can be ignored.

---
name: upgrade-pjrt
description: >
  Upgrade PJRT headers, proto files, and plugin artifact version to a new XLA
  commit. Use when the user says "upgrade PJRT", "update XLA headers",
  "bump PJRT version", or wants to sync vendored files to a newer XLA revision.
user_invocable: true
tools: Read, Edit, Glob, Grep, Bash, Write, Agent, AskUserQuestion
---

# Upgrade PJRT

The `pjrt` package vendors headers and proto files from the
[openxla/xla](https://github.com/openxla/xla) repository and downloads
pre-built PJRT plugin binaries from
[zml/pjrt-artifacts](https://github.com/zml/pjrt-artifacts). Upgrading
involves syncing these files to a new XLA commit and updating the artifact
version.

## Steps

### 1. Clone the XLA source at the target commit

Ask the user for the target commit hash and the ZML artifact version if not
provided. Remind them that the XLA commit hash for a given ZML artifact version
can be found in the ZML CI workflow file (it is an input to the build workflow).

```bash
git clone --depth 1 https://github.com/openxla/xla.git <path>
git -C <path> fetch --depth 1 origin <commit>
git -C <path> checkout <commit>
```

### 2. Check for new files to copy

Before copying, inspect the XLA source to see if additional header or proto
files need to be added to the copy lists:

- **Headers**: Check `#include` directives in the copied headers for any new
  `xla/` includes not already listed in `tools/copy-header.R`.
- **Protos**: Check `import` directives (excluding `google/protobuf/`) in the
  copied protos for any new `xla/` imports not already listed in
  `tools/copy-proto.R`. Follow transitive imports.

If new files are needed, add them to the respective `HEADER_FILES` or
`PROTO_FILES` vectors in the copy scripts.

### 3. Copy headers and protos

Set `XLA_SRC` to the cloned XLA directory and run the copy scripts from the
package root:

```bash
XLA_SRC=<path> Rscript tools/copy-header.R
XLA_SRC=<path> Rscript tools/copy-proto.R
```

These scripts just copy files — they do not apply patches.

### 4. Apply patches

The `tools/patch/` directory contains unified diffs between the original XLA
files and our modified copies. Each patch file is named after the file it
modifies (with `/` replaced by `-`).

Apply all patches from the package root:

```bash
git apply tools/patch/*.patch
```

#### What the patches fix

- **`xla-pjrt-c-pjrt_c_api.h.patch`**: Changes the `_PJRT_API_STRUCT_FIELD`
  macro to append `_` to field names (`fn_type##_`), avoiding C name collisions
  between typedef names and struct field names. Updates `PJRT_Api_STRUCT_SIZE`
  accordingly.
- **`xla-ffi-api-c_api.h.patch`**: Same pattern for the FFI API — renames
  typedef function types with `_` suffix and adjusts the
  `_XLA_FFI_API_STRUCT_FIELD` macro.
- **`xla-ffi-api-api.h.patch`**: Adds `#include <stdexcept>`, adds
  `throw std::runtime_error(...)` after switch statements missing default cases
  (fixes `-Wreturn-type`), and changes `&` to `&&` in a fold expression (fixes
  `-Wbitwise-instead-of-logical`).
- **`xla-ffi-api-ffi.h.patch`**: Adds `throw std::runtime_error(...)` after a
  switch in `ByteWidth()`.
- **`xla-backends-autotuner-backends.proto.patch`**: Converts
  `edition = "2023"` to `syntax = "proto3"` so the file compiles with protoc
  3.21 (protobuf@21), which is widely available on Ubuntu and macOS.

#### When patches fail to apply

If the upstream code changed in the patched regions, the patches will fail. In
that case:

1. Inspect the rejected hunks to understand the conflict.
2. Apply the same logical change manually to the new file.
3. Regenerate the patch (see step 8).

### 5. Update CUDA dependency versions

Check the ZML build workflow for the CUDA toolkit, cuDNN, and nvshmem versions
used by the new PJRT artifact. Update the following files to match:

- **`.github/workflows/R-CMD-check.yaml`**: CUDA container image tags (both the
  `-runtime-` and `-cudnn-devel-` variants), nvshmem repo/package versions in
  the "Install CUDA libraries manually" step, and the `cuda12.X` R package
  reference in `extra-packages`.
- **`R/plugin.R`**: `cuda_r_package` in the `the[["config"]]` list (e.g.
  `"cuda12.8"` may become `"cuda12.9"`).
- **`.github/workflows/test-cuda.yaml`**: `cudnn` and `cuda-nvrtc` versions in
  the conda environment.

The R-CMD-check workflow has a `manual-cuda` input (triggerable via
`workflow_dispatch`) that installs CUDA libraries directly from NVIDIA instead
of using the cuda R package. This is intended for testing during PJRT upgrades
before the cuda R package has been updated to match.

### 6. Update the PJRT artifacts version

Edit `R/plugin.R` and change the version string returned by
`plugin_version()`.

### 7. Verify the build

```bash
Rscript -e 'devtools::install()'
```

**Important**: Do not call `devtools::load_all()` and `devtools::test()` in the
same R process (protobuf descriptor crash). Use separate `Rscript -e` calls.

### 8. Regenerate patches (if any file was modified)

If you had to adjust patches or make additional edits to the copied files,
regenerate **all** patches to keep them in sync:

```bash
for file in <list of modified files>; do
  diff -u <xla-source>/$file <our-copy>/$file \
    | sed "1s|.*|--- a/$file|;2s|.*|+++ b/$file|" \
    > tools/patch/$(echo $file | tr '/' '-').patch
done
```

Only files with actual diffs should have patch files.

### 9. Create PR and monitor CI

Use the `/pr-create` skill to create a pull request. Wait for CI to pass and
debug any failures. Windows CI is expected to fail and can be ignored.

If the cuda R package has not been updated yet for the new CUDA version,
trigger the workflow manually with `manual-cuda: true` via `workflow_dispatch`
to test with directly installed NVIDIA libraries.

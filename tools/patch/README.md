# Local modifications to vendored XLA files

`inst/include/` and `inst/proto/` are copied verbatim from
[openxla/xla](https://github.com/openxla/xla) by `tools/copy-header.R` and
`tools/copy-proto.R`. Everything we change afterwards lives here as a patch, so
an upgrade is a re-copy plus a re-apply rather than a merge.

A patch is named after the file it modifies, relative to the package root, with
`/` replaced by `-`:

| file | patch |
|---|---|
| `inst/include/xla/ffi/api/api.h` | `inst-include-xla-ffi-api-api.h.patch` |
| `inst/proto/xla/backends/autotuner/backends.proto` | `inst-proto-xla-backends-autotuner-backends.proto.patch` |

The `inst-include-` / `inst-proto-` prefix is what lets each copy script apply
only the patches for the tree it just refreshed (see `apply_patches()` in
`tools/patch.R`). A vendored file with no local modifications has no patch file.

To change a vendored file, edit `inst/include/...` or `inst/proto/...` directly
and then regenerate its patch:

```bash
XLA_SRC=<path-to-openxla/xla> Rscript tools/regen-patch.R inst/include/xla/ffi/api/api.h
```

`XLA_SRC` must point at the XLA checkout the files were copied from, at the
commit recorded for the current `plugin_version()`. See the `upgrade-pjrt` skill
(`.claude/skills/upgrade-pjrt.md`) for what each patch fixes and for the full
upgrade procedure.

# Shared helpers for the vendoring scripts (tools/copy-header.R, tools/copy-proto.R).
#
# Vendored XLA files are copied verbatim; every local modification is a patch in
# tools/patch/. A patch is named after the file it modifies, relative to the
# package root, with "/" replaced by "-" -- e.g. inst/include/xla/ffi/api/api.h
# is patched by tools/patch/inst-include-xla-ffi-api-api.h.patch. That naming
# lets each copy script pick out only the patches for the tree it just refreshed.

PATCH_DIR <- "tools/patch"

patch_name <- function(path) {
  fs::path(PATCH_DIR, paste0(gsub("/", "-", path, fixed = TRUE), ".patch"))
}

copy_files <- function(src_root, dest_root, files) {
  for (file in files) {
    from <- fs::path(src_root, file)
    if (!fs::file_exists(from)) {
      stop("File does not exist in the XLA source tree: ", from)
    }
    dest <- fs::path(dest_root, file)
    fs::dir_create(fs::path_dir(dest), recurse = TRUE)
    fs::file_copy(from, dest, overwrite = TRUE)
    cat("Copied", file, "\n")
  }
}

# Applies every patch in tools/patch/ that belongs to `dest_root`.
apply_patches <- function(dest_root) {
  prefix <- paste0(gsub("/", "-", dest_root, fixed = TRUE), "-")
  patches <- fs::dir_ls(PATCH_DIR, glob = paste0("*/", prefix, "*.patch"))
  for (patch in patches) {
    cat("Applying patch", as.character(patch), "\n")
    status <- system2("git", c("apply", shQuote(patch)))
    if (status != 0) {
      stop(
        "Failed to apply ",
        patch,
        ".\n",
        "The upstream file likely changed in a patched region. Fix the copy by ",
        "hand, then regenerate the patch with tools/regen-patch.R."
      )
    }
  }
}

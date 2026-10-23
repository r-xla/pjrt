# Regenerates a patch in tools/patch/ from the current state of a vendored file.
#
# Usage: XLA_SRC=<path-to-openxla/xla> Rscript tools/regen-patch.R <dest-path>...
#
# <dest-path> is the package-relative path of the vendored file, e.g.
#   Rscript tools/regen-patch.R inst/include/xla/ffi/api/api.h
#
# The patch is the diff between the pristine XLA file and our copy. If the two
# are identical the patch file is removed, since there is nothing to patch.

XLA_SRC <- Sys.getenv("XLA_SRC", "../../openxla/xla")
if (!dir.exists(XLA_SRC)) {
  stop("XLA source directory does not exist: ", XLA_SRC)
}

source("tools/patch.R")

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop("Usage: Rscript tools/regen-patch.R <dest-path>...")
}

for (dest in args) {
  # strip the inst/include or inst/proto prefix to get the path within the XLA tree
  rel <- sub("^inst/(include|proto)/", "", dest)
  from <- fs::path(XLA_SRC, rel)
  if (!fs::file_exists(from)) {
    stop("File does not exist in the XLA source tree: ", from)
  }

  patch <- patch_name(dest)
  # diff exits 1 when the files differ, which is the normal case here.
  diff <- suppressWarnings(system2(
    "diff",
    c("-u", shQuote(from), shQuote(dest)),
    stdout = TRUE
  ))

  if (!length(diff)) {
    if (fs::file_exists(patch)) {
      fs::file_delete(patch)
      cat("Removed", as.character(patch), "(no local modifications)\n")
    }
    next
  }

  # Rewrite the ---/+++ header lines to package-relative paths so the patch
  # applies with `git apply` from the package root.
  diff[1] <- paste0("--- a/", dest)
  diff[2] <- paste0("+++ b/", dest)
  fs::dir_create(PATCH_DIR, recurse = TRUE)
  writeLines(diff, patch)
  cat("Wrote", as.character(patch), "\n")
}

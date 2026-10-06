# Copies PJRT API headers from the XLA source directory to the R package.
# Usage: XLA_SRC=<path-to-openxla/xla> Rscript tools/copy-header.R
#
# The copy is verbatim; local modifications live in tools/patch/ and are applied
# afterwards. See tools/patch/README.md.

XLA_SRC <- Sys.getenv("XLA_SRC", "../../openxla/xla")
if (!dir.exists(XLA_SRC)) {
  stop("XLA source directory does not exist: ", XLA_SRC)
}

DEST_ROOT <- "inst/include"

HEADER_FILES <- c(
  "xla/pjrt/c/pjrt_c_api.h",
  "xla/pjrt/c/pjrt_c_api_ffi_extension.h",
  "xla/ffi/api/c_api.h",
  "xla/ffi/api/api.h",
  "xla/ffi/api/ffi.h"
)

source("tools/patch.R")

copy_files(XLA_SRC, DEST_ROOT, HEADER_FILES)
apply_patches(DEST_ROOT)

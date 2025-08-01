# Copies PJRT API headers from the XLA source directory to the R package.
# Usage: Rscript tools/copy-header.R

XLA_SRC <- Sys.getenv("XLA_SRC", "../xla")
if (!dir.exists(XLA_SRC)) {
  stop("XLA source directory does not exist: ", XLA_SRC)
}

if (!dir.exists("inst/include")) {
  dir.create("inst/include", recursive = TRUE)
}

HEADER_FILES <- c(
  "xla/pjrt/c/pjrt_c_api.h",
  "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
)

for (file in HEADER_FILES) {
  from <- fs::path(XLA_SRC, file)
  dest <- fs::path("inst/include", file)

  if (!fs::dir_exists(fs::path_dir(dest))) {
    fs::dir_create(fs::path_dir(dest), recurse = TRUE)
  }

  fs::file_copy(from, dest, overwrite = TRUE)

  if (basename(file) == "pjrt_c_api.h") {
    content <- readLines(dest)
    pattern <- "^#define _PJRT_API_STRUCT_FIELD\\(fn_type\\) fn_type\\* fn_type$"
    replacement <- "\n// This is needed to be able to compile on CRAN\n#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type##_"
    content <- gsub(pattern, replacement, content)
    writeLines(content, dest)
    cat("Applied macro definition edit to:", dest, "\n")
  }
}

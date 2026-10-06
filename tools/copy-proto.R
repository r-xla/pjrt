# Copies the protobuf files required for the PJRT interface from the XLA source
# directory to the R package.
# Usage: XLA_SRC=<path-to-openxla/xla> Rscript tools/copy-proto.R
#
# PROTO_FILES must be closed under `import` (excluding google/protobuf/*); after
# an upgrade, re-check the import graph for files that need to be added here.
#
# The copy is verbatim; local modifications live in tools/patch/ and are applied
# afterwards. See tools/patch/README.md.

XLA_SRC <- Sys.getenv("XLA_SRC", "../../openxla/xla")
if (!dir.exists(XLA_SRC)) {
  stop("XLA source directory does not exist: ", XLA_SRC)
}

DEST_ROOT <- "inst/proto"

PROTO_FILES <- c(
  "xla/pjrt/proto/compile_options.proto",
  "xla/stream_executor/device_description.proto",
  "xla/xla.proto",
  "xla/xla_data.proto",
  "xla/autotune_results.proto",
  "xla/stream_executor/cuda/cuda_compute_capability.proto",
  "xla/stream_executor/sycl/oneapi_compute_capability.proto",
  "xla/autotuning.proto",
  "xla/tsl/protobuf/dnn.proto",
  "xla/service/hlo.proto",
  "xla/service/metrics.proto",
  "xla/backends/autotuner/backends.proto"
)

source("tools/patch.R")

copy_files(XLA_SRC, DEST_ROOT, PROTO_FILES)
apply_patches(DEST_ROOT)

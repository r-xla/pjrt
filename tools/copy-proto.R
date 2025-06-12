# Generates the Protobuf Files required for the PJRT interface.
# Usage: Rscript tools/copy-proto.R
# Expects a xla source directory on the same root directory of this folder
# Or the XLA_SRC environment variable to be set

XLA_SRC <- Sys.getenv("XLA_SRC", "../xla")

if (!dir.exists(XLA_SRC)) {
  stop("XLA source directory does not exist: ", XLA_SRC)
}

if (!dir.exists("inst/proto")) {
  dir.create("inst/proto", recursive = TRUE)
}

PROTO_FILES <- c(
  "xla/pjrt/proto/compile_options.proto",
  "xla/stream_executor/device_description.proto",
  "xla/xla.proto",
  "xla/xla_data.proto",
  "xla/autotune_results.proto",
  "xla/stream_executor/cuda/cuda_compute_capability.proto",
  "xla/autotuning.proto",
  "xla/tsl/protobuf/dnn.proto",
  "xla/service/hlo.proto",
  "xla/service/metrics.proto"
)

for (file in PROTO_FILES) {
  from <- fs::path(XLA_SRC, file)
  dest <- fs::path("inst/proto", file)

  if (!fs::dir_exists(fs::path_dir(dest))) {
    fs::dir_create(fs::path_dir(dest))
  }

  fs::file_copy(from, dest, overwrite = TRUE)
}

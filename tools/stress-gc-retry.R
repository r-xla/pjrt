# Stress test for the gc-on-OOM retry path in pjrt.
#
# Usage:
#   Rscript tools/stress-gc-retry.R [platform] [chunk_mb] [n_chunks]
#
# Defaults to cuda. On CUDA, the script repeatedly allocates large buffers
# without keeping R references to them, so PJRTBuffer external pointers
# accumulate as unreachable R objects. When the PJRT plugin reports
# RESOURCE_EXHAUSTED, the new try_alloc wrapper should call R's gc() (via
# rpjrt::call_r_gc), free the unreferenced buffers, and retry the
# allocation. Success criterion: at least one such retry actually fires,
# observed via impl_gc_call_count().

args <- commandArgs(trailingOnly = TRUE)
platform <- if (length(args) >= 1L) args[[1L]] else "cuda"
chunk_mb <- if (length(args) >= 2L) as.numeric(args[[2L]]) else 256
n_chunks <- if (length(args) >= 3L) as.integer(args[[3L]]) else 200L

library(pjrt)

# Internal counter exposing how many times the gc-on-OOM retry fired. Pulled
# from the namespace here (rather than `pjrt:::`) since this is a standalone
# script and the counter is not exported.
impl_gc_call_count <- getFromNamespace("impl_gc_call_count", "pjrt")

cat(sprintf("Platform:        %s\n", platform))
cat(sprintf("Chunk size:      %.0f MB (f32)\n", chunk_mb))
cat(sprintf("Max iterations:  %d\n", n_chunks))

# Force the chosen plugin to load and pick a device.
client <- pjrt_client(platform)
device <- devices(client)[[1L]]
cat(sprintf("Device:          %s\n", format(device)))

# Number of f32 elements per chunk.
n_elt <- as.integer((chunk_mb * 1024 * 1024) / 4)

baseline <- impl_gc_call_count()
cat(sprintf("\nStarting gc_call_count: %d\n\n", baseline))

# Disable R's automatic GC so unreachable buffers accumulate. This is the
# whole point: without retry-on-OOM we would crash here. With the new
# wrapper, the plugin's RESOURCE_EXHAUSTED should be caught and gc() called
# on demand.
prev_gctorture <- gctorture(FALSE)
gcinfo(FALSE)
on.exit(gctorture(prev_gctorture), add = TRUE)

ok <- TRUE
for (i in seq_len(n_chunks)) {
  # Allocate, immediately discard. Reference count drops to 0; the XPtr is
  # eligible for finalization but only finalized at the next gc().
  result <- tryCatch(
    {
      pjrt_buffer(numeric(n_elt), dtype = "f32", device = device)
      NULL
    },
    error = function(e) e
  )
  if (inherits(result, "error")) {
    cat(sprintf("\nFAILED at iter %d: %s\n", i, conditionMessage(result)))
    ok <- FALSE
    break
  }
  if (i %% 10L == 0L) {
    cat(sprintf(
      "  iter %3d  gc_call_count=%d\n",
      i,
      impl_gc_call_count() - baseline
    ))
  }
}

retries <- impl_gc_call_count() - baseline
cat(sprintf("\nTotal gc-on-OOM retries triggered: %d\n", retries))

if (!ok && retries == 0L) {
  stop("Hit OOM but no gc retry fired -- try_alloc wiring is broken")
}
if (retries == 0L) {
  cat("No retries observed. The plugin never reported RESOURCE_EXHAUSTED.\n")
  cat("On CPU this is expected (host memory is huge). Try platform=cuda\n")
  cat("or raise chunk_mb / n_chunks to push past the device memory limit.\n")
} else {
  cat("Retry path exercised successfully.\n")
}

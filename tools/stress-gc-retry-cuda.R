# Stress test for the gc-on-OOM retry path in pjrt (CUDA device memory).
#
# This is the CUDA counterpart to tools/stress-cpu-memory.R: here R's GC is
# blind to VRAM pressure, so unreferenced buffers accumulate until PJRT reports
# RESOURCE_EXHAUSTED and the try_alloc wrapper forces a gc() and retries. On
# CPU the bytes live in an R RAWSXP that ordinary GC can see, so that script
# verifies reclamation happens without any retry instead.
#
# Usage:
#   Rscript tools/stress-gc-retry-cuda.R [platform] [chunk_mb] [n_chunks]
#
# Defaults to cuda. On CUDA, the script repeatedly executes a no-argument
# program that materializes a large device buffer from a constant (`iota`),
# without keeping R references to the outputs, so PJRTBuffer external pointers
# accumulate as unreachable R objects. Crucially, no large *host* buffer is
# ever allocated, so R's GC stays blind to the device-memory pressure (unlike
# pjrt_buffer(numeric(n), ...), whose RAWSXP would itself nudge R's GC). When
# the PJRT plugin reports
# RESOURCE_EXHAUSTED, the new try_alloc wrapper should call R's gc() (via
# rpjrt::call_r_gc), free the unreferenced buffers, and retry the
# allocation. Success criterion: at least one such retry actually fires,
# observed via impl_gc_call_count().

args <- commandArgs(trailingOnly = TRUE)
platform <- if (length(args) >= 1L) args[[1L]] else "cuda"
chunk_mb <- if (length(args) >= 2L) as.numeric(args[[2L]]) else 2560
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

# Compile a no-argument program that materializes a large device buffer
# straight from a constant (an `iota` of n_elt f32 elements). This is the
# crucial difference from tools/stress-cpu-memory.R: we never allocate a large
# host buffer, so R's GC stays blind to the pressure. The device-resident
# output is what accumulates as unreachable PJRTBuffer external pointers, so
# the only thing that can reclaim VRAM is the gc-on-OOM retry path itself.
src <- sprintf(
  "func.func @main() -> tensor<%dxf32> {
  %%0 = stablehlo.iota dim = 0 : tensor<%dxf32>
  func.return %%0 : tensor<%dxf32>
}",
  n_elt,
  n_elt,
  n_elt
)
exec <- pjrt_compile(pjrt_program(src), device = device)

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
  # Execute, immediately discard the output. Reference count drops to 0; the
  # XPtr is eligible for finalization but only finalized at the next gc().
  result <- tryCatch(
    {
      pjrt_execute(exec)
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

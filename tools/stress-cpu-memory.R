# Stress test for CPU host-memory management in pjrt.
#
# Usage:
#   Rscript tools/stress-cpu-memory.R [platform] [chunk_mb] [n_chunks]
#
# Defaults to cpu. This is the CPU counterpart to tools/stress-gc-retry-cuda.R.
# On CPU a buffer's bytes live in an R RAWSXP stashed in the XPtr's protected
# slot, so the memory is counted in R's Vcells heap and ordinary R garbage
# collection both *sees* the pressure and *reclaims* the bytes when the buffer
# becomes unreachable. There is no gc-on-OOM retry to exercise here (host RAM
# rarely exhausts and R already triggers collection under Vcells pressure);
# instead we verify the two halves of the keepalive invariant:
#
#   1. A live buffer pins its bytes -- object.size() and Vcells both reflect the
#      real data size, proving the RAWSXP is reachable through the XPtr.
#   2. A discarded buffer's bytes are reclaimed -- allocating far more total
#      bytes than fit in a bounded working set does not grow resident memory,
#      proving GC frees the RAWSXP once the buffer is unreachable.
#
# A leak in the keepalive (RAWSXP retained after the buffer dies) or a failure
# to back the buffer with a RAWSXP at all would show up as unbounded Vcells
# growth across the discard loop.

args <- commandArgs(trailingOnly = TRUE)
platform <- if (length(args) >= 1L) args[[1L]] else "cpu"
chunk_mb <- if (length(args) >= 2L) as.numeric(args[[2L]]) else 256
n_chunks <- if (length(args) >= 3L) as.integer(args[[3L]]) else 200L

library(pjrt)

# Read the RAWSXP pinned in a buffer XPtr's protected slot. Not exported, so
# pulled from the namespace (rather than `pjrt:::`) since this is a standalone
# script. Used below to watch the keepalive migrate across a donating execute.
impl_test_xptr_prot <- getFromNamespace("impl_test_xptr_prot", "pjrt")

# Resident size of R's vector heap in MB, after collection so the figure
# reflects only memory still reachable (i.e. not yet-collectable garbage).
# Collect twice: the first gc finalizes an unreachable buffer XPtr (running
# ~PJRTBuffer and releasing PJRT's alias), which only then makes its prot-slot
# RAWSXP collectable -- the second gc actually reclaims those bytes.
vcells_mb <- function() {
  gc(full = TRUE, verbose = FALSE)
  g <- gc(full = TRUE, verbose = FALSE)
  g["Vcells", "(Mb)"]
}

chunk_bytes <- chunk_mb * 1024 * 1024
total_mb <- chunk_mb * n_chunks

cat(sprintf("Platform:        %s\n", platform))
cat(sprintf("Chunk size:      %.0f MB (f32)\n", chunk_mb))
cat(sprintf("Iterations:      %d\n", n_chunks))
cat(sprintf("Total allocated: %.0f MB (over the run)\n", total_mb))

client <- pjrt_client(platform)
device <- devices(client)[[1L]]
cat(sprintf("Device:          %s\n", format(device)))

if (!grepl("cpu", tolower(format(device)))) {
  stop("stress-cpu-memory.R is a CPU test; got a non-CPU device. Pass cpu.")
}

# Number of f32 elements per chunk.
n_elt <- as.integer(chunk_bytes / 4)

# ---------------------------------------------------------------------------
# Invariant 1: a live buffer pins its bytes on R's heap.
# ---------------------------------------------------------------------------
baseline <- vcells_mb()
held <- pjrt_buffer(numeric(n_elt), dtype = "f32", device = device)

osize_mb <- as.numeric(object.size(held)) / (1024 * 1024)
held_mb <- vcells_mb() - baseline
cat(sprintf("\nLive buffer object.size():   %.0f MB (data is %.0f MB)\n", osize_mb, chunk_mb))
cat(sprintf("Vcells growth while held:    %.0f MB\n", held_mb))

# object.size() traverses the prot-slot RAWSXP, so it must report ~the data
# size, not the few bytes of a bare external pointer.
if (osize_mb < 0.5 * chunk_mb) {
  stop(sprintf(
    "object.size() = %.0f MB but data is %.0f MB -- RAWSXP keepalive missing",
    osize_mb,
    chunk_mb
  ))
}
if (held_mb < 0.5 * chunk_mb) {
  stop(sprintf(
    "holding a buffer only grew Vcells by %.0f MB (expected ~%.0f MB)",
    held_mb,
    chunk_mb
  ))
}

# Drop the reference; the bytes must come back after collection.
rm(held)
reclaimed_mb <- vcells_mb() - baseline
cat(sprintf("Vcells after dropping it:    %+.0f MB (should be ~0)\n", reclaimed_mb))
if (reclaimed_mb > 0.5 * chunk_mb) {
  stop(sprintf(
    "dropping the buffer left %.0f MB unreclaimed -- keepalive leaks",
    reclaimed_mb
  ))
}

# ---------------------------------------------------------------------------
# Invariant 2: discarded buffers are reclaimed, so a long run stays bounded.
# ---------------------------------------------------------------------------
cat("\nAllocating and discarding buffers:\n")
peak_mb <- 0
ok <- TRUE
for (i in seq_len(n_chunks)) {
  # Allocate, immediately discard. Reference count drops to 0; R's GC reclaims
  # the backing RAWSXP under Vcells pressure.
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
    resident <- vcells_mb() - baseline
    peak_mb <- max(peak_mb, resident)
    cat(sprintf("  iter %3d  resident=%+.0f MB\n", i, resident))
  }
}

if (!ok) {
  stop("CPU allocation loop failed before completing")
}

final_mb <- vcells_mb() - baseline
cat(sprintf("\nTotal bytes cycled:   %.0f MB\n", total_mb))
cat(sprintf("Peak resident growth: %.0f MB\n", peak_mb))
cat(sprintf("Final resident growth: %+.0f MB\n", final_mb))

# The working set is a single buffer at a time, so resident memory should stay
# within a small multiple of one chunk even though we cycled `total_mb` of
# allocations through it. A leak would scale with `total_mb` instead.
budget_mb <- 8 * chunk_mb
if (peak_mb > budget_mb) {
  stop(sprintf(
    "peak resident %.0f MB exceeds %.0f MB budget after cycling %.0f MB -- leak",
    peak_mb,
    budget_mb,
    total_mb
  ))
}

# ---------------------------------------------------------------------------
# Invariant 3: a non-donating pjrt_execute() reclaims its inputs once the loop
# drops them. Each iteration uploads a fresh input (RAWSXP-backed), runs a
# non-aliasing program, and discards both the input and the output. The input's
# keepalive is on R's heap, so Vcells sees it; if pjrt_execute retained inputs
# across iterations (e.g. the deferred-release queue never draining), resident
# memory would grow with the iteration count instead of staying bounded.
#
# The output is fresh PJRT-owned device memory with no RAWSXP keepalive, so its
# bytes are not on R's heap -- dropping it lets the XPtr's finalizer free them.
# (The donating case, where the output *inherits* the input's RAWSXP, is
# Invariant 4 below.)
# ---------------------------------------------------------------------------
cat("\nExecuting and discarding (non-donating):\n")

# Elementwise multiply with no aliasing: the output is fresh device memory and
# the input stays valid until we drop it.
exec_mlir <- sprintf(
  '
module {
  func.func @main(%%arg0: tensor<%dxf32>) -> tensor<%dxf32> {
    %%two = stablehlo.constant dense<2.0> : tensor<%dxf32>
    %%out = stablehlo.multiply %%arg0, %%two : tensor<%dxf32>
    return %%out : tensor<%dxf32>
  }
}
',
  n_elt,
  n_elt,
  n_elt,
  n_elt,
  n_elt
)
exec <- pjrt_compile(
  pjrt_program(src = exec_mlir, format = "mlir"),
  device = device
)

exec_baseline <- vcells_mb()
exec_peak_mb <- 0
for (i in seq_len(n_chunks)) {
  res <- tryCatch(
    {
      xi <- pjrt_buffer(numeric(n_elt), dtype = "f32", device = device)
      oi <- pjrt_execute(exec, xi)
      rm(xi, oi)
      NULL
    },
    error = function(e) e
  )
  if (inherits(res, "error")) {
    cat(sprintf("\nFAILED at exec iter %d: %s\n", i, conditionMessage(res)))
    stop("execute stress loop failed")
  }
  if (i %% 10L == 0L) {
    resident <- vcells_mb() - exec_baseline
    exec_peak_mb <- max(exec_peak_mb, resident)
    cat(sprintf("  iter %3d  resident=%+.0f MB\n", i, resident))
  }
}
cat(sprintf("Peak resident growth (execute): %.0f MB\n", exec_peak_mb))
if (exec_peak_mb > budget_mb) {
  stop(sprintf(
    "execute peak resident %.0f MB exceeds %.0f MB budget after cycling %.0f MB -- input keepalive leaks",
    exec_peak_mb,
    budget_mb,
    total_mb
  ))
}

# ---------------------------------------------------------------------------
# Invariant 4: a donating pjrt_execute() migrates the backing RAWSXP from the
# donated input's XPtr to the aliased output's XPtr (raw-pointer reassignment),
# and the migrated bytes are still reclaimed once the output dies.
#
# On CPU the input's storage *is* a RAWSXP, and donation aliases it as the
# output's storage. So pjrt_execute must move that RAWSXP into the output's prot
# slot and clear the input's -- otherwise the output would read freed memory
# (keepalive lost) or the bytes would be double-freed. A bug in the reassignment
# would surface here as wrong readback, a crash under GC, or unbounded growth.
# ---------------------------------------------------------------------------
cat("\nDonating pjrt_execute() keepalive migration:\n")

# Multiply-in-place: arg0 is donated to output 0 (tf.aliasing_output), so PJRT
# reuses the input's storage for the output.
donate_mlir <- sprintf(
  '
module @double_inplace {
  func.func @main(%%arg0: tensor<%dxf32> {tf.aliasing_output = 0 : i32}) -> tensor<%dxf32> {
    %%two = stablehlo.constant dense<2.0> : tensor<%dxf32>
    %%out = stablehlo.multiply %%arg0, %%two : tensor<%dxf32>
    return %%out : tensor<%dxf32>
  }
}
',
  n_elt,
  n_elt,
  n_elt,
  n_elt,
  n_elt
)
donate_exec <- pjrt_compile(
  pjrt_program(src = donate_mlir, format = "mlir"),
  device = device
)

# Correctness, once: the RAWSXP must move input -> output, the input must be
# emptied, and the doubled bytes must read back correctly through the migrated
# keepalive even after the input is dropped and the GC is stressed.
x <- pjrt_buffer(rep(1.5, n_elt), dtype = "f32", device = device)
x_prot <- impl_test_xptr_prot(x)
stopifnot(is.raw(x_prot))

out <- pjrt_execute(donate_exec, x)
if (!identical(impl_test_xptr_prot(out), x_prot)) {
  stop("donated input's RAWSXP did not migrate to the output's prot slot")
}
if (!is.null(impl_test_xptr_prot(x))) {
  stop("donated input's prot slot was not cleared after donation")
}

rm(x)
invisible(vcells_mb()) # finalize the donated input XPtr under collection
first <- as_array(out)[[1L]]
if (abs(first - 3.0) > 1e-6) {
  stop(sprintf("readback through migrated keepalive wrong: got %g, want 3", first))
}
cat("  migration ok: RAWSXP moved input -> output, bytes survived (1.5 * 2 = 3)\n")

# Stress: donate-and-discard many times. Each output aliases its input's RAWSXP
# in place, so the live working set is ~one chunk; if the migrated keepalive
# leaked, resident memory would grow with the iteration count instead.
rm(out)
donate_baseline <- vcells_mb()
donate_peak_mb <- 0
for (i in seq_len(n_chunks)) {
  res <- tryCatch(
    {
      xi <- pjrt_buffer(numeric(n_elt), dtype = "f32", device = device)
      oi <- pjrt_execute(donate_exec, xi)
      # Touch the output's prot so a broken migration (NULL prot / freed bytes)
      # is caught rather than silently passing.
      if (is.null(impl_test_xptr_prot(oi))) {
        stop(sprintf("iter %d: output has no migrated keepalive", i))
      }
      NULL
    },
    error = function(e) e
  )
  if (inherits(res, "error")) {
    cat(sprintf("\nFAILED at donate iter %d: %s\n", i, conditionMessage(res)))
    stop("donation stress loop failed")
  }
  if (i %% 10L == 0L) {
    resident <- vcells_mb() - donate_baseline
    donate_peak_mb <- max(donate_peak_mb, resident)
    cat(sprintf("  iter %3d  resident=%+.0f MB\n", i, resident))
  }
}
cat(sprintf("Peak resident growth (donation): %.0f MB\n", donate_peak_mb))
if (donate_peak_mb > budget_mb) {
  stop(sprintf(
    "donation peak resident %.0f MB exceeds %.0f MB budget -- migrated keepalive leaks",
    donate_peak_mb,
    budget_mb
  ))
}

# ---------------------------------------------------------------------------
# Invariant 5: a non-donating pjrt_execute() must keep its zero-copy CPU input
# alive until the (async) execution has finished reading it. This is the
# "not freed too early" half of the input lifetime (Invariant 3 covers the
# "eventually released" half). pjrt_execute dispatches asynchronously and reads
# its inputs on a background thread; on CPU an input's bytes live in an R RAWSXP
# that PJRT only aliases, so dropping the input and running the GC mid-flight
# must not reclaim it -- otherwise the running computation reads freed memory
# (use-after-free).
#
# We observe this directly: a finalizer on the input fires iff it is collected.
# A correct result additionally proves the bytes were not freed mid-flight.
# Single execution in this fresh process, so no unrelated buffer finalizer runs
# during our gc() to drain the deferred-release queue early.
# ---------------------------------------------------------------------------
cat("\nNon-donating execute input keepalive:\n")

if (!requireNamespace("stablehlo", quietly = TRUE)) {
  stop("Invariant 5 needs the stablehlo package to build the program")
}

# out = (sqrt(arg1^2) repeated) * arg0. With arg1 == 1 the chain collapses to 1,
# so out == arg0 -- but XLA must evaluate every op (runtime-dependent, not
# constant-folded). arg0 (the *dropped* input) flows only into the final
# multiply, so a correct result proves arg0's bytes survived the GC intact.
ka_n <- 1048576L
ka_reps <- 64L
stablehlo::hlo_func()
ka_arg0 <- stablehlo::hlo_input("arg0", "f32", shape = ka_n)
ka_arg1 <- stablehlo::hlo_input("arg1", "f32", shape = ka_n)
ka_chain <- ka_arg1
for (i in seq_len(ka_reps)) {
  ka_chain <- stablehlo::hlo_sqrt(stablehlo::hlo_multiply(ka_chain, ka_chain))
}
ka_func <- stablehlo::hlo_return(stablehlo::hlo_multiply(ka_chain, ka_arg0))
ka_exec <- pjrt_compile(pjrt_program(stablehlo::repr(ka_func)), device = device)

ka_ones <- pjrt_buffer(rep(1.0, ka_n), dtype = "f32", device = device) # arg1, kept alive
ka_x <- pjrt_buffer(rep(2.0, ka_n), dtype = "f32", device = device)
ka_collected <- new.env()
ka_collected$fired <- FALSE
# ka_x is an external pointer, so it can carry a finalizer; it fires when ka_x is
# garbage collected -- which pjrt_execute must prevent until the execution that
# reads it completes.
reg.finalizer(
  ka_x,
  function(e) {
    ka_collected$fired <- TRUE
  },
  onexit = FALSE
)

ka_out <- pjrt_execute(ka_exec, ka_x, ka_ones)
rm(ka_x)
# Finalizers fire on GC, not on rm(); ka_x is now unreachable from R, so without
# the keepalive it would be collected here.
gc(full = TRUE)
gc(full = TRUE)
if (ka_collected$fired) {
  stop(
    "input buffer was collected while its execution was still pending -- ",
    "use-after-free: pjrt_execute did not keep the input alive"
  )
}
cat("  input stayed alive across GC while the execution was pending\n")

ka_res <- as.numeric(as_array(ka_out)) # blocks until the execution completes
if (!isTRUE(all.equal(ka_res, rep(2.0, ka_n), tolerance = 1e-4))) {
  stop("result is wrong -- the input's bytes were corrupted mid-flight")
}
cat("  result is correct (input bytes intact)\n")

cat("\nCPU host-memory management verified: keepalive pins live buffers,\n")
cat("ordinary GC reclaims discarded ones, donation migrates the backing RAWSXP\n")
cat("from input to output, and a non-donating execute keeps its input alive for\n")
cat("the duration of the async computation -- without leaking or freeing early.\n")

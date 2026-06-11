# Buffer Memory Management

A `PJRTBuffer` is a device-resident tensor. On CPU its bytes live in host RAM;
on CUDA they live in VRAM. In all cases a buffer is an R external pointer
(`Rcpp::XPtr<PJRTBuffer>`) with a finalizer, so its memory is released when the
XPtr becomes unreachable and R's GC finalizes it — buffer lifetime is ultimately
driven by ordinary R garbage collection.

> This document currently covers **CUDA device memory and the out-of-memory
> retry**. The CPU zero-copy story, input/output donation, and layout-aware
> readback are added by later changes.

---

## CUDA: device memory, freed via GC, with an OOM retry

On CUDA a buffer's bytes live in VRAM, entirely separate from host RAM. The
`PJRTBuffer` XPtr owns that device allocation, and it is released only when the
XPtr is finalized and `~PJRTBuffer` calls `PJRT_Buffer_Destroy` — so, as
everywhere, device memory is reclaimed by ordinary R garbage collection of
unreachable buffers.

The problem is that **R's GC cannot see device-memory pressure**. From R's point
of view a `PJRTBuffer` is a tiny external pointer; R has no idea it pins hundreds
of megabytes of VRAM. R will happily let thousands of unreferenced buffers
accumulate before it bothers to collect them — so the GPU runs out of memory
while plenty of it is actually reclaimable, producing a spurious
`RESOURCE_EXHAUSTED`.

### How far the host can run ahead (bounded async dispatch)

PJRT execution is asynchronous: `pjrt_execute()` enqueues work and returns
immediately with output buffers that may not be ready, and a buffer's output
VRAM is allocated when the execution is **enqueued**, not when the GPU finishes
it. So a host loop can dispatch several iterations before any complete, and each
in-flight execution's outputs occupy VRAM simultaneously. This is a real source
of pressure independent of whether a single iteration fits.

It is **not unbounded**, though. XLA's StreamExecutor GPU client throttles
dispatch with a semaphore: `LocalDeviceState` holds a `compute_semaphore_` whose
capacity is `max_inflight_computations`, constructed as **32** for the GPU client
(`xla/pjrt/local_device_state.{h,cc}`, `xla/pjrt/gpu/se_gpu_pjrt_client.cc`).
Once 32 computations are enqueued but not yet finished, the next dispatch
**blocks on the host** until one drains. So the host cannot run arbitrarily far
ahead; worst-case run-ahead memory is bounded by roughly
`32 × (per-iteration allocation)`. For a 1M-float buffer (~4 MB) that is a few
hundred MB at most — usually far less, since iterations typically complete before
the limit is reached.

So the semaphore caps the *number of in-flight iterations*, and (later)
input/output donation caps the *per-iteration footprint* by reusing storage in
place. What neither addresses is the **garbage** that a loop leaves behind:
unreferenced output buffers from completed iterations. In a language with eager
reference counting (e.g. Python/JAX) those are freed the instant they are
dropped. R's mark-sweep GC frees them only on the next collection, and it is
blind to VRAM — so they pile up. That is the gap the OOM retry fills.

### `try_alloc` (`src/utils.h`)

The fix mirrors the R `torch` package: when a device allocation fails with
`RESOURCE_EXHAUSTED`, force a GC and retry once. PJRT allocation calls are
wrapped in `try_alloc(api, alloc_fn, suppress_logs)`:

1. Run `alloc_fn()`.
2. If it returns an error whose code is `PJRT_Error_Code_RESOURCE_EXHAUSTED`,
   destroy that error, call `rpjrt::call_r_gc()`, and run `alloc_fn()` **once**
   more.
3. Surface any other error — or a still-failing retry — via `check_err`.

`alloc_fn` must be callable twice (the underlying `*_Args` struct is refilled on
each call, so reuse is safe). Currently wrapped (`src/client.cpp`):

- `PJRTClient::buffer_from_host_async` → `PJRT_Client_BufferFromHostBuffer`
  (backs `pjrt_buffer()`, `pjrt_scalar()`).
- `PJRTLoadedExecutable::execute_async` → `PJRT_LoadedExecutable_Execute`
  (backs `pjrt_execute()`).

### `call_r_gc` (`src/gc.cpp`)

Runs, in order, on the main R thread:

1. R's `gc(full = TRUE)` — finalizes unreachable `PJRTBuffer` XPtrs, whose
   destructors call `PJRT_Buffer_Destroy` and release device memory.
2. `R_RunPendingFinalizers()` — ensures those finalizers actually run before we
   retry, rather than at some later GC.

It bumps a counter exposed via `impl_gc_call_count()` so tooling/tests can
confirm the retry path fired; `tools/stress-gc-retry.R` is a CUDA stress harness
that allocates large unreferenced buffers until the retry triggers. The retry is
otherwise silent; setting the `PJRT_DEBUG` environment variable makes
`try_alloc` print `"RESOURCE_EXHAUSTED — ran R gc, retrying"` when it fires
(`rpjrt::debug_inform`).

This mechanism is not CUDA-specific in principle — CPU allocations go through the
same wrapper — but it only matters in practice on CUDA, where VRAM is the scarce,
invisible-to-R resource. On CPU allocation failure is rare and the retry is
largely moot.

### Suppressing the recovered OOM's logs

XLA's BFC allocator prints a multi-line OOM report (the "ran out of memory trying
to allocate …" line plus an occupancy dump, all at `LOG(WARNING)`) to stderr on
the failing attempt. Since we recover via gc-and-retry, that attempt's output
would otherwise alarm the user for nothing. The diagnostic is written *during*
`alloc_fn`, before it returns, so to hide it the redirect must already be in
place — i.e. the capture has to wrap *every* call.

`begin_stderr_capture` / `end_stderr_capture` (`src/utils.cpp`) redirect fd 2
into a sink, then on the OOM path discard it and on every other path replay it
(so non-OOM diagnostics are never hidden). The sink is a **process-wide
singleton**, created once and reused (an anonymous RAM-backed `memfd_create` on
Linux, where CUDA runs; `tmpfile()` elsewhere): recreating a temp file per call
dominated the cost (~12–20 µs/call vs ~2 µs reusing one). Reuse is safe because
`try_alloc` only runs on the main R thread, so the sink never has concurrent
users.

The capture is gated by `suppress_logs`, passed `false` on CPU (where OOM
recovery does not happen, so the hot path pays nothing) and `true` on other
backends — sourced from `PJRTClient::is_cpu()` for uploads and a cached `is_cpu_`
on the executable for execute.

### Error handling note

`check_err` (`src/utils.cpp`) extracts the message from a `PJRT_Error`, then
calls `destroy_error` to free it before throwing — the error object is owned by
the caller and must be destroyed even on the failure path. `get_error_code`
likewise destroys any inner error raised by `PJRT_Error_GetCode` itself, falling
back to `UNKNOWN` so the original error still surfaces.

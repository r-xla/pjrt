# Buffer Memory Management

A `PJRTBuffer` is a device-resident tensor. 
Here, device can either be CPU where the buffers lives in RAM or GPU where it lives in VRAM.
Depending on this, the memory management differs.
This document describes it.

This document covers buffer memory and also intermediate memory needed for transfers.
See `buffer-transfer.svg` in this directory for a diagram of both transfer directions.

In all cases a buffer is an R external pointer (`Rcpp::XPtr<PJRTBuffer>`) with
an attached finalizer. The buffer's memory is released when that XPtr becomes
unreachable and R's GC finalizes it — so buffer lifetime is ultimately driven
by ordinary R garbage collection. What that finalization *does* is where CPU
and CUDA diverge.

---

## CPU: host RAM owned by a RAWSXP (so R's GC can see it)

On CPU, a buffer's bytes live in host RAM, and we deliberately make those bytes
an R `RAWSXP` rather than a PJRT-owned allocation. The **primary reason is to
make R's garbage collector aware of the memory.** An external pointer is tiny,
so if a CPU `PJRTBuffer` held its bytes in an opaque PJRT allocation (as a CUDA
buffer holds VRAM), R would have no idea how much memory the buffer pins and
would not trigger collection under that pressure — exactly the blindness
described in the CUDA section below. Backing the buffer with a `RAWSXP` puts the
bytes on R's own heap (counted in `Vcells`), so ordinary R GC sees the pressure
and reclaims the memory when the buffer becomes unreachable.

This applies to **every** CPU buffer, regardless of dtype or entry point. All
CPU creation paths — `pjrt_buffer()`, `pjrt_scalar()`, `pjrt_empty()`, and the
raw constructor — funnel through one helper, `make_cpu_buffer()`
(`src/pjrt.cpp`), which allocates the `RAWSXP`, fills it, and hands it to PJRT.
Because host and device memory are the same physical RAM on CPU, PJRT then
*aliases* that `RAWSXP` for the buffer's lifetime rather than making its own
device copy — the buffer's memory simply *is* the `RAWSXP`.

The mechanics of `make_cpu_buffer()`:

1. Allocate a `RAWSXP` of the exact byte size.
2. Fill it. How depends on the dtype, but the resulting ownership is identical:
   - **convert** when R's type differs from the target dtype (e.g.
     `double → f32`, `logical → pred`, `int → i8`);
   - **`memcpy`** when R's in-memory layout already matches byte-for-byte
     (`double → f64`, `int → i32`, `integer64`, raw input);
   - **nothing** for `pjrt_empty()` (contents left unspecified).
3. Call `PJRT_Client_BufferFromHostBuffer` with
   `PJRT_HostBufferSemantics_kMutableZeroCopy`. PJRT aliases the `RAWSXP`'s
   bytes for the buffer's whole lifetime instead of taking its own copy.
4. Stash the `RAWSXP` in the **protected slot** of the buffer's external
   pointer (the fourth argument to `Rcpp::XPtr`). R's GC then keeps the
   `RAWSXP` alive for *exactly* as long as the `PJRTBuffer` XPtr is reachable.

**Always a private copy.** Step 2 always copies into a *fresh* `RAWSXP`; a CPU
buffer never aliases the user's own R vector. This is what makes
`kMutableZeroCopy` safe: because PJRT exclusively owns the aliased bytes, a
donating `pjrt_execute()` can overwrite them in place without corrupting any R
object the user still holds (R's copy-on-modify would not protect against an
in-place device write).

When the XPtr becomes unreachable, the finalizer order is what makes this safe
(`src/buffer.cpp`, `~PJRTBuffer` line 98):

1. `PJRT_Buffer_Destroy` runs first, releasing PJRT's alias on the bytes.
2. The `RAWSXP` in the protected slot then becomes collectable and R reclaims
   it from its own vector heap.

There is **no double-free**: under `kMutableZeroCopy` the caller owns the host
bytes and PJRT only aliases them — `PJRT_Buffer_Destroy` releases PJRT's handle
but does not free the host allocation.

**Why `kMutableZeroCopy` and not `kImmutableZeroCopy`?** A CPU buffer may later
be *donated* to `pjrt_execute()` as an aliased output, which the immutable
variant explicitly forbids. The caller-side contract is otherwise identical:
don't mutate the bytes while PJRT holds the alias, and keep them alive until
`PJRT_Buffer_Destroy` fires.

### Observable consequence: `object.size()`

Because the backing `RAWSXP` lives in the XPtr's protected slot, R's
`object.size()` traverses it and reports (approximately) the real data size of a
CPU buffer rather than the handful of bytes an external pointer alone occupies.
This is pinned down by the `describe("CPU zero-copy keepalive")` tests in
`tests/testthat/test-buffer.R`.

### Donation keepalive migration

When an executable donates an input to an output, PJRT reuses the input's
storage for the output and invalidates the input handle. On CPU the host-side
`RAWSXP` keepalive must follow the storage from the input XPtr to the output
XPtr. The full story — what donation is, how we learn the alias mapping, the
may-alias/must-alias distinction, and why the migration is gated on
`is_deleted()` — is its own section: see [Buffer donation](#buffer-donation)
below.

---

## CUDA: device memory, freed via GC, with an OOM retry

On CUDA the buffer's bytes live in GPU memory, entirely separate from host RAM.
The `PJRTBuffer` XPtr owns that device allocation, and it is released only when
the XPtr is finalized and `~PJRTBuffer` calls `PJRT_Buffer_Destroy`
(`src/buffer.cpp:98`). So, as on CPU, device memory is reclaimed by ordinary R
garbage collection of unreachable buffers.

The problem is that **R's GC cannot see device-memory pressure**. From R's point
of view a `PJRTBuffer` is a tiny external pointer; R has no idea it pins
hundreds of megabytes of VRAM. (This is precisely the blindness the CPU
`RAWSXP` trick avoids — but there is no equivalent here, since the bytes live on
the GPU, not on R's heap.) R will happily let thousands of unreferenced buffers
accumulate before it bothers to collect them — so the GPU runs out of memory
while plenty of it is actually reclaimable, producing a spurious
`RESOURCE_EXHAUSTED`.

The fix mirrors the R `torch` package: when a device allocation fails with
`RESOURCE_EXHAUSTED`, force a GC and retry once.

### `try_alloc` (`src/utils.h`)

PJRT allocation calls are wrapped in `try_alloc(api, alloc_fn)`:

1. Run `alloc_fn()`.
2. If it returns an error whose code is `PJRT_Error_Code_RESOURCE_EXHAUSTED`,
   destroy that error, call `rpjrt::call_r_gc()`, and run `alloc_fn()` **once**
   more.
3. Surface any other error — or a still-failing retry — via `check_err`.

`alloc_fn` must be callable twice (the underlying `*_Args` struct is refilled on
each call, so reuse is safe).

Currently wrapped (`src/client.cpp`):

- `PJRTClient::buffer_from_host_async` → `PJRT_Client_BufferFromHostBuffer`
  (backs `pjrt_buffer()`, `pjrt_scalar()`, `pjrt_empty()`).
- `PJRTLoadedExecutable::execute_async` → `PJRT_LoadedExecutable_Execute`
  (backs `pjrt_execute()`).

**Suppressing the OOM logs.** XLA's BFC allocator writes the OOM report (the
`"ran out of memory trying to allocate …"` line plus a multi-line occupancy
dump, all at `LOG(WARNING)`) to stderr *during* `alloc_fn`, before it returns —
so to hide it on the recovered path the redirect must already be in place, i.e.
the capture has to wrap *every* call. This matters on CUDA in particular:
because R's GC is blind to VRAM pressure, allocations routinely fail with
`RESOURCE_EXHAUSTED`, recover via the gc-and-retry, and would otherwise print an
alarming-but-recovered dump on each one. The capture (`begin_stderr_capture` /
`end_stderr_capture`, `src/utils.cpp`) redirects fd 2 into a sink, then on the
OOM path discards it and on every other path replays it (so non-OOM diagnostics
are never hidden).

The sink is a **process-wide singleton**, created once and reused — an anonymous
RAM-backed `memfd_create` on Linux (where CUDA runs), falling back to an
unlinked `tmpfile()` elsewhere. Recreating a temp file per call was the dominant
cost (≈12–20 µs/call on a benchmark; it roughly tripled small-op latency);
reusing one sink (`ftruncate` to reset between calls) drops that to ≈2 µs.
Reuse is safe because `try_alloc` only runs on the main R thread, so the sink
never has concurrent users.

`try_alloc` also takes a `suppress_logs` flag, passed `false` on CPU and `true`
on other backends, sourced from `PJRTClient::is_cpu()` for uploads and a cached
`is_cpu_` on the executable for execute. On CPU — where allocation rarely fails
and the gc-retry is moot — the capture is skipped entirely (free is still better
than ≈2 µs); on CUDA the cheap reusable-sink capture keeps every recovered OOM
quiet.

### `call_r_gc` (`src/gc.cpp`)

Runs, in order, on the main R thread:

1. R's `gc(full = TRUE)` — finalizes unreachable `PJRTBuffer` XPtrs, whose
   destructors call `PJRT_Buffer_Destroy` and release device memory.
2. `R_RunPendingFinalizers()` — ensures those finalizers actually run before we
   retry, rather than at some later GC.
3. `process_pending_releases()` — drains the deferred-release queue (see the
   transfer-memory section below).

It also bumps a counter exposed via `impl_gc_call_count()` so tooling and tests
can confirm the retry path actually fired. See `tools/stress-gc-retry.R` for a
CUDA stress harness that allocates large unreferenced buffers until the retry
triggers.

This mechanism is not CUDA-specific in principle — CPU allocations go through
the same `try_alloc` wrapper — but it matters in practice on CUDA, where VRAM
is the scarce, invisible-to-R resource.

### Error handling note

`check_err` (`src/utils.cpp`) extracts the message from a `PJRT_Error`, then
calls `destroy_error` to free it before throwing — the error object is owned by
the caller and must be destroyed even on the failure path. `get_error_code`
likewise destroys any inner error raised by `PJRT_Error_GetCode` itself,
falling back to `UNKNOWN` so the original error still surfaces.

---

## Buffer donation

*Donation* lets `pjrt_execute()` reuse an input buffer's storage for one of its
outputs instead of allocating fresh output memory. This is how anvil implements
in-place updates (e.g. a training step `w <- step(w, grad)` that overwrites the
weights). When an input is donated, PJRT hands the input's device memory to the
output and **invalidates the input handle** — the input must not be used again.

This section collects everything that makes donation correct and safe in `pjrt`,
because the interaction between XLA's aliasing semantics and our R-side memory
management is subtle. The mechanism lives in `impl_loaded_executable_execute`
(`src/pjrt.cpp`) and `load_input_output_aliases_` (`src/client.cpp`); the
behaviour is pinned by the `describe("CPU zero-copy keepalive")` donation tests
in `tests/testthat/test-buffer.R`.

### Where the alias mapping comes from

`PJRTLoadedExecutable` caches the `(input_index → output_index)` donation map at
construction (`load_input_output_aliases_`), read **from the program source we
just compiled**, not from the plugin's compiled executable:

- **MLIR** (the anvil path): the entry function's signature carries donation as a
  `{tf.aliasing_output = M : i32}` attribute on each donated argument (emitted by
  `stablehlo`). We scan the `@main` argument list once and map each donated
  argument's *positional* index to `M`. The index is positional because
  `stablehlo` names values with a global counter (`%0, %1, …`), so the name does
  not encode the parameter number.
- **HLO**: the serialized `HloModuleProto` carries an `input_output_alias`
  config, read directly (mirrors XLA's `HloInputOutputAliasConfig`).

Both paths assume each alias targets a single output or a flat (one-level)
output tuple, so the output index is `output_shape_index(0)` (or 0 for a
non-tuple output) — exactly PJRT's flat output-buffer index. This is what
anvil/stablehlo produce: stablehlo has no tuple type, so outputs are a flat
tensor list that lowers to a one-level HLO tuple. A *nested* output tuple would
flatten to PJRT buffers depth-first and break that identity; the HLO parser
**errors** on a multi-component `output_shape_index` rather than mis-mapping
(it can only arise from hand-written HLO).

This is deliberately **not** recovered from `PJRT_Executable_OptimizedProgram`.
The PJRT C API has no call that returns the alias config; the only way to read it
back from a compiled executable is to parse the optimized HLO from
`OptimizedProgram`, which several plugins (e.g. jax-metal, IREE) leave
unimplemented. Aliasing is part of the program the caller hands us — XLA
preserves caller-specified donation — so parsing the source needs no plugin
support and behaves identically on CPU, CUDA, and Metal. (There is no clean
MLIR→HLO conversion available to us either: that lives in XLA's
`MlirToXlaComputation`, which would require linking the whole LLVM/MLIR stack and
is not exposed through the C ABI.)

### `may-alias` vs `must-alias`: a property of each alias entry

XLA tags every alias entry with a *kind* (`xla/hlo/ir/hlo_input_output_alias_config.h`).
It is a property of a single `output[i] ← parameter[n]` binding, not of a buffer
or of the executable:

- **`kMustAlias`** — the caller *guarantees* the input is donated; the runtime is
  obliged to reuse it. No copy fallback is emitted.
- **`kMayAlias`** — an *optimization opportunity*. The runtime reuses the input's
  buffer only if it is donatable at runtime, and **otherwise silently inserts a
  copy**, leaving the input valid. (From the `SetUpAlias` doc comment in
  `xla/hlo/builder/xla_builder.h`: *"if [the input] is not donated at runtime, a
  copy will be inserted by XLA to prevent buffer clobbering."*)

**anvil's `donate=` produces a `may-alias`, not a `must-alias`.** The
`tf.aliasing_output` attribute lowers via `XlaBuilder::SetUpAlias`, whose `kind`
argument defaults to `kMayAlias`. The compiler can also *introduce* may-aliases
on its own (`OptimizeInputOutputBufferAlias`), pairing a registered buffer donor
with a same-size output when convenient.

So "this input is aliased to that output" does **not** mean the input was
actually donated. Whether it was depends on runtime donatability: an input
listed in `non_donatable_input_indices`, or one that has external references, is
**not** donated, and XLA copies instead.

### Migrate only what PJRT actually donated (`is_deleted()` gate)

Because of the above, we must not assume a parsed alias entry means donation
happened. If we migrated the keepalive and nulled the input handle for a
may-alias that was actually *copied*, we would null a **still-live** input
buffer — leaking its device memory and, on CUDA/Metal, **double-freeing** it when
the input's finalizer later runs `PJRT_Buffer_Destroy` on a handle XLA never
invalidated.

The fix is to ask PJRT what it actually did. After `Execute`, for each declared
alias we check `PJRTBuffer::is_deleted()` (wrapping `PJRT_Buffer_IsDeleted`) on
the input, and only migrate when it reports deleted. This is reliable because
donation flips the input's deleted state **synchronously inside the `Execute`
call**: PJRT's `ConfirmDonation` runs `std::move` on the tracked device buffer
before the C API call returns, so `CommonPjRtBuffer::IsDeleted()`
(`device_buffer_ == nullptr`) is already true on return for a donated input and
false for a copied may-alias. Gating on `is_deleted()` makes our correctness
independent of the alias *kind* and of whatever runtime donatability decision
PJRT made.

### Keepalive migration (CPU)

For each declared alias whose input PJRT actually donated:

1. Move the `RAWSXP` from the donated input XPtr's protected slot to the aliased
   output XPtr's protected slot (`R_SetExternalPtrProtected`).
2. Clear the input XPtr's protected slot.
3. Null the donated input's `PJRT_Buffer*` so its finalizer becomes a no-op
   (PJRT already invalidated the handle during `Execute`).

After this, the output's bytes stay alive for exactly as long as the output XPtr
is reachable (the output's storage *is* the migrated `RAWSXP`), and any operation
on the donated input raises a clean R-level error (`"called on deleted or donated
buffer"`, via `PJRTBuffer::checked_buffer`) rather than crashing. On CUDA/Metal
there is no `RAWSXP`, so steps 1–2 are vacuous; step 3 is what prevents the
double-free.

### Multiple aliases are unambiguous

The alias config is a fixed **1:1 `output → parameter` binding**: each output
aliases at most one parameter (*"a given output buffer shape index can refer only
to one parameter+index"*), and each input is donated at most once. So when
`is_deleted(input_i)` is true, output `i` — its declared partner — is exactly the
buffer that reused `input_i`'s storage; migrating `input_i`'s `RAWSXP` to
`output_i` is correct with no runtime "which output got it" guesswork.

The one case that could be ambiguous — the **same** physical buffer passed as two
donated parameters (`out0←in0`, `out1←in1`, but `in0` and `in1` are the same
buffer A, so A could only go to one of the outputs) — cannot occur: PJRT rejects
it before any donation via `TestBufferDonationClashes` (`xla/pjrt/utils.cc`),
returning *"Attempt to donate the same buffer twice in Execute()"*. The sibling
cases `f(a, donate(a))` and `f(donate(a), a)` are rejected too. Our `try_alloc` /
`check_err` surfaces this as a clean R error, and since it fails before any
donation, all inputs remain valid.

### Why `kMutableZeroCopy` (recap)

This is also why CPU buffers are created with
`PJRT_HostBufferSemantics_kMutableZeroCopy` and always back a *private* copy of
the user's data (see the CPU section above): a may-alias output may donate and
overwrite the input's bytes in place, so the aliased `RAWSXP` must be exclusively
owned by `pjrt`, never the user's own R vector. The immutable zero-copy variant
forbids donation outright and so cannot be used.

---

## Reading back: respecting the device layout

`as_array()` / `as_raw()` copy a device buffer to host via
`PJRT_Buffer_ToHostBuffer`, which delivers the bytes in the **buffer's own
device layout**. That layout is *usually* row-major (dense, major-to-minor),
but it is not guaranteed: an executable output can be pinned to a different
layout (e.g. column-major via `mhlo.layout_mode = "{0,1}"`), and XLA is free to
choose layouts for performance. We cannot ask `ToHostBuffer` to normalize for
us — the CPU runtime accepts a `host_layout` argument but ignores it (it does
not relayout), so we must respect the device layout on our side.

The readback therefore queries the layout and reorders the bytes itself:

1. `PJRTBuffer::minor_to_major()` (`src/buffer.cpp`) reads
   `PJRT_Buffer_GetMemoryLayout` and returns the layout as a minor-to-major
   permutation of logical dimensions (row-major is `[n-1, …, 0]`, column-major
   `[0, …, n-1]`). It handles exactly the dense, untiled layouts our readback
   can faithfully reorder. 0-D/1-D buffers return the trivial order (layout
   irrelevant). For rank ≥ 2 it **errors** on anything it cannot honor — a
   strided layout, an actual tiling (`num_tiles > 0`), or a rank mismatch —
   rather than silently assuming row-major and returning wrong data. These
   cases do not arise on CPU/CUDA/Metal (all hand back dense untiled layouts;
   tiling is a TPU feature pjrt does not support), so the error is a guard
   against a future/exotic backend, not a path hit in normal use. A genuinely
   non-row-major *untiled* output (e.g. column-major) is a `minor_to_major`
   permutation with `num_tiles == 0` and is handled, not errored.
2. The permutation rides along the async path: `impl_buffer_to_host_async`
   returns it, the `PJRTArrayPromise` carries it, and `value()` passes it to
   `impl_raw_to_array`.
3. `device_to_row_major()` (`src/utils.h`) reorders the device-physical bytes
   into logical row-major; the existing `row_to_col_order()` then transposes to
   R's column-major. When the device layout is already row-major (the common
   case) `device_to_row_major()` is a plain copy, so there is no extra work.

Without this, a non-row-major output would be silently misread as row-major and
materialize transposed/garbled — see the regression test "as_array respects a
non-row-major executable output layout" in `tests/testthat/test-buffer.R`.

---

## Appendix: temporary host memory for transfers

On a non-CPU device, host and device memory are distinct, so the host bytes are
only **transient staging** for the upload — they need to live just until the
async copy into VRAM completes. (On CPU this concern does not arise — the
aliased `RAWSXP` *is* the buffer.) Two strategies appear in `src/pjrt.cpp`,
chosen exactly as on CPU by whether a type conversion is needed:

- **Owned copy** (`create_buffer_from_array_async`, non-CPU branch, and
  `impl_client_buffer_empty`): when the dtype differs (or for `pjrt_empty()`),
  copy/convert R's data into a `std::vector`, hand its pointer to PJRT, and
  `delete` the vector in the transfer event's `on_ready` callback.
- **Preserve the R object** (`create_buffer_from_array_async_no_convert`,
  `create_buffer_from_raw`): when R's layout already matches the dtype
  byte-for-byte, the R data is handed to PJRT *directly* — genuinely zero-copy,
  no host copy at all. `R_PreserveObject` pins the R object and it is queued for
  release once the transfer completes.

Because the `on_ready` callback fires on a **PJRT callback thread**, it cannot
call `R_ReleaseObject` directly. Instead it calls `rpjrt::queue_release(obj)`,
which pushes the SEXP onto a mutex-guarded queue
(`src/deferred_release.{h,cpp}`). The queue is drained on the main R thread by
`process_pending_releases()`. There is no dedicated reaper — instead it is
called at every main-thread PJRT operation that is part of a normal workflow,
so the queue self-clears during ordinary use:

- `~PJRTBuffer` (so it runs during ordinary GC of any buffer);
- `pjrt_execute()` (`impl_loaded_executable_execute`, at the start) — the
  hot-loop entry point, so preserved inputs don't accumulate across iterations;
- `await()` / `is_ready()` on a buffer, and the readback await
  (`impl_host_data_await`, behind `as_array()`);
- `call_r_gc()` (the OOM-retry path).

It only ever releases objects whose transfer has already completed (and so were
enqueued); an in-flight upload's object stays pinned until its `done_with_host_buffer`
event fires.

---

## Invariants (summary)

- **Every** CPU buffer's bytes live in a `RAWSXP` in its XPtr's protected slot
  — regardless of dtype or entry point (all paths go through `make_cpu_buffer`)
  — so the memory is counted on R's heap and R's GC can see and reclaim it, and
  they stay alive for exactly the XPtr's reachable lifetime.
- A CPU buffer's `RAWSXP` is always a *private copy*: it never aliases the
  user's own R vector, so a donating `pjrt_execute()` can mutate it in place
  safely.
- `PJRT_Buffer_Destroy` always runs before the backing `RAWSXP` is reclaimed
  (finalizer ordering), so PJRT never reads freed memory and the bytes are never
  double-freed.
- An input is treated as donated **only if PJRT actually donated it** — confirmed
  per-input via `is_deleted()` after `Execute`, not inferred from the declared
  alias (which, being a may-alias, PJRT may copy instead). A genuinely donated
  input migrates its keepalive to the aliased output and is neutralized (null
  `PJRT_Buffer*`, cleared protected slot, R-level error on use); a copied
  may-alias is left untouched and stays usable. The same physical buffer can
  never be donated twice in one `Execute` (PJRT errors out), so the
  output↔input mapping is unambiguous.
- A CUDA buffer's device memory is freed only when its XPtr is finalized; a
  `RESOURCE_EXHAUSTED` allocation triggers exactly one GC-and-retry, and any
  other error (or a second failure) propagates to R.
- Host objects preserved for in-flight non-CPU transfers are released via the
  thread-safe deferred-release queue, drained on the main R thread.
- Readback respects the device buffer's actual layout: the bytes are reordered
  from the device's minor-to-major order into the R array, so a non-row-major
  (untiled) buffer materializes correctly rather than transposed. A layout the
  readback cannot faithfully reorder (strided, tiled, or rank-mismatched —
  none of which occur on CPU/CUDA/Metal) raises a clear error rather than
  silently returning wrong data.

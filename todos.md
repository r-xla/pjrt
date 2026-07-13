* [x] remove this sentinel or understand what it's used for
      Removed. Its only site was the device conflict under the infer policy;
      that is now a native error, so `dispatch_sentinel()` is gone from pjrt and
      both fallback branches are gone from anvl.
* [x] understand how "auto" dispatch is now handled
      anvl's `jit_auto()` detects the backend at call time (a direct field
      scan over the evaluated args) and lazily builds one per-backend
      JitFunction, each with its own pjrt Dispatcher; the fast entry
      (`jit_run_args`) skips the re-capture. The dispatcher itself never sees
      "auto" -- by the time it runs, the backend is fixed.
* [x] move also the output wrapping functionality into dispatcher.cpp
      Done as part of the backend-abstraction refactor (see
      specs/2026-07-10-dispatcher-backend-abstraction-design.md): the pjrt
      engine wraps output buffers into AnvlArrays from per-entry templates and
      re-nests them natively via out_tree; `dispatch()` now returns the call's
      finished result and anvl's `jit_wrap_outputs_native()` is deleted.
* [x] ensure the benchmark shows we are now reasonably fast
      benchmarks/jit-launch-overhead.R updated to the new layering (the wrap
      is inside `dispatch()` now, and the old keyhash self-test hook is gone).

## Deferred bugs (found while making the dispatcher own validation)

These are all outside the dispatcher's core logic. Revisit once it is settled.

* [ ] **anvl traces a static `AnvlArray` as a program input.** Any `AnvlArray`
      passed as a static argument makes the compiled program expect one more
      buffer than execution supplies:
      `Execution supplied 2 buffers but compiled program expected 3 buffers`.
      True for `"xla"` and `"plain"` arrays alike, so it is not about the
      backend. `dispatch.cpp` now rejects the case up front, but the
      ahead-of-time `xla()` path has no such guard and still hits it.

* [ ] **anvl's `is_valid_r()` has no `is.object()` guard.**
      `(is.numeric(x) || is.logical(x)) && (is.array(x) || length(x) == 1L)`
      accepts a classed numeric such as `structure(1, class = "myclass")`, which
      then traces as a real input. pjrt's `classify_rdata()` rejects any classed
      value, so the two validators disagree. The dispatcher now errors by name;
      the `xla()` AOT path still accepts it and fails later with the buffer-arity
      message above. Fix by adding the guard so both agree.

* [ ] **`flatten_rec()` recurses without a depth guard.** A deeply nested
      argument (~50k levels) overflows the C stack and kills the R session; it
      cannot be caught. Needs an explicit depth limit that raises an R error.

* [ ] **Tree names are read encoding-blind.** `flatten_rec()` builds names with
      `std::string(CHAR(nm))` rather than `Rf_translateCharUTF8()`, so the same
      name in UTF-8 and in latin1 gives two different trees, hence two cache
      entries for one call signature. Wasted compiles, not wrong results.

* [x] **Non-ASCII string statics collapse to a sentinel in `hash_atomic()`.**
      Fixed. `hash_atomic()` now folds `Rf_translateCharUTF8()` (guarding
      "bytes"-encoded strings, which can't be translated), mirroring R's
      `shash()`: the same text in latin1 and UTF-8 folds alike, and distinct
      non-ASCII strings hash apart instead of sharing one bucket. Covered by the
      "strings key on their UTF-8 content, across encodings" test in
      test-dispatch.cpp.

* [ ] **`KeyLeaf` hashes are computed inside `unordered_map`'s hasher.**
      `hash_atomic()` on a static can allocate, and `r_identical()` can longjmp,
      from within the map's hash/equality callbacks. Precomputing each leaf's
      hash at `KeyLeaf` construction would keep allocation and longjmp out of
      libstdc++'s internals.

* [ ] **`nv_scalar()` is never dtype-ambiguous.** `nv_scalar(3)` and
      `nv_scalar(3, dtype = "f32")` both report `ambiguous = FALSE`, while a bare
      literal `3` is `ambiguous = TRUE`. So `f(x, nv_scalar(3))` and `f(x, 3)`
      have different avals and compile twice, even though pjrt's dispatcher now
      merges kArray and kRData leaves of an equal aval. If a scalar built without
      an explicit dtype should be ambiguous, this is where to fix it -- and the
      merge would then fire where users actually hit it.

* [ ] **The key's device can differ from the entry's device.** A jitted closure
      over, say, a CUDA constant, called with only literal arguments, keys on the
      *default* device (nothing named one) while `compile_xla()` compiles for the
      constant's device. Never a wrong result -- the constant is fixed per
      dispatcher, so every entry compiles for the same device -- but changing the
      default forces a redundant recompile of an identical program.

* [ ] **Unverified benchmark claim.** After collapsing `kXlaArray`/`kQuickrArr`
      into `kArray`, `benchmarks/jit-launch-overhead.R` reported `machinery_us`
      dropping ~10.5 -> ~5.2 us and `keyhash_us` ~2.8 -> ~1.1 us. The change
      cannot plausibly account for that; it was never confirmed against machine
      state. Do not cite these numbers until re-measured A/B.

## Deferred design cleanups

* [x] **`Engine::supports_move_inputs()` is the wrong abstraction; delete it.**
      Done. It was a negative capability predicate -- the core asked the engine
      for permission and then did the work itself -- the same smell as the `ok` /
      `is_anvl` flags that became `std::optional`: a bool standing in for the
      presence of a capability.

      The fix was to let the closure engine pin as well. `ClosureEngine` already
      delegates execution, output wrapping, and re-nesting wholesale to `r_fun`,
      so "place your own inputs on the device you compiled for" is one more
      clause in a contract that is already total delegation, not a new kind of
      trust. Under a pinned dispatcher the backend's `compile` callback is what
      chose the device, so the `r_fun` it returns closes over it: pjrt passes
      nothing extra and does nothing at execute time. Written into `?dispatcher`,
      including that `r_fun` sees only `$data` and not `$device`, so its placing
      must be idempotent.

      Gone: the virtual predicate, the `move_inputs requires engine = "pjrt"`
      guard, and the `move_inputs` parameter on `Engine::run()`. The policy is
      now engine state, fixed at construction (`make_engine()`), because placing
      an input is the engine's own business -- only it knows what `$data` is.
      `Dispatcher::move_inputs_` stays: the core needs it for the key, the engine
      for the placing, and both are set from the one R-level argument.

      **Deliberately not done: the DevicePolicy object (part 2 of the original
      plan).** Once the closure engine may pin, the coupling that motivated it is
      gone -- there is no longer an invariant for pjrt to enforce across "device
      leaves the key" and "inputs get placed", because a closure engine legitimately
      places nothing itself. What remains is one immutable bool used by two
      collaborators for two different jobs. Wrapping that in a polymorphic policy
      would have to reach into `PjrtEntry`'s client/device to do the placing
      (tighter coupling, not looser) and would add a virtual call per array leaf
      to a hot path with an explicit launch-overhead benchmark. Revisit only if a
      third device policy appears.

* [x] **The compile callback should declare its output dtypes/shapes; build the
      wrap templates eagerly.** Done. `ambiguous_out` (a logical vector) is
      replaced by `out_avals`: one `list(dtype, shape, ambiguous)` per output
      leaf -- the same shape as the input avals the callback already *receives*
      in `info$avals`, so anvl's `avals_from_dispatch()` and the new
      `compile_xla()` side are exact inverses.

      The wrap material used to come from two sources: `out_tree` /
      `ambiguous_out` from the callback, but dtype and shape scavenged off the
      real output buffers on the entry's *first execution*. Now the backend
      single-sources the description of its own outputs -- the arrangement the
      closure engine already has by construction (`r_fun` wraps its own outputs;
      pjrt has no opinion). Chosen over reading PJRT's `PJRT_Executable_Output*`
      C API, which would be authoritative but is a pjrt-engine-only mechanism
      with no analogue for closure.

      Payoff, all realized: `build_entry()` builds the templates on the cold path,
      the `if (pe->templates == R_NilValue)` branch is gone from the hot path, and
      `PjrtEntry` is immutable once constructed -- so `Engine::run()` now takes
      `const CacheEntry&`, which confines the `preserve()` rooting discipline to
      `build_entry`.

      Trade-off accepted knowingly: the templates could not previously lie (they
      were read from the buffers the executable really produced). A backend whose
      declaration disagrees with its executable now stamps a wrong `$dtype`/`$shape`
      onto correct buffers. The one half pjrt can still settle is kept: `run()`
      checks the declared output count against the executable's actual one.

      anvl (perf/dispatch-overhead) updated in lockstep: `compile_xla()` returns
      `out_avals`, and the AOT `xla()` path -- which wraps its own outputs and only
      wants the flags -- derives `ambiguous_out` from it.

* [x] **`Engine::run()` was handed the cache key and a leaf-parallel SEXP array;
      give it the call's inputs instead.** Done. It used to receive `exec_sexp`
      (parallel to *all* leaves, statics included, a static's slot holding the
      leaf itself) plus the `CacheKey`, and had to re-derive from them which
      leaves were actually execute inputs -- counting the non-statics to size its
      input list, then skipping the statics again while filling it. That is the
      dispatcher core's knowledge, recomputed inside every engine on every call.

      The core now builds the input sequence while it classifies (`ExecInput`:
      the SEXP, its aval, and whether it must be uploaded) and hands the engine
      exactly that. The count *is* `inputs.size()`; there is nothing to skip; and
      `Engine::run()` no longer takes a `CacheKey` at all -- the engine layer no
      longer knows what a cache key or a static argument is.

      Protocol consequence: `r_fun` now receives only the dynamic leaves. It used
      to get the statics too and drop them itself, re-asserting they matched --
      a check that could never fire, since the dispatcher keys statics with
      `identical()`, so a cache hit already proves it. anvl's quickr flat wrapper
      loses that block; the non-flat wrapper (no dispatcher in front of it) keeps
      `quickr_assert_static_args()`.

* [ ] replace protect/unprotect with Rcpp::Shield<SEXP>
* [ ] Use Rcpp objects so we don't even need PROTECT/UNPROTECT
* [ ] Avoid using R_PreserveObject as Rcpp has it's own internal precious list now
* [ ] is move_inputs really needed per cache entry> I think it should be a global flag.

* clean docs for dispatcher.R

To read:
* [ ] finish reading pjrt dispatch_engine.cpp
* [ ] dispatcher.cpp

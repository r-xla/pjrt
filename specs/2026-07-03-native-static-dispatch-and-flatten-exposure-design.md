# Native dispatch: static args, pjrt-owned pytree, one cache for all backends

Date: 2026-07-03
Status: Approved (design) — implementation phased A → B → C, starting with A
Repos touched: `pjrt` (C++ + R), `anvl` (R)

## Decisions

0. **Naming:** `filter_list_node` → `filter_by_names`; `reindex_tree` + the
   value/grad `ListNode` combine → `tree_concat` (consolidated per "no baggage").
   A combined `flatten_with_tree` primitive is **deferred** (YAGNI) until the
   double-traversal on R compile paths is shown to matter.
1. **The entire pytree module moves to pjrt; the `Node` is opaque to anvl.**
   (Supersedes the earlier "keep R S3 Nodes" idea.) `Node` becomes a C++ object
   behind an external pointer, owned by pjrt. anvl never inspects its fields — it
   only builds, flattens, unflattens, and compares. pjrt owns the whole module:
   the performance-critical + structural ops in C++, and the leaf-function
   orchestration in pjrt's R layer over those primitives (see Part B). anvl
   re-exports the public names and deletes `R/flatten.R`.
2. **One spec, three parts, full arc (A → B → C).** The user wants the native
   pjrt_dispatcher to eventually cover *everything* (design-doc §6). Part A (native
   static) is self-contained and approved; Part B moves the pytree module to pjrt
   (opaque Node); Part C generalizes the pjrt_dispatcher to all backends and cases,
   retiring every R fallback/cache. The parts are layered — A and B are the
   foundation C builds on. **Implementation is expected to be phased A → B → C**
   (each independently landable/reviewable); this spec captures the whole arc.

## Background

The native eager-dispatch fast path lives in `pjrt/src/dispatch.cpp` and handles
`device = NULL` jit calls whose leaves are all xla `AnvlArray`s. Two problems
motivate this work:

- **Static args are unhandled and the key machinery is dead.** Any call carrying
  a static value (a flag, an integer `n`, …) makes `impl_dispatch_run` return the
  sentinel, so the R-side `xlamisc::LRUCache` does all caching/compilation
  (confirmed empirically: a static jit grows the R cache, native stays at 0). The
  `KeyLeaf::is_static` / value-identity branch in `CacheKeyHash`/`CacheKeyEq` is
  never reached — `is_static` is never set `true` anywhere. An uncommitted change
  sets its `identical()` flags to `16` — which (contrary to what this
  paragraph originally claimed) turned out to be exactly R's default
  `identical()`; see A.5.
- **Two sources of truth for flatten.** `dispatch.cpp` ports anvl's
  `R/flatten.R` (`flatten`/`build_tree`/`unflatten`) into C++. The two were only
  "verified by hand" and can silently drift; drift corrupts the cache key or the
  output wrapping. The design doc (`anvl/benchmarks/cpp-hot-path-design.md` §2)
  already assigns ownership of the Node tree to pjrt — this finishes that.

Consumers of the three primitives are entirely within anvl (`reverse.R`,
`jit.R`, `graph.R`, `backend-xla.R`, `primitives.R`, `graph-to-quickr.R`); no
sibling package uses them. The only class extending these generics beyond
list/NULL/leaf is anvl's `MarkedArgs`.

---

## Part A — Native static-arg support

### A.1 Interface

- **pjrt C++:** `impl_dispatch_create(int capacity, SEXP miss_fn, SEXP static_names)`.
  `static_names` is a `STRSXP` of top-level argument names (empty for pure-dynamic
  jits). Stored on `pjrt_dispatcher` as `std::unordered_set<std::string> static_names_`.
- **pjrt R:** `pjrt_dispatcher(capacity, compile, static = character())`.
- **anvl:** `jit_xla_impl` passes `static = static` when creating the pjrt_dispatcher.
  The compile callback `jit_xla_compile_cb(f, static, donate)` already handles
  static via `jit_prepare_args(args, static, …)` — **no change**.
- `pjrt_dispatch(pjrt_dispatcher, args)` and `impl_dispatch_run(handle, args)`
  signatures are **unchanged** — static-ness comes from the handle.

### A.2 Native flatten + static marking

Static-ness is determined **only at the top level** by argument name, then
propagated to every leaf beneath a static arg — exactly `MarkedArgs` semantics
(`rep(is_marked, times = subsize)`).

- `flatten_rec` gains a parallel `std::vector<char>& is_static` output and a
  `bool inherited_static` input: each `LeafNode` pushes `inherited_static`;
  recursion passes it down; `NullNode` contributes no leaf (so a static arg that
  *is* `NULL` yields no `is_static` entry — matches `subsize = 0`).
- Top-level marking happens in `impl_dispatch_run` (not inside the generic
  `flatten_rec`): for each top-level child `k`,
  `child_static = !static_names_.empty() && has_names && static_names_.count(names[k])`,
  then recurse with that flag. When `static_names_` is empty the marking step is
  skipped entirely → **zero added work on the pure-dynamic hot path**.

### A.3 Classification & execution assembly

Per leaf:
- **static** → `KeyLeaf{is_static = true, value = leaf}` (any SEXP; no extraction).
- **dynamic** → must be an xla `AnvlArray` (`extract_leaf`, else sentinel);
  contributes an `aval` and its buffer.

`device` is taken from the **first dynamic buffer**. If there are **no dynamic
buffers** (all-static or zero-arg), return the sentinel so R keeps doing device
inference from the trace (covers `nv_scalar(1, device = x)` static-device-arg
cases — unchanged).

Execution inputs = `const_arrays ++ dynamic buffers ++ phantoms`. **Static leaves
are excluded from execution** (they are baked into the executable as constants),
matching R's `args_flat[!is_static_flat]`.

### A.4 Static key lifetime (GC)

The cache **key** now holds SEXPs for static leaves.

- On a miss/insert: `R_PreserveObject` each static key value and stash it in the
  `CacheEntry` as `std::vector<SEXP> static_key_values`; `release_entry` releases
  them on eviction/teardown. The key's `KeyLeaf::value` points at the same SEXP,
  kept alive by that preserve.
- The pjrt_dispatcher only `set()`s on a miss (always a fresh key → `push_front`,
  never the replace branch), so there is no double-preserve.
- Lookup keys are transient (stack-built each call; their SEXPs are the live call
  args, rooted by R) and preserve nothing.

### A.5 The `identical()` flags (CORRECTED during implementation)

**The original claim here (40 right, 16 wrong) was inverted.** Per R's
`identical.c`, `R_compute_identical`'s flag bits are *USE* bits: a bit is set
when the corresponding property must be *compared*. Default `identical()`
(`ignore.bytecode=TRUE, ignore.environment=FALSE, ignore.srcref=TRUE`) sets
only `IDENT_USE_CLOENV (16)` — compare closure environments, ignore
bytecode/srcref — so **`flags=16` is correct** and is what the code keeps.
`40 = USE_BYTECODE | USE_SRCREF` would have ignored environments (wrongly
merging distinct closures) while comparing bytecode and srcrefs. Verified
empirically; pinned by `impl_dispatch_static_key_eq` tests covering value-,
environment-, bytecode-, and srcref-sensitivity.

---

## Part B — Move the pytree module to pjrt; `Node` opaque to anvl

The whole module moves to pjrt. `Node` is a C++ object behind an external
pointer. `flatten_rec` (C++ `Node` + leaves) remains the single traversal that
encodes the flatten semantics (`is_bare_list`, NULL handling, order, names); the
dispatch hot path keeps using it directly (stack `Node`, no xptr allocation),
while the exposed API wraps a heap `Node` in an xptr. Same type, one logic.

### B.1 Native C++ surface (pjrt)

The performance-critical and structural ops — anything that must *walk* the tree
— are C++:

- `build_tree(x) -> Node xptr`, `flatten(x) -> list`, `unflatten(node, leaves) -> obj`.
- `tree_equal(a, b) -> logical`, `tree_hash(node)` — structural (already exist for
  the cache key; exposed here).
- `tree_size(node)`, `child_sizes(node) -> int[]`, `flat_names(node) -> chr[]`
  (per-leaf top-level arg name), `tree_path(node, i) -> string` (leaf → path).
- `filter_by_names(node, names) -> Node` — keep top-level children by name +
  reindex (autodiff `reverse.R:135`, gradient `out_tree`).
- `tree_concat(nodes, names) -> Node` — build a parent over child Nodes,
  reassigning leaf indices contiguously in one pass (replaces the two
  `reindex_tree(..., shared counter)` + `ListNode(...)` in `reverse.R:513-518`).
- `mask_from_names(node, names) -> logical` — top-level-name → flat leaf mask.
  **The single marking primitive**: subsumes Part A's static marking, autodiff's
  wrt mask (`reverse.R:171`), and `stablehlo.R`'s donate mask; lets us **delete**
  `mark_some`/`MarkedArgs`/`MarkedListNode` (no marked subclass needed).
- `tree_repr(node) -> string` — canonical structural string for error messages /
  `format`/`print`. (Also serves as the R-fallback cache key during the B phase;
  that use disappears once Part C removes the R fallback.)
- `tree_diff(a, b)` — native walker returning the first divergence (prefix path +
  the two diverging subtrees) for `pmap_tree`'s error message.

**Why native ops, not sub-node accessors.** We deliberately do *not* expose
`children(node)`/`leaf_index(node)` to let the old R algorithms walk the tree: a
child handle would alias into a **parent-owned** sub-tree (dangling-pointer
hazard if the parent is freed) and would make the Node non-opaque. Instead every
op above walks inside C++ and returns either a **fully-owned result Node** or a
**plain R value** (mask, sizes, names) — no sub-node handles escape to R.

### B.2 R-layer orchestration (pjrt/R)

`map_tree` and `pmap_tree` are thin orchestration — build/flatten/apply an **R**
function per leaf/unflatten — so they live as **R functions in pjrt's R layer**
over the native primitives (no C++ leaf-callback machinery; cli-style errors via
`tree_path` preserved). `pmap_tree`'s structure check uses native `tree_equal`,
and its mismatch message uses native `tree_diff`.

### B.3 anvl changes

- Delete `R/flatten.R`. Re-export the public tree API from pjrt (as `reexports.R`
  already does for tengen), so `anvl::flatten`, `build_tree`, `unflatten`,
  `tree_size`, `tree_path`, `map_tree`, `pmap_tree`, `tree_diff` keep working.
- Rewire the tree-internals consumers to the native API (no more `$nodes`/`$names`):
  - `reverse.R` (autodiff): `filter_list_node` → `filter_by_names`; the two
    `reindex_tree(…, shared counter)` + `ListNode(list(value,grad), …)` →
    `tree_concat`; `vapply(in_tree$nodes, tree_size)` → `child_sizes` (or
    `flat_names`); `flat_mask_from_names` → `mask_from_names`;
    `build_tree(mark_some(args, wrt))` → `build_tree(args)` + `mask_from_names(…, wrt)`.
  - `stablehlo.R`: `flat_mask_from_names(graph$in_tree, donate)` → `mask_from_names`.
  - `jit.R` / `backend-xla.R`: the R fallback's `jit_prepare_args` builds
    `in_tree` and `is_static_flat` via `build_tree` + `mask_from_names` (drops the
    `MarkedArgs` path and the flat/non-flat special-case — native `build_tree` is
    already fast).
  - `graph.R`: `in_tree`/`out_tree` graph fields now hold opaque Node xptrs;
    consumers use native ops. A cached `out_tree`'s C++ Node must be preserved for
    the cache entry's lifetime (like `exec`/`const_arrays`).

### B.4 Layering

pjrt owns only the *generic* tree (build/flatten/unflatten/compare/mask/filter/…).
It never learns anvl concepts — `MarkedArgs` is gone, and `mask_from_names(names)`
is fully generic (used for static, donate, and wrt alike). Respects the stack and
matches design-doc §2.

---

## Part C — One native dispatch/cache engine for all backends and cases (§6)

Goal: the native pjrt_dispatcher becomes the **single** cache + dispatch path for every
jit call. Retire the XLA R fallback (its sentinel branch and `cache`), the quickr
R cache, and `jit_call_xla`. One native LRU cache per jitted function. The only R
left on a hit is `currently_tracing()` (nested-jit pass-through); on a miss, the
compile callback; and for quickr, the compiled R closure at execute.

### C.1 Cases the pjrt_dispatcher must cover
- xla `AnvlArray` leaves (Part A).
- Static leaves (Part A).
- **R-literal / R-array dynamic leaves** (`x + 1`, `x + matrix(...)`): aval from
  `default_dtype` + shape; uploaded to the resolved device at execute via the
  existing buffer impls (the native equivalent of `pjrt_scalar`/`pjrt_buffer`).
- **Device / `device_arg` moves**: the target device is part of the key; inputs
  `copy_buffer`'d to it natively. `device_arg` reads the device from the named
  (static) arg. The pjrt_dispatcher is created **unconditionally** (not only when
  `device = NULL`).
- **quickr backend**: R-array-backed leaves; aval from dtype/shape (no device);
  execute calls the cached compiled R closure on the flat R arrays.

The SENTINEL/fallback set becomes empty for real inputs; genuinely invalid inputs
become errors, not fallbacks.

### C.2 Backend / execute strategy
The pjrt_dispatcher is created per (function, backend, device policy) — a jit fixes its
backend, and the `"auto"` backend stays a thin R multiplexer
(`jit_auto_detect_backend` picks the backend in R, then routes to that backend's
lazily-created pjrt_dispatcher). The cache entry becomes a small variant:
- **xla:** PJRT `exec` + consts + phantom specs → native
  `impl_loaded_executable_execute` (as today).
- **quickr:** an R closure (`compiled$fun`) → invoked via `Rcpp::Function` on the
  flat R-array leaves.
Execute branches on the entry kind; flatten / key / cache / wrap stay generic.

### C.3 Native input materialization & device resolution
Assemble inputs natively, reusing existing pjrt impls (do **not** reimplement):
xla-array leaves pass through (or `copy_buffer` to the target on a move);
R-literal/array leaves upload via the buffer impls; phantoms as in Part A; quickr
leaves pass their R arrays straight to the closure. Device resolution reproduces
`jit_prepare_args`' rules natively: first input device wins, conflicting devices
error, no device → default; `device_arg` extracts from the named static arg.

### C.4 Error-message parity (path-annotated, cli-formatted)
`check_jit_input` (deleted here) produces path-annotated errors via `tree_path`:
`"Found AnvlArray input {.arg {path}} on unexpected device …"` and
`"Attempted to autoconvert {.arg {path}} to an {.cls AnvlArray}."`. The native
validation must reproduce these: compute the path with native `tree_path(tree, i)`,
then raise through a **thin R `cli_abort` helper on the error path only** (errors
are rare, so the callback is free, and it preserves both the `{path}` context and
cli styling — a plain `Rcpp::stop` would lose the formatting). Same reasoning that
keeps `map_tree`/`pmap_tree`'s leaf-error context in the R layer.

### C.5 What gets deleted / simplified
- anvl: the XLA R fallback branch of `jit_xla_impl` **and its `xlamisc::LRUCache`**;
  `jit_quickr_impl`'s R cache; `jit_call_xla`; R `jit_key_leaves`; the
  `jit_prepare_call` entry helper. `jit_with_backend` stops creating an R cache.
- `cache_size()` collapses to `pjrt_dispatch_size()` (no more two-cache sum).
- **Kept** (still R, on the miss/compile path): `to_avals`, tracing, and
  compilation. The compile callback receives the already-flattened leaves +
  static mask + resolved device from native, so it no longer re-flattens.

### C.6 Risks
- Native R-data upload must reuse `pjrt_buffer`/`pjrt_scalar` semantics exactly
  (column-major, dtype coercion) — divergence corrupts inputs. Cover with
  round-trip tests against the R path.
- Device-inference parity with `jit_prepare_args` (first-device / multi-device
  error / default / `device_arg`).
- quickr execute via `Rcpp::Function` must match `jit_quickr_impl`'s `r_args_flat`
  construction (`AnvlArray` → `as_array`).
- The compile-callback contract now returns an execute-variant (PJRT exec vs R
  closure); keep it small and explicit.

## Testing

Part A (pjrt):
- **Static key self-test:** a self-test entry that can construct a static
  `KeyLeaf` (extend `build_key_from_leaves` to accept a static marker, or add
  `impl_dispatch_static_key_eq`). Assert: same value → equal; different value →
  not equal; two closures differing **only** in environment → **not** equal
  (proves `flags = 40` respects environments).
- **End-to-end:** `impl_dispatch_create(cap, cb, static = "flag")`, run
  `list(x = arr, flag = TRUE)` vs `flag = FALSE` → 2 misses; repeat → hit; assert
  the static leaf never appears in execution inputs (e.g. the miss callback sees
  it but the executable is fed only the dynamic buffer).

Part A (anvl):
- Existing static tests keep passing (`cache_size()` sums both caches).
- Add: a static jit now grows the **native** cache (`pjrt_dispatch_size`), not the
  R cache.

Part B:
- **Move the tree tests to pjrt.** `anvl/tests/testthat/test-flatten.R` becomes
  pjrt tests of the native module (round-trip, names, NULL, `tree_size`,
  `tree_path`, `filter_by_names`, `reindex`, `mask_from_names`, `map_tree`,
  `pmap_tree`, `tree_diff`). Keep a thin anvl test that the re-exports resolve.
- **Conformance / round-trip:** property test over random nested structures
  (lists, names, NULLs, atomics, classed leaves) asserting
  `unflatten(build_tree(x), flatten(x))` reproduces `x`, and that
  `mask_from_names` matches the old `flat_mask_from_names`/`mark_some` output on
  the same inputs (guards the marking-consolidation).
- **Autodiff regression:** the existing `reverse.R` / gradient tests must pass
  after the rewire (they exercise `filter_by_names`/`tree_concat`/masks indirectly).

Part C:
- **Coverage parity:** the full existing anvl jit test-suite (xla + quickr) must
  pass with the R fallback/caches removed — every case now routed natively.
- **Upload round-trip:** native R-literal/array upload produces buffers
  byte-identical to `pjrt_scalar`/`pjrt_buffer` (column-major, each dtype).
- **Device moves:** `jit(f, device=…)` and `device_arg` route natively and match
  the old R path's results; multi-device inputs still error with the same message.
- **Error-message parity:** the "unexpected device" and "autoconvert" errors keep
  their exact `{path}` context and cli formatting after the native migration.
- **quickr via pjrt_dispatcher:** a quickr jit populates the native cache
  (`pjrt_dispatch_size`), recompiles on signature change, hits on repeat.
- **`cache_size()`** now equals `pjrt_dispatch_size()` alone.

## Fallback boundary — shrinks per phase, gone after C

The SENTINEL/R-fallback set contracts as the arc progresses; correctness is never
worse than the R path at any phase:

- **After A:** native handles all-xla-array + static calls (`device = NULL`). Still
  falls back for R-literal/array leaves (`x + 1`), quickr, multi-device,
  `device`/`device_arg` moves, all-static/zero-dynamic calls.
- **After C:** the pjrt_dispatcher covers all of the above; there is **no R fallback**.
  The only non-dispatch paths are the `currently_tracing()` pass-through (nested
  jit) and hard errors on genuinely invalid inputs.

## Incidental fixes already applied on this branch

- Added `dispatch.R` to `pjrt/DESCRIPTION`'s `Collate:` field (was missing;
  `R CMD INSTALL`/`build` was broken).

## Consumers / migration impact

All tree consumers are within anvl; no sibling-package callers. `R/flatten.R` is
deleted and its public API re-exported from pjrt, so the anvl-facing names are
preserved. Rewire sites:

- `reverse.R:22,24,29,135,171,179,514,515` — autodiff (marking, filter, reindex,
  child sizes, masks). Heaviest rewire.
- `stablehlo.R:138` — `flat_mask_from_names` → `mask_from_names`.
- `jit.R:316,317,465` + `backend-xla.R:137,357,358` — R fallback prep + output
  unflatten; drop `MarkedArgs`, key the fallback cache via `tree_repr`.
- `graph.R:625,626,826` + `graph-to-quickr.R:43,183` + `primitives.R:2477,2574,2580`
  — build/flatten/unflatten over opaque Node xptrs.
- Delete `mark_some`/`MarkedArgs`/`MarkedListNode` and the R `Node` constructors
  (`LeafNode`/`ListNode`/`NullNode`); `flatten_fun` moves to pjrt (it wraps
  build_tree/unflatten).

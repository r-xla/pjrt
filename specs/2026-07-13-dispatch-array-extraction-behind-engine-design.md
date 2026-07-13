# Dispatch: move array-leaf extraction behind the engine

Date: 2026-07-13
Status: proposed

## Problem

pjrt's native dispatcher reads an `AnvlArray` input's fields directly, as a plain
named list, in `anvl_fields()` (`src/dispatch_engine.cpp`): `$data`, `$dtype`,
`$shape`, `$device`, `$ambiguous`, `$backend`. anvl's `AnvlBackend` contract
(`anvl/R/backend.R`) guarantees no such thing: an `AnvlArray` is only promised a
`$data` field. Every other property is contract-accessible *only* through the
backend's accessor functions — `dtype()`, `shape()`, `ambiguous()`, `device()`,
`backend()`. anvl itself always goes through those accessors.

So the dispatcher silently depends on a **stricter layout than anvl promises**.
It works today only because the `xla`/`quickr` backends happen to store
everything as fields — and the `plain` backend already breaks the pattern: it
stores no `$device` field and computes it in its `device()` accessor. A
conforming backend with any computed/lazy accessor would break native dispatch.

Two aggravating facts:

1. **The read is monolithic and eager.** `anvl_fields()` pulls all six fields for
   every leaf, regardless of which engine consumes them.
2. **The xla route doesn't even use `$dtype`/`$shape`.** `PjrtEngine::array_aval`
   overrides the generic reader and takes dtype/shape from the `PJRTBuffer`
   (authoritative; can't be falsified by a drifted field). So on the xla route
   those two field reads are pure waste — and they are exactly the reads with no
   contract backing.

## Goal

Push "how to read a leaf's metadata" behind the engine, and read only what each
engine actually uses. Concretely:

- The xla (native) path stops reading `$dtype`/`$shape` entirely.
- The generic (non-xla) path obtains metadata through anvl's guaranteed
  accessors instead of assuming a field layout.
- The dispatcher core keeps owning input *policy* (validation, device rules,
  key building); only *extraction* moves.

## Design

### One interface, two implementations

Replace `anvl_fields()` (shared, eager, all six fields) and the separate
`Engine::array_aval()` step with a single per-engine reader that returns exactly
what the core needs:

```cpp
struct ArrayLeaf {        // what the core needs from one array input
  aval av;                // dtype + shape + ambiguous
  SEXP data;              // $data — the exec input / buffer (the guaranteed field)
  SEXP device;            // the R device object (the key's token source)
  std::string backend;    // the leaf's backend tag (for policy + error messages)
};

// nullopt when `leaf` is not an AnvlArray, so the core falls through to
// classify_rdata() exactly as today.
virtual std::optional<ArrayLeaf> read_array(SEXP leaf) const = 0;
```

`read_array` folds in what `array_aval()` used to do; the standalone
`array_aval()` virtual and the `AnvlFields` struct go away.

**`PjrtEngine::read_array`** (native, xla): reads `$data` as the `PJRTBuffer`,
takes dtype/shape from the buffer, reads `$ambiguous` for the aval bit, `$device`
for the token, `$backend` for the tag. All C++, no R dispatch. **Never reads
`$dtype`/`$shape`.** `$data` is contract-guaranteed; the residual field reads
(`$ambiguous`/`$device`/`$backend`) are the co-developed native pairing of pjrt's
own engine with its own backend, not a generic assumption. Preserves the
existing "xla array must hold a PJRTBuffer in `$data`" and dtype-representable
errors.

**`ClosureEngine::read_array`** (generic): reads `$data` (guaranteed), then calls
an **anvl-supplied extractor closure** on the leaf to obtain
`{aval, device, backend}`, built on anvl's accessor generics. Contract-clean for
*any* backend. The per-leaf R cost lands only on the already-R-heavy closure
path (which runs an R closure per execute anyway).

The extractor returns the same record the generic path builds today — dtype as a
tengen `DataType` object, integer shape, logical ambiguous, the device object, a
backend string — so the C++ conversion (`anvl_dtype_from_tengen`,
`check_dtype_representable`) is unchanged; only the *source* changes from `$`
fields to accessors.

### The core keeps the policy

The `impl_dispatch_run` classification loop is unchanged in responsibility: it
still owns plain-reject, backend-match, and the device-conflict/canonicalization
rules — now applied to the `ArrayLeaf` the engine returns rather than to a
hardcoded field parse. Recognizing bare R data (`classify_rdata`) and static
leaves stays exactly as-is.

### Configuring the extractor

`ClosureEngine` needs the extractor closure; `PjrtEngine` does not. Thread it
through construction:

- pjrt: `impl_dispatch_create(...)` gains an `extractor_fn` parameter (an R
  closure, or `NULL`). `make_engine()` forwards it; `ClosureEngine` stores it,
  `PjrtEngine` ignores it.
- R: `dispatcher()` gains an `extractor` parameter, required for non-xla
  backends, unused for `"xla"`.
- anvl: `jit_quickr_impl` passes
  `extractor = function(leaf) list(aval = list(dtype = dtype(leaf), shape =
  shape(leaf), ambiguous = ambiguous(leaf)), device = device(leaf), backend =
  backend(leaf))` (built on the backend's accessors). The xla path
  (`jit_xla_impl`) passes none.

## What each route depends on after the change

- **xla:** `$data` (guaranteed) + `$ambiguous`, `$device`, `$backend`. The
  unused, unguaranteed `$dtype`/`$shape` reads are gone.
- **non-xla:** `$data` (guaranteed) + one anvl extractor closure that honors the
  `AnvlBackend` contract. No direct field assumptions at all.

## Decisions (confirmed)

- **The xla/non-xla asymmetry is intended.** xla reads its non-`$data` fields in
  C++ for speed; non-xla goes through the extractor for contract-correctness.
  Routing xla through an R extractor would defeat the native fast path.
- **`$data` stays a direct core read on both paths**, since it is the one field
  the contract guarantees — not routed through the extractor.

## Non-goals

- No change to the cache key, the LRU, the compile-callback contract, or output
  wrapping.
- No change to how bare R literals/arrays (`kRData`) or static leaves are
  classified.
- Not making the xla path go through R accessors (deliberately kept native).

## Testing

- **C++ (`src/test-dispatch.cpp`)** already fabricates leaves and asserts the
  aval-keyed behavior; extend it so `PjrtEngine`/`ClosureEngine` `read_array` is
  exercised where the cache key is built.
- **R (`tests/testthat/test-dispatch.R`)** — preserve the existing behaviors,
  especially:
  - "an xla leaf's aval comes from its buffer, not its `$dtype`/`$shape`" (the
    lying-array test) must still pass, now because the xla reader never consults
    those fields.
  - "the closure engine serves a backend pjrt has never heard of" must still
    pass, now driven through the supplied extractor.
  - input-validation error messages (plain-reject, cross-backend, naming the
    argument) must be preserved verbatim.
- **Contract test (new):** a closure-backend array whose non-`$data` metadata is
  *computed by accessors* (no `$dtype`/`$device`/... fields present) must
  dispatch correctly — the property the current code silently violates. This is
  the regression guard for the whole change.
- Full anvl integration suite (xla + quickr through `jit()`) stays green.

## Files touched

- `src/dispatch_engine.h` — `ArrayLeaf`, `read_array` virtual; drop `AnvlFields`
  and `array_aval`.
- `src/dispatch_engine.cpp` — `PjrtEngine::read_array` (native),
  `ClosureEngine::read_array` (extractor); `make_engine` gains the extractor;
  remove `anvl_fields`.
- `src/dispatch.cpp` — classification loop calls `engine.read_array`;
  `impl_dispatch_create` gains `extractor_fn`.
- `R/dispatch.R` — `dispatcher()` gains `extractor`; regenerate RcppExports.
- `anvl/R/backend-quickr.R` — pass the accessor-based extractor;
  `anvl/R/backend-xla.R` — unchanged (no extractor).
- Tests as above.

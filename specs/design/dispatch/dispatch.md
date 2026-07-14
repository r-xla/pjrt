# Native Dispatch

A `Dispatcher` is the fast path behind anvl's `jit()`: it repeatedly executes a
compiled program on inputs of the same signature. It owns an executable cache
keyed on the inputs' structure and abstract values, and calls back into R to
compile only on a cache miss. Everything on a cache hit — flatten, key, look up,
assemble inputs, execute, wrap the outputs — happens in C++, without an R call.

The R API is `dispatcher()`, `dispatch()` and `dispatcher_size()`; see
`?dispatcher` for the callback contract and the parameters. This document
describes how it works underneath.

## The pieces

| File | Holds |
| --- | --- |
| `dispatch.h` / `dispatch.cpp` | the backend-agnostic core: the `Dispatcher`, flatten/classify/key, the compile protocol, the Rcpp entry points |
| `dispatch_key.h` | the cache key: `AnvlDtype`, `Aval`, `KeyLeaf`, `CacheKey`, and its hash + equality |
| `dispatch_engine.h` / `.cpp` | the `Engine` interface and its two implementations, `PjrtEngine` and `ClosureEngine` |
| `lru_cache.h` | the generic LRU container the cache is built on |
| `hash.h` / `hash.cpp` | `hash_combine` plus the R-value folds the key hashes with |
| `tree.h` / `tree.cpp` | the `RTree` (shared with the exposed Rtree API), used to flatten a call and as key material |

## A call

`impl_dispatch_run(dispatcher, args)`:

1. **Flatten.** `flatten_rec()` turns the argument list into an `RTree` plus a
   flat leaf vector. `static_leaf_mask()` overlays a per-leaf static bit: an
   argument named in `static` marks every leaf of its subtree.
2. **Classify.** Each leaf becomes a `KeyLeaf` and, unless static, an
   `ExecInput`. Every rejection happens here, naming the offending argument by
   its path in the argument tree (`leaf_subject()`), so the compile callback is
   only ever asked to compile inputs already known to be executable — never to
   validate them. There is no fallback path for the caller.
3. **Resolve the device** (see *Devices*).
4. **Probe the cache.** On a miss, the R `compile` callback is handed the
   material this call already derived — the tree, the leaves, the static mask,
   the avals, the resolved device — rather than the bare `args` to classify a
   second time. The avals it receives are the ones the key was built from, so
   the program it compiles cannot disagree with the key it is filed under. The
   engine validates the callback's result and builds the entry.
5. **Run.** `Engine::run()` executes the call and returns the finished value.

## The cache key

`CacheKey` is the input tree, one `KeyLeaf` per leaf, and a device token. It has
no backend component: a dispatcher accepts arrays of exactly one backend and
owns its own cache, so no two keys of one cache could differ in it.

A leaf is keyed one of two ways (`keyed_by_value()`):

- **By its `Aval`** (dtype, shape, `ambiguous`) — an `AnvlArray` of the
  dispatcher's backend (`kArray`), or a bare R literal/array (`kRData`). These
  two kinds are deliberately *not* distinguished by the key: they differ only in
  where execution finds the input (the leaf's own `$data`, or a fresh upload),
  which is settled per call from the call's own leaves. The program compiled for
  an `Aval` is the same either way, so keying them apart would compile it twice —
  `f(x, y)` and `f(x, 1)` with matching avals share one executable. `kind`
  survives only to steer input assembly.
- **By its value** — a static argument (`kStatic`), compared with `identical()`
  and baked into the executable as a constant.

`r_identical()` tightens R's `identical()` with two flags. `IDENT_USE_CLOENV`
compares closure environments, so two distinct closures with the same body do not
merge. `IDENT_NUM_AS_BITS` compares numbers bitwise: R's default merges `+0.0`
with `-0.0`, and `bit64` stores `NA_integer64_` as the bit pattern of `-0.0`, so
without it a static `NA_integer64_` and a static `0` would share a cache entry
and silently run each other's executable. The cost is that `+0.0` and `-0.0` now
compile separate (identical) entries — a finer key can waste a compile, never
return the wrong program.

The hash folds the tree, the device token, and each leaf. Value-keyed leaves fold
their contents (`hash_atomic()`; closures fold their formals and body via
`hash_closure()`) purely to keep `identical()` off the common path. A fold may
only collide, never split, what the equality joins — that is the rule every fold
here is written to.

## Engines

The core is backend-agnostic. Everything a backend has an opinion about — how an
`Aval` is read off one of its arrays, what a cache entry holds, how a call
executes, how its outputs are wrapped — sits behind `Engine`. The core never
branches on the engine; it calls the hooks.

**`PjrtEngine`** (`backend = "xla"`) is the native fast path. `read_array()`
takes dtype, shape and device off the `PJRTBuffer` in `$data`, which caches them
natively and so cannot drift from the array's R-level fields (this engine never
consults them). `run()` assembles the executable's inputs — const arrays, the
call's inputs, freshly allocated phantom donation buffers — executes, and wraps
each output buffer natively.

Output wrapping is template-driven: everything about an output except its buffer
is fixed per cache entry, so `build_entry()` builds one template `AnvlArray` per
output from the avals the callback declared, and a call only shallow-copies a
template per output and drops the buffer into its `$data` slot, then re-nests via
the entry's `out_tree`. `dispatch()` therefore returns the call's *finished*
result, and the array-wrapper layout has a single source of truth: pjrt both
reads it and writes it.

**`ClosureEngine`** (any other backend) is the generic vehicle for a backend pjrt
knows nothing about (anvl's `"quickr"`). The entry holds a compiled R closure;
`run()` calls it with the flat list of dynamic leaves and returns its value
verbatim. Execution, output wrapping and input placement are all the backend's
business.

A future native backend is a third `Engine` subclass; a new R-level backend needs
no C++ at all.

## The array contract

anvl's `AnvlBackend` contract guarantees an `AnvlArray` only a `$data` field;
every other property is reachable only through the backend's accessors, and a
conforming backend may compute them lazily. So `$data` is the one field the core
reads directly. Everything else comes from the engine: `PjrtEngine` off the
buffer, `ClosureEngine` through the `extractor` closure the backend supplies
(built on its own accessors).

The asymmetry is intended: routing the xla path through an R extractor would
defeat the native fast path, and pjrt's engine is co-developed with pjrt's own
backend.

## Devices

The key carries a `DeviceToken` — the address of a *canonical* device object.
It is never dereferenced, only compared and folded, so one `const void*`
identifies a device of any backend: no `identical()`, no variant type, no
per-backend branch in the hash or the equality.

`Engine::canonical_device()` maps a device object to its representative by object
identity first (free for a backend that interns its devices) and `r_identical()`
as a fallback, so a backend handing out equal-but-distinct device objects
collapses to one token rather than splitting the cache. Canonical objects are
preserved for the dispatcher's lifetime, which is what keeps a token's address
stable. `PjrtEngine` overrides the hook to intern by the underlying
`PJRT_Device*`, so a buffer-sourced device and a resolver-sourced one collapse to
the same token — letting `f(x)` and `f(1)` share an entry.

Two device policies:

- **`move_inputs = FALSE`** (default): the first array's device is the call's
  device, every later array must agree, and the device is part of the key. A call
  with no array input resolves the backend's *current* default through the
  `default_device` callback — resolved afresh per call, and part of the key, so an
  entry compiled under one default is never served after the default changes.
- **`move_inputs = TRUE`**: each entry has a target device (the one its `compile`
  call returned) and the engine places every input on it at execute time. The key
  then carries no device, so inputs may arrive from any device.

`move_inputs` is a policy flag, not a device: *which* device an entry targets is
the callback's business and need not be the same for every entry (anvl's
`jit(device = )` fixes one up front; its `device_arg()` derives one per static
value). That is why the target lives per entry rather than on the dispatcher.

Placing an input is the engine's own business, since only it knows what `$data`
holds: `PjrtEngine` copies a buffer that lives elsewhere; `ClosureEngine` does
nothing, so the backend's `r_fun` must place its own inputs, idempotently.

## Memory

Every R object the dispatcher holds — the callbacks, the engines' entry material,
the key's static values — is held in an Rcpp type (`XPtr`, `List`, `Function`,
`RObject`). It is rooted for as long as the dispatcher or the entry lives and
dropped when that ends, so there is no preserve/release bookkeeping to get wrong:
an entry abandoned mid-construction (because the compile callback threw) releases
exactly what it had taken, and a cached one is released by the LRU destroying it.

## Testing

- `tests/testthat/test-dispatch.R` drives the dispatcher directly (both engines,
  key behaviour, every validation error) and, where anvl is installed (it is in
  `Suggests`), end-to-end through `anvl::jit()`.
- `src/test-dispatch.cpp` unit-tests the cache key in C++ via testthat's Catch
  integration. The key is where a mistake does not error but silently returns
  someone else's executable, and C++ tests can fabricate a device token or a
  `kRData` leaf directly — neither of which an R fixture over `PJRTBuffer`s can
  express. This is why the key lives in its own header.
- The Catch tests are compiled out unless `PJRT_BUILD_CPP_TESTS=1` is set at
  build time (see `configure`); `tests/testthat/test-cpp.R` skips when they are.
  Changing the flag needs a `--preclean` install, or already-compiled objects are
  reused and the flag has no effect.

## Consumer

anvl's `jit()` is the only consumer: `R/backend-xla.R` creates a dispatcher with
the default `"xla"` backend, and `R/backend-quickr.R` one with
`backend = "quickr"` plus an accessor-based `extractor`. Both then call
`pjrt::dispatch()` on the evaluated argument list and return its result directly.

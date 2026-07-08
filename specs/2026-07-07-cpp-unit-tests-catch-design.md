# Native C++ unit tests via testthat + Catch

## Goal

Add real in-C++ unit tests for pjrt's header-accessible C++ internals, using
testthat's Catch integration. Retire the two `impl_dispatch_*_selftest` R
exports that exist only to reach C++ logic from R and that Catch supersedes.

## Scope

**In:** the `RTree` operations in `tree.h`, and the `LRUCache` container
(extracted from `dispatch.cpp` into its own header).

**Out:** `dispatch.cpp`'s cache-key internals (`CacheKey`/`aval`/`KeyLeaf` and
the hashers) stay put. The `key_hash`/`key_eq`/`static_key_eq` self-tests are
kept R-side because they read real `PJRTBuffer`s and use R's `identical()`,
which is awkward to reconstruct in pure C++. `tree_hash` stays dispatch-local
and remains covered through those retained self-tests.

## Changes

1. **Harness** — `use_catch()` scaffolds `src/test-runner.cpp`, adds `testthat`
   to `LinkingTo`, and registers `run_testthat_tests`. A one-line
   `tests/testthat/test-cpp.R` runs the C++ suite inside `devtools::test()`.
2. **Extract** the `LRUCache` template verbatim into `src/lru_cache.h`;
   `dispatch.cpp` includes it. It is a self-contained generic container, so the
   extraction is clean and now has a second consumer (the tests).
3. **`src/test-tree.cpp`** — flatten/unflatten round-trips (leaf, nested, named,
   `NULL`/NullNode, `list()` vs named-empty), `tree_eq` (structure / names /
   `has_names` distinctions), `tree_size_rec`.
4. **`src/test-lru.cpp`** — recency ordering, capacity eviction, `on_evict`
   firing, `get`/`set` update semantics, `clear()`.
5. **Retire** `impl_dispatch_node_selftest` and `impl_dispatch_lru_selftest`
   plus their R-side tests; regenerate `RcppExports`/`NAMESPACE`.

## Verification

`devtools::test()` runs both the R tests and the Catch suite; the Catch tests
must appear in the output and pass. The protobuf double-registration caveat
does not apply: the Catch tests run inside a single `test()` process, not a
`load_all()` + `test()` mix.

## Risks

- `LinkingTo: testthat` adds a build-time header dependency (testthat is
  already a test dependency) and Catch lengthens compile time (one-time cost in
  `test-runner.cpp`).
- `use_catch()` file edits are inspected after it runs; if it conflicts with
  the package's routine registration, the runner + registration are set up by
  hand instead.

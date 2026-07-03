# Native static dispatch (Part A) + pjrt-owned pytree (Part B) — Implementation

Date: 2026-07-03
Derived from: `2026-07-03-native-static-dispatch-partA-plan.md` (Part A, followed
verbatim) and `2026-07-03-native-static-dispatch-and-flatten-exposure-design.md`
(Part B, made concrete here). Part C is out of scope.

Repos: `pjrt` (branch `feat/native-dispatch`), `anvl` (branch
`perf/dispatch-overhead`).

## Part A — Native static-arg dispatch (pjrt + anvl)

Implemented exactly per the Part A plan, tasks 1–4:

1. **`identical()` flags fix + self-test.** `CacheKeyEq`'s static branch uses
   `R_compute_identical(x, y, /*flags=*/40)` (`IGNORE_BYTECODE | IGNORE_SRCREF`,
   `ignore.environment = FALSE` — R's default `identical()`). New R-callable
   self-test `impl_dispatch_static_key_eq(a, b)` proves value- and
   environment-sensitivity.
2. **Plumbing.** `impl_dispatch_create(capacity, miss_fn, static_names)`;
   `pjrt_dispatcher(capacity, compile, static = character())`;
   `Dispatcher::static_names()` (an `std::unordered_set<std::string>`);
   `CacheEntry::static_key_values` released in `release_entry`.
3. **Static marking + run handling.** `flatten_rec` gains
   `std::vector<char>& is_static` + `bool inherited_static`; `impl_dispatch_run`
   builds the top-level `ListNode` itself, marks children whose name is in
   `static_names()`, keys static leaves by value, excludes them from execution,
   takes the device from the first *dynamic* buffer, and returns the sentinel
   when there is no dynamic buffer (all-static / zero-arg calls keep the R
   fallback). Static key values are `R_PreserveObject`d into the entry on a
   miss.
4. **anvl wiring.** `jit_xla_impl` passes `static = static` to
   `pjrt::pjrt_dispatcher()`. New test: a static xla jit populates the native
   cache (`pjrt_dispatch_size`), not the R cache.

## Part B — The pytree module moves to pjrt; `Node` is opaque

### B.1 pjrt C++: `src/tree.h` + `src/tree.cpp`

`src/tree.h` (namespace `rpjrt`): the `Node` struct, `is_bare_list`,
`flatten_rec` (with the Part-A `is_static`/`inherited_static` parameters),
`unflatten_rec`, `node_hash`, `node_eq` move here from `dispatch.cpp` —
**one** traversal encodes the flatten semantics for both the dispatch hot path
(stack `Node`, no allocation) and the exposed API (heap `Node` behind an
`Rcpp::XPtr`, S3 class `"PJRTNode"`).

`src/tree.cpp` exposes (all `// [[Rcpp::export]]`, `impl_tree_*`):

| impl | R wrapper | returns |
|---|---|---|
| `impl_tree_build(x)` | `build_tree(x)` | `PJRTNode` xptr |
| `impl_tree_flatten(x)` | `flatten(x)` | list of leaves |
| `impl_tree_unflatten(node, x)` | `unflatten(node, x)` | rebuilt object |
| `impl_tree_size(node)` | `tree_size(node)` | int (leaf count) |
| `impl_tree_equal(a, b)` | `tree_equal(a, b)` | lgl (structural) |
| `impl_tree_kind(node)` | `tree_kind(node)` | `"leaf" \| "list" \| "null"` |
| `impl_tree_names(node)` | `tree_names(node)` | top-level child names or `NULL` |
| `impl_tree_child_kinds(node)` | `child_kinds(node)` | chr of kinds (root must be a list) |
| `impl_tree_child_sizes(node)` | `child_sizes(node)` | int per top-level child |
| `impl_tree_flat_names(node)` | `flat_names(node)` | per-leaf top-level name |
| `impl_tree_path(node, i)` | `tree_path(node, i)` | `"a$b"` / `"[[2]]"` path |
| `impl_tree_filter_by_names(node, names)` | `filter_by_names(node, names)` | new owned `PJRTNode`, reindexed |
| `impl_tree_concat(nodes, names)` | `tree_concat(nodes, names)` | new owned parent `PJRTNode`, leaves renumbered contiguously |
| `impl_tree_mask_from_names(node, names)` | `mask_from_names(node, names)` | lgl per leaf: leaf under a top-level child whose name is in `names` |
| `impl_tree_repr(node)` | `tree_repr(node)` | canonical string, e.g. `"list(a = *, b = list(*, NULL))"` |
| `impl_tree_diff(a, b)` | `tree_diff(a, b)` | `NULL` or `list(prefix, a, b)` with `a`/`b` the **repr strings** of the diverging subtrees |

Decisions:

- **No sub-node handles escape to R** (design B.1): `filter_by_names` /
  `tree_concat` copy into fully-owned result Nodes; `tree_diff` returns repr
  strings, not Nodes (its only consumer formats them into an error message).
- **`mask_from_names` is strict:** empty `names` → all-`FALSE`; the "`NULL`
  means everything" convention of the old `flat_mask_from_names` is handled at
  the (single) call site that wants it.
- `tree_repr` reproduces anvl's old `format.ListNode` exactly (`*`, `NULL`,
  `list(...)`, `name = ` only for non-empty names) — it also serves as the
  R-fallback cache-key material (an xptr in a `hashtab` key would compare by
  address and never hit).
- `print.PJRTNode` / `format.PJRTNode` delegate to `tree_repr` (deterministic,
  snapshot-friendly).

### B.2 pjrt R layer: `R/tree.R`

Thin wrappers over the impls (with checkmate input checks where cheap), plus
the leaf-function orchestration that stays in R by design:

- `map_tree(.x, .f, ...)`, `pmap_tree(.l, .f, ...)` — ported from anvl
  verbatim, but using native `tree_equal` for the structure check and native
  `tree_diff` for the mismatch message. cli-styled leaf-error context via
  `tree_path` is preserved.
- `flatten_fun(f, ..., in_node = NULL)` — ported from anvl (wraps
  `build_tree`/`unflatten`).

`DESCRIPTION` gains `Collate: 'tree.R'`; docs via roxygen; NAMESPACE via
`devtools::document()`.

### B.3 pjrt tests

`tests/testthat/test-tree.R`: anvl's `test-flatten.R` moved and adapted
(round-trip, NULL semantics, structural distinctness, `map_tree`, `pmap_tree`,
formatting, `tree_diff`, `tree_path`, `flatten_fun`) with snapshots
regenerated in pjrt, plus new coverage for `tree_equal`, `tree_kind`,
`tree_names`, `child_kinds`, `child_sizes`, `flat_names`, `filter_by_names`,
`tree_concat`, `mask_from_names`, `tree_repr`, and a seeded property test:
random nested structures (lists/names/NULLs/atomics/classed leaves) must
round-trip `unflatten(build_tree(x), flatten(x)) == x`, with
`mask_from_names` matching an R-side reference implementation.

### B.4 anvl rewires

- **Delete `R/flatten.R`** (incl. `mark_some`/`MarkedArgs`/`MarkedListNode`,
  R `Node` constructors, `new_counter`, `reindex_tree`, `filter_list_node`,
  `flat_mask_from_names`, R `flatten_fun`, `tree_diff*`). Delete
  `tests/testthat/test-flatten.R` + its snaps (moved to pjrt); add a thin
  re-export test.
- **`R/reexports.R`**: re-export from pjrt: `flatten`, `build_tree`,
  `unflatten`, `tree_size`, `tree_path`, `map_tree`, `pmap_tree`, `tree_diff`
  (the previously-public anvl names). Internal-only helpers are called with
  `pjrt::` prefix.
- **`R/jit.R`** `jit_prepare_args`: drop the flat fast path and the MarkedArgs
  branch — `in_tree <- pjrt::build_tree(args)`, `args_flat <- pjrt::flatten(args)`,
  `is_static_flat <- pjrt::mask_from_names(in_tree, static)`.
- **`R/backend-xla.R`** + **`R/backend-quickr.R`**: the R fallback cache keys
  replace the `in_tree` component with `pjrt::tree_repr(in_tree)` (a fresh
  xptr per call would never `identical()`-hit in the hashtab).
- **`R/reverse.R`**: `prepare_gradient_args` uses `build_tree` +
  `mask_from_names` (all-`TRUE` when `wrt` is NULL); `filter_list_node` →
  `pjrt::filter_by_names`; `compute_requirements` uses `mask_from_names`
  (all-`TRUE` when `wrt` empty) and `pjrt::flat_names` for the error message;
  the value/grad combine uses `pjrt::tree_concat`.
- **`R/stablehlo.R:138`**: `flat_mask_from_names` → `pjrt::mask_from_names`
  (call site already guarded by `length(donate) > 0`).
- **`R/graph-to-quickr.R`** / **`R/rules-quickr.R`**: `inherits(x, "LeafNode")`
  → `pjrt::tree_kind(x) == "leaf"`; the nested-list probe uses
  `child_kinds`; top-level formal names use `tree_names` / `child_sizes`.
- **`R/graph.R`**: `flatten_fun` now comes from pjrt.
- `DESCRIPTION` Collate drops `flatten.R`; `devtools::document()` regenerates
  NAMESPACE/Rd (old flatten Rd files deleted).

### B.5 Semantics preserved (the contract the tests pin down)

- bare list (VECSXP without class) recurses; classed object/atomic/function is
  a leaf; `NULL` is a NullNode: part of the tree (and cache key), no leaf, no
  index.
- `unflatten` restores names exactly (including `""` slots); a leaf root
  returns the single value; a null root returns `NULL`.
- `tree_path`: named → `a$b$c`, unnamed → `[[j]]`, root leaf → `""`.
- `filter_by_names` keeps top-level children by name, reindexes leaves
  contiguously, errors if the tree has no names, returns the input node when
  all children are kept.

## Verification

- pjrt: `devtools::document()`; full `testthat` suite green.
- anvl: `R CMD INSTALL` pjrt first, then full `testthat` suite green
  (autodiff, jit xla/quickr, graph — they exercise every rewired site).
- `make format` + `jarl check` in both repos before the final commits.

# Dispatcher

A native fast path for repeatedly executing a compiled program on inputs
of the same signature, intended to be used in anvl. It owns an
executable cache keyed on the inputs' structure and abstract values,
calling back into R to compile only on a cache miss.

`dispatcher()` creates a dispatcher for one program family;
[`dispatch()`](https://r-xla.github.io/pjrt/dev/reference/dispatch.md)
runs a call through it and returns the call's result.

## Usage

``` r
dispatcher(
  capacity,
  compile,
  static = character(),
  backend = "pjrt",
  move_inputs = FALSE,
  default_device = NULL,
  extractor = NULL
)
```

## Arguments

- capacity:

  (`integer(1)`)  
  Maximum number of compiled executables to cache.

- compile:

  (`function`)  
  Cache-miss callback, called as `compile(info)`. `info` carries what
  the dispatch already derived from the call, so that the callback need
  not classify the inputs a second time:

  - `args`: the dispatched argument list,

  - `in_tree`: its `RTree` (see
    [`build_tree`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md)),

  - `leaves`: the flat leaf list (see
    [`flatten`](https://r-xla.github.io/pjrt/dev/reference/flatten.md)),

  - `is_static`: a [`logical()`](https://rdrr.io/r/base/logical.html)
    mask over `leaves`,

  - `avals`: per leaf, `NULL` if static, else the
    `list(kind, dtype, shape)` the cache key was built from. `kind` is
    `"array"` for one of the backend's arrays and `"rdata"` for a bare R
    literal or array – the two are different cache keys, because bare R
    data has no dtype of its own and the caller may compile it into a
    different program. `dtype` is a canonical dtype string (`"f32"`,
    `"i64"`, ...) for an `"array"` leaf, and for an `"rdata"` leaf its R
    storage type instead – its
    [`typeof()`](https://rdrr.io/r/base/typeof.html), so `"double"`,
    `"integer"` or `"logical"`. That is what the value *is*, not a dtype
    it is not yet: `"double"` is not `"f64"`, and what the leaf is
    uploaded at is `input_dtypes`. `shape` is an
    [`integer()`](https://rdrr.io/r/base/integer.html), empty for a
    scalar,

  - `default_device`: the device this call resolved because no array
    input named one – the device the cache key was built on, so
    `compile` must compile for it rather than resolve a default of its
    own. `NULL` when an array named the device, or under `move_inputs`.

  For `backend = "pjrt"` it must return a named list with:

  - `exec`: a
    [`pjrt_compile`](https://r-xla.github.io/pjrt/dev/reference/pjrt_compile.md)d
    executable,

  - `client`, `device`: the
    [`pjrt_client`](https://r-xla.github.io/pjrt/dev/reference/pjrt_client.md)
    and the device the entry is compiled for,

  - `out_tree`: the `RTree` of the outputs (see
    [`build_tree`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md)),

  - `out_avals`: one aval per output leaf of `out_tree`, each a
    `list(dtype = <string>, shape = <integer>)`. The outputs are wrapped
    from these.

  - `const_arrays` (optional): buffers prepended to the inputs,

  - `phantom_specs` (optional): a list of
    `list(dtype = <string>, shape = <integer>)` donation-output buffers
    to allocate fresh per call.

  Either kind of result may additionally carry:

  - `input_dtypes`: a
    [`character()`](https://rdrr.io/r/base/character.html) with one
    entry per dynamic leaf, in order, naming the dtype that input is
    supplied at. `NA` leaves an input alone, and is the only valid entry
    for an array input: an array is supplied as it is, so declaring a
    dtype for one is an error rather than a no-op.

    With `backend = "pjrt"` every bare R leaf must name a dtype: bare R
    data has no dtype of its own, and only the compiled program knows
    what it is used as, so the engine uploads it at the dtype declared
    here and never guesses one. It is what lets a program that consumes
    an R double as `f64` get the exact value rather than one rounded
    through `f32` first. The `"rdata"` aval's `dtype` is the leaf's R
    storage type, so it is never an answer to this: the callback names a
    real dtype. The field may be omitted only for a call whose inputs
    are all arrays.

    Not every R storage type uploads at every dtype, and a pair that
    cannot be uploaded is rejected here rather than at execute time: a
    `"double"` input takes any dtype, an `"integer"` input any but
    `"bool"`, and a `"logical"` input only `"bool"`.

    Any other `backend` uploads nothing: `r_fun` gets each R value
    itself, so no entry could take effect and every one of them must be
    `NA`. A declared dtype is rejected there rather than accepted and
    ignored, for the same reason it is at an array input. The field is
    still length-checked, and may still be omitted entirely.

  For any other `backend` it must return a named list with:

  - `r_fun`: a function called with the list of the call's dynamic
    leaves, in order and with an array leaf contributing its `$data`,
    returning the call's finished value. Static leaves are *not* passed:
    they are constants of the closure `compile` just built, and a cache
    hit already proves the call's statics are
    [`identical()`](https://rdrr.io/r/base/identical.html) to the ones
    it was built from.

- static:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of top-level arguments that are static (not arrays). Static
  values are part of the cache key and are excluded from execution.
  Defaults to none. They are compared with `identical(num.eq = FALSE)`,
  i.e. numbers compare bitwise, which is what keeps a
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  `NA` – stored as the bit pattern of `-0` – from sharing a cache entry
  with `0`.

- backend:

  (`character(1)`)  
  The `$backend` tag every `"AnvlArray"` input must carry, and the tag
  stamped on wrapped outputs. It also selects the execution engine (see
  *Backends*); anvl's quickr backend passes `"quickr"`.

- move_inputs:

  (`logical(1)`)  
  If `TRUE`, each cache entry has a target device – the `device` its
  `compile` call returned – and the engine places every input on it at
  execute time; the cache key then carries no device, so inputs may
  arrive from any device. Default `FALSE`: the first array's device is
  the call's device, and a conflicting input is an error.

  It is a policy flag, not a device: *which* device an entry targets is
  the `compile` callback's business, and need not be the same for every
  entry (anvl's `jit(device = )` fixes one up front, its `device_arg()`
  derives one per static argument value).

  Placing an input is the engine's business, since only it knows what
  `$data` holds. With `backend = "pjrt"` an input living elsewhere is
  copied to the entry's device. With any other backend pjrt does
  nothing, so **`r_fun` must place its own inputs** – it receives only
  their `$data`, not their `$device`, so the placing has to be
  idempotent.

- default_device:

  (`function` \| `NULL`)  
  Called with no arguments to get the backend's *current* default
  device, whenever a call has no array input to read a device from. Its
  result is part of the cache key, so an entry compiled under one
  default device is never served after the default changes. Required
  unless `move_inputs = TRUE`, which fixes the device per entry.

  Devices are compared by object identity first and by
  [`identical()`](https://rdrr.io/r/base/identical.html) as a fallback,
  so equal-but-distinct device objects count as one device. Interning
  them (one object per device, alive for the session, as
  [`as_pjrt_device()`](https://r-xla.github.io/pjrt/dev/reference/as_pjrt_device.md)
  does) is therefore not required, but is recommended: an interned
  device resolves in one pointer comparison, and every distinct object a
  backend hands out stays alive for the dispatcher's lifetime.

- extractor:

  (`function` \| `NULL`)  
  Reads a non-`"pjrt"` array's metadata via the backend's accessors,
  called as `extractor(leaf)` and returning
  `list(aval = list(dtype, shape), device, backend)` – `dtype` a tengen
  `DataType`, `shape` an
  [`integer()`](https://rdrr.io/r/base/integer.html). The aval's kind is
  not the extractor's to say: whatever it returns is an array leaf.
  Required for any backend other than `"pjrt"`; ignored for `"pjrt"`
  (see *Backends*).

## Value

`dispatcher()` returns a `Dispatcher`.

## Details

Each
[`dispatch()`](https://r-xla.github.io/pjrt/dev/reference/dispatch.md)
call flattens the inputs and builds a cache key: a dynamic leaf
contributes its abstract value – its kind (one of the backend's arrays,
or bare R data), its dtype and its shape – a static leaf its value
(compared with [`identical()`](https://rdrr.io/r/base/identical.html)).
On a hit the cached executable runs immediately; on a miss `compile` is
called to produce a new cache entry.

Inputs are validated before the cache is probed, and a rejection names
the offending argument by its path in the argument tree. An input must
be an `"AnvlArray"` of the dispatcher's `backend`, a length-1 atomic
scalar, or an [`is.array()`](https://rdrr.io/r/base/array.html) value.
`compile` is therefore never asked to validate: it is called only on a
cache miss, and only for inputs already known to be executable.

## Backends

The `backend` selects the execution engine, and everything
backend-specific sits behind it:

- `backend = "pjrt"` executes a compiled PJRT executable natively: array
  inputs contribute their `$data` buffer, bare R literals and arrays are
  uploaded at the dtype the entry's `input_dtypes` declared for them,
  and the outputs are wrapped back into `"AnvlArray"`s – lists of
  `$data`, `$dtype`, `$shape`, `$device` and `$backend` – and re-nested
  via `out_tree`, all without leaving C++.

- any other `backend` calls the compiled R closure `compile` returned,
  which returns the call's finished value. Execution, output wrapping
  and input placement therefore stay under the backend's control. This
  is the path for any non-PJRT backend (e.g. anvl's `"quickr"`).

Of an array input, only `$data` is ever assumed: for `"pjrt"` its dtype,
shape and device are read off the `PJRTBuffer` directly, and for any
other backend they come from `extractor`. A backend is free to store
them as fields or to compute them on demand.

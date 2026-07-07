# pjrt (development version)

## Features

* pjrt now owns the Rtree module (pjrt's R analog of JAX's pytree), which
  includes functions like `build_tree()`, `flatten()`, `unflatten()`, etc..
* `inspect_hlo()` returns the HLO intermediate representations the XLA
  compiler produces for a program -- the input (`before_optimizations`) and
  optimized (`after_optimizations`) HLO -- to help debug compilation (#194).
  Enable it by setting the dump flags in `XLA_FLAGS` (e.g.
  `--xla_dump_to=<dir> --xla_dump_hlo_as_text`) at the start of the session,
  before the first compilation; `inspect_hlo()` errors with instructions if
  they are not set.

## Bug fixes

* `check_err()` no longer leaks the underlying `PJRT_Error` when
  converting a plugin error into an R exception.
* Reading a buffer back to the host now respects the device buffer's
  actual memory layout. A non-row-major (but untiled) executable output —
  e.g. one pinned to a column-major layout via `mhlo.layout_mode` — is
  reordered correctly instead of being returned transposed. A layout the
  readback cannot faithfully reorder (strided, tiled, or rank-mismatched)
  now raises a clear error rather than silently returning wrong data.

## Features

* `pjrt_buffer()`, `pjrt_scalar()`, and `pjrt_execute()` now call R's
  garbage collector and retry once when the plugin reports
  `RESOURCE_EXHAUSTED`. Unreferenced `PJRTBuffer` external pointers are
  finalized between attempts so their device memory is released before the
  retry.
* The first time a PJRT plugin needs to be downloaded, interactive sessions
  now ask for confirmation before downloading (similar to `torch`).
  Non-interactive sessions no longer download automatically. The
  `PJRT_INSTALL` environment variable overrides this: set it to `"1"` to
  always download without asking, or `"0"` to never download.
* Added an `install_pjrt()` function which is a slight convenience wrapper
  for downloading the plugins.

## Internal

* The test suite now only runs when the `PJRT_TEST` environment variable is
  set to `"1"`, so it is skipped on CRAN (where the required PJRT plugin
  cannot be downloaded). CI sets `PJRT_TEST=1`.

# pjrt 0.4.0

## Features

* Added QR, LU, SVD, and symmetric eigendecomposition support on both
  CPU and CUDA via the FFI registration mechanism.
* Added an vignette on how to register custom calls via the FFI
  registration mechanisms with coverage of both CUDA and CPU-specific
  aspects.
* Added support for the `bit64` package to better support long integers.
* `pjrt_buffer()`, `pjrt_scalar()`, and `as_array()` gain a `check`
  argument (default `FALSE`). When `TRUE`, the call errors instead of
  silently losing information: on input if `data` contains `NA`s, on
  output if the materialized R vector contains a value that's
  indistinguishable from `NA` or that has wrapped through the integer
  container.
* `as_array()` on a `ui32` buffer now returns a `bit64::integer64`
  instead of a base `integer`, so values `>= 2^31` round-trip losslessly
  rather than wrapping to negative.

# pjrt 0.3.0

## Features

* Added `buffer_copy()` function to copy buffer between devices.
* New `pjrt_register_custom_call()` allows external packages to register C/C++
  XLA FFI handlers with the PJRT plugin. Registration is deferred until the
  plugin loads, so handlers can be registered during `.onLoad()`.
* `pjrt_device()` now returns cached `PJRTDevice` instances, so repeated calls
  for the same device yield objects with stable identity (useful for hashing
  and caching, e.g. in `{anvl}`'s JIT).

## Bug fixes

* The configure script now uses the `protoc` compiler from the same installation
  as the linked protobuf library, preventing version mismatches when multiple
  protobuf versions are installed.
* Compiling a program for a specific CPU device (e.g. `cpu:1`) now targets
  that device instead of silently falling back to `cpu:0`.
* Fixed device targeting when compiling against a distributed PJRT client,
  where global device IDs and local hardware ordinals diverge.

## Error messages

* Improved error message when attempting to use CUDA on unsupported OS
  or platform.


# pjrt 0.2.0

## Asynchronous API

Operations such as host <-> device transfers and program execution were previously only
synchronous. Now, they are asynchronous which has considerable performance
benefits, especially on GPU.
Specifically:

* `pjrt_buffer()` and `pjrt_execute()` return immediately, but the returned buffer is not
  necessarily ready. To await a transfer or computation of a buffer, use
  `await()`. However, this is handled within PJRT, so this function never has to
  be called by a user.
* `as_array()` is still synchronous, but there is now the asynchronous version
  `as_array_async()` but this is rarely needed.
  If used, it returns a `PJRTArrayPromise` object which can be converted to
  an R `array`/`vector` via `value()`.
* To check whether a `PJRTBuffer` or `PJRTArrayPromise` is ready, use
  `is_ready()`.

## Features

* Added `dtype` support for `PJRTBuffer`s via the `tengen::dtype` S3 generic. `"bool"` is now accepted as an alias for `"i1"`/`"pred"`.
* Accept `DataType` objects in the `dtype` parameter of `pjrt_buffer()`.
* Support `device` argument in `pjrt_compile()`.

## Bug fixes

* Protect from segfaults in raw to buffer conversion.
* Protect from segfault during device mismatch in `pjrt_execute()`.

## Platforms & Installation

* Added support for Linux ARM (aarch64) using CPU backend.
* Simplified CUDA installation via the `cuda12.8` package, which now only
  requires compatible drivers to be installed.

## Miscellaneous

* The printer for `PJRTBuffer` now uses `"bool"` instead of `"pred"` to avoid
  discrepancies with {anvl}.

# pjrt 0.1.1

## Bug fixes

* Fix formatting of +-Inf/NaN for f64


# pjrt 0.1.0

* Initial release

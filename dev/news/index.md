# Changelog

## pjrt (development version)

### Bug fixes

- `check_err()` no longer leaks the underlying `PJRT_Error` when
  converting a plugin error into an R exception.

### Features

- [`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md),
  [`pjrt_scalar()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md),
  and
  [`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
  now call R’s garbage collector and retry once when the plugin reports
  `RESOURCE_EXHAUSTED`. Unreferenced `PJRTBuffer` external pointers are
  finalized between attempts so their device memory is released before
  the retry.
- The first time a PJRT plugin needs to be downloaded, interactive
  sessions now ask for confirmation before downloading (similar to
  `torch`). Non-interactive sessions no longer download automatically.
  The `PJRT_INSTALL` environment variable overrides this: set it to
  `"1"` to always download without asking, or `"0"` to never download.

### Internal

- The test suite now only runs when the `PJRT_TEST` environment variable
  is set to `"1"`, so it is skipped on CRAN (where the required PJRT
  plugin cannot be downloaded). CI sets `PJRT_TEST=1`.

## pjrt 0.4.0

### Features

- Added QR, LU, SVD, and symmetric eigendecomposition support on both
  CPU and CUDA via the FFI registration mechanism.
- Added an vignette on how to register custom calls via the FFI
  registration mechanisms with coverage of both CUDA and CPU-specific
  aspects.
- Added support for the `bit64` package to better support long integers.
- [`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md),
  [`pjrt_scalar()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md),
  and
  [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
  gain a `check` argument (default `FALSE`). When `TRUE`, the call
  errors instead of silently losing information: on input if `data`
  contains `NA`s, on output if the materialized R vector contains a
  value that’s indistinguishable from `NA` or that has wrapped through
  the integer container.
- [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
  on a `ui32` buffer now returns a
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  instead of a base `integer`, so values `>= 2^31` round-trip losslessly
  rather than wrapping to negative.

## pjrt 0.3.0

### Features

- Added `buffer_copy()` function to copy buffer between devices.
- New
  [`pjrt_register_custom_call()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_register_custom_call.md)
  allows external packages to register C/C++ XLA FFI handlers with the
  PJRT plugin. Registration is deferred until the plugin loads, so
  handlers can be registered during `.onLoad()`.
- [`pjrt_device()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_device.md)
  now returns cached `PJRTDevice` instances, so repeated calls for the
  same device yield objects with stable identity (useful for hashing and
  caching, e.g. in `{anvil}`’s JIT).

### Bug fixes

- The configure script now uses the `protoc` compiler from the same
  installation as the linked protobuf library, preventing version
  mismatches when multiple protobuf versions are installed.
- Compiling a program for a specific CPU device (e.g. `cpu:1`) now
  targets that device instead of silently falling back to `cpu:0`.
- Fixed device targeting when compiling against a distributed PJRT
  client, where global device IDs and local hardware ordinals diverge.

### Error messages

- Improved error message when attempting to use CUDA on unsupported OS
  or platform.

## pjrt 0.2.0

### Asynchronous API

Operations such as host \<-\> device transfers and program execution
were previously only synchronous. Now, they are asynchronous which has
considerable performance benefits, especially on GPU. Specifically:

- [`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)
  and
  [`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
  return immediately, but the returned buffer is not necessarily ready.
  To await a transfer or computation of a buffer, use
  [`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md).
  However, this is handled within PJRT, so this function never has to be
  called by a user.
- [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
  is still synchronous, but there is now the asynchronous version
  [`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md)
  but this is rarely needed. If used, it returns a `PJRTArrayPromise`
  object which can be converted to an R `array`/`vector` via
  [`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md).
- To check whether a `PJRTBuffer` or `PJRTArrayPromise` is ready, use
  [`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md).

### Features

- Added `dtype` support for `PJRTBuffer`s via the
  [`tengen::dtype`](https://r-xla.github.io/tengen/reference/dtype.html)
  S3 generic. `"bool"` is now accepted as an alias for `"i1"`/`"pred"`.
- Accept `DataType` objects in the `dtype` parameter of
  [`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md).
- Support `device` argument in
  [`pjrt_compile()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_compile.md).

### Bug fixes

- Protect from segfaults in raw to buffer conversion.
- Protect from segfault during device mismatch in
  [`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md).

### Platforms & Installation

- Added support for Linux ARM (aarch64) using CPU backend.
- Simplified CUDA installation via the `cuda12.8` package, which now
  only requires compatible drivers to be installed.

### Miscellaneous

- The printer for `PJRTBuffer` now uses `"bool"` instead of `"pred"` to
  avoid discrepancies with {anvl}.

## pjrt 0.1.1

### Bug fixes

- Fix formatting of +-Inf/NaN for f64

## pjrt 0.1.0

- Initial release

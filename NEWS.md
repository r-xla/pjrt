# pjrt (development version)

# pjrt 0.3.0

## Features

* Added `buffer_copy()` function to copy buffer between devices.
* New `pjrt_register_custom_call()` allows external packages to register C/C++
  XLA FFI handlers with the PJRT plugin. Registration is deferred until the
  plugin loads, so handlers can be registered during `.onLoad()`.
* `pjrt_device()` now returns cached `PJRTDevice` instances, so repeated calls
  for the same device yield objects with stable identity (useful for hashing
  and caching, e.g. in `{anvil}`'s JIT).

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

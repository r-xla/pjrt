# Changelog

## pjrt (development version)

### Asynchronous API

Operations such as host \<-\> device transfers and program execution
were previously only synchronous. Now, they are asynchronous which has
considerable performance benefits, especially on GPU. Specifically: \*
[`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)
and
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
return immediately, but the returned buffer is not necessarily ready. To
await a transfer or computation of a buffer, use
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md).
However, this is handled within PJRT, so this function never has to be
called from a user. \*
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
is still synchronous, but there is now the asynchronous version
[`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md)
but this is rarely needed. If used, it returns a `PJRTArrayPromise`
object which can be converted to an R `array`/`vector` via
[`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md). \* To
check whether a `PJRTBuffer` or `PJRTArrayPromise` is ready, use
[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md).

### Features

- Added `dtype` support for `PJRTBuffer`s via the
  [`tengen::dtype`](https://r-xla.github.io/tengen/reference/dtype.html)
  S3 generic. `"bool"` is now accepted as an alias for `"i1"`/`"pred"`.
- Support `device` argument in
  [`pjrt_compile()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_compile.md).

### Bug fixes

- Protect from segfaults in raw to buffer conversion.

### Miscellaneous

- The printer for `PJRTBuffer` now uses `"bool"` instead of `"pred"` to
  avoid discrepancies with {anvil}.

## pjrt 0.1.1

### Bug fixes

- Fix formatting of +-Inf/NaN for f64

## pjrt 0.1.0

- Initial release

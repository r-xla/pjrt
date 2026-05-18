# Convert a PJRTBuffer to an R Array

Transfer buffer data from device to host and return an R array.

## Usage

``` r
# S3 method for class 'PJRTBuffer'
as_array(x, scan_na = FALSE, ...)
```

## Arguments

- x:

  ([`PJRTBuffer`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md))  
  Buffer to convert.

- scan_na:

  (`logical(1)`)  
  If `TRUE` and the buffer dtype is one of the four integer dtypes that
  round-trip through a signed R container (`i32` / `ui32` via `integer`,
  `i64` / `ui64` via
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)),
  scan the materialized vector for the reserved NA bit pattern
  (`INT_MIN` or `INT64_MIN`) and raise an error if any are present.
  No-op for float, boolean, and small-integer dtypes (which have no
  NA-collision risk).

- ...:

  Additional arguments (unused).

## Value

An R `array` (or `vector` for shape
[`integer()`](https://rdrr.io/r/base/integer.html)).

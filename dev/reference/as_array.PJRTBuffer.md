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
  If `TRUE` and the buffer dtype is `"i32"`, scan the materialized R
  integer vector for `NA_integer_` values and raise an error if any are
  present. No-op for non-`i32` dtypes.

- ...:

  Additional arguments (unused).

## Value

An R `array` (or `vector` for shape
[`integer()`](https://rdrr.io/r/base/integer.html)).

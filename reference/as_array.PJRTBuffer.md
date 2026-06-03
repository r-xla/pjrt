# Convert a PJRTBuffer to an R Array

Transfer buffer data from device to host and return an R array.

## Usage

``` r
# S3 method for class 'PJRTBuffer'
as_array(x, check = FALSE, ...)
```

## Arguments

- x:

  ([`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.md))  
  Buffer to convert.

- check:

  (`logical(1)`)  
  If `TRUE`, sanity-check the materialized R vector against losing
  information across the device-to-host boundary, and abort if any
  problematic value is detected:

  - **`i32` / `i64`**: any `NA` in the result. R's `NA_integer_` shares
    the bit pattern `INT_MIN`; `bit64`'s `NA_integer64_` shares
    `INT64_MIN`. A legitimate device value at those bit patterns is
    indistinguishable from `NA` once materialized in R.

  - **`ui64`**: any negative value in the result. `ui64` is stored as
    [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
    (signed 64-bit), which wraps values `>= 2^63` to negative — exactly
    `2^63` becomes `NA_integer64_`, anything above becomes a non-NA
    negative integer64.

  No-op for float, boolean, and small/unsigned-32 integer dtypes —
  `ui32` is now stored as `integer64` and has full headroom, so it
  cannot produce a wrapped or NA value.

- ...:

  Additional arguments (unused).

## Value

An R `array` (or `vector` for shape
[`integer()`](https://rdrr.io/r/base/integer.html)).

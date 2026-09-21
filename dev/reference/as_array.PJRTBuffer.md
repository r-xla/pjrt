# Convert a PJRTBuffer to an R Array

Transfer buffer data from device to host and return an R array.

## Usage

``` r
# S3 method for class 'PJRTBuffer'
as_array(x, check = "warn", ...)
```

## Arguments

- x:

  ([`PJRTBuffer`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md))  
  Buffer to convert.

- check:

  (`character(1)` \| `FALSE`)  
  How to report a materialized value that R's type cannot hold: `"warn"`
  (the default) warns and returns it anyway, `"err"` aborts, and `FALSE`
  skips the scan altogether. `TRUE` is not accepted — with two levels of
  strictness it does not say which one is meant.

  The cases scanned for are:

  - **`i32` / `i64`**: any `NA` in the result. R's `NA_integer_` shares
    the bit pattern `INT_MIN`; `bit64`'s `NA_integer64_` shares
    `INT64_MIN`. A legitimate device value at those bit patterns is
    indistinguishable from `NA` once materialized in R.

  - **`ui64`**: any negative value in the result. `ui64` is stored as
    [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
    (signed 64-bit), which wraps values `>= 2^63` to negative — exactly
    `2^63` becomes `NA_integer64_`, anything above becomes a non-NA
    negative integer64.

  Each case is a value the R type genuinely cannot hold, so the check
  has no false positives: it fires exactly when the returned vector
  would misrepresent the buffer.

  No-op for float, boolean, and small/unsigned-32 integer dtypes —
  `ui32` is stored as `integer64` and has full headroom, so it cannot
  produce a wrapped or NA value.

- ...:

  Additional arguments (unused).

## Value

An R `array` (or `vector` for shape
[`integer()`](https://rdrr.io/r/base/integer.html)).

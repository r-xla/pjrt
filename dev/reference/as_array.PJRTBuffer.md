# Convert a PJRTBuffer to an R Array

Transfer buffer data from device to host and return an R array.

With `await = FALSE`, the transfer runs asynchronously: the array is
returned immediately with its final type, `dim` and class, and reading
its values blocks until the transfer has completed. Use
[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md)
to check whether it has (non-blocking) and
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md) to
block until it has. The values are then checked when they are first
read, or by
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md).

`check = "err"` requires `await = TRUE`. With `await = FALSE` the check
runs only once the values are first read, and that read can come from
any code – [`print()`](https://rdrr.io/r/base/print.html), arithmetic,
another package's C code. An error would then surface there, far from
the
[`as_array()`](https://r-xla.github.io/xlamisc/reference/as_array.html)
call that caused it, and interrupt code that does not expect reading a
vector to fail. A warning does neither, so `check = "warn"` and
`check = FALSE` are allowed.

## Usage

``` r
# S3 method for class 'PJRTBuffer'
as_array(x, check = "warn", await = TRUE, ...)
```

## Arguments

- x:

  ([`PJRTBuffer`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md))  
  Buffer to convert.

- check:

  (`character(1)` \| `FALSE`)  
  How to report a materialized value that R's type cannot hold: `"warn"`
  (the default) warns and returns it anyway, `"err"` aborts (only with
  `await = TRUE`), and `FALSE` skips the scan altogether. `TRUE` is not
  accepted — with two levels of strictness it does not say which one is
  meant.

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

- await:

  (`logical(1)`)  
  Whether to wait for the transfer to the host (the default). If
  `FALSE`, return immediately with an array whose values arrive
  asynchronously.

- ...:

  Additional arguments (unused).

## Value

An R `array` (or `vector` for shape
[`integer()`](https://rdrr.io/r/base/integer.html)).

## See also

[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md),
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md),
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)

## Examples

``` r
buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
as_array(buf)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4

# Start the transfer, and only block when the values are read
result <- as_array(buf, await = FALSE)
is_ready(result)
#> [1] TRUE
result
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```

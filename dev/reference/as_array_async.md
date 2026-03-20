# Convert buffer to R array asynchronously

Start an asynchronous transfer of buffer data from device to host.
Returns immediately with a `PJRTArrayPromise` object.

Use [`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md) to
get the R array (blocks if not ready). Use
[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md)
to check if transfer has completed (non-blocking).

## Usage

``` r
as_array_async(x, ...)
```

## Arguments

- x:

  A `PJRTBuffer` object.

- ...:

  Additional arguments (unused).

## Value

A `PJRTArrayPromise` object. Call
[`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md) to get
the R array.

## See also

[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html),
[`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md),
[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md),
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md),
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md)

## Examples

``` r
buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
result <- as_array_async(buf)
is_ready(result)
#> [1] TRUE
value(result)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```

# Convert buffer to R array asynchronously

Start an asynchronous transfer of buffer data from device to host.

Returns immediately with an R array that stands for the transfer's
result: it has its final type, `dim` and class already, and reading its
values blocks until the transfer has completed. Use
[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md)
to check if the transfer has completed (non-blocking), and
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md) to
block until it has.

The values are checked as
[`as_array()`](https://r-xla.github.io/xlamisc/reference/as_array.html)
checks them, once the transfer has completed: when they are first read,
or by [`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md).
With `check = "err"`, an array holding a value R cannot represent errors
on every read.

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

An R `array` (or `vector` for shape
[`integer()`](https://rdrr.io/r/base/integer.html)).

## See also

[`as_array()`](https://r-xla.github.io/xlamisc/reference/as_array.html),
[`is_ready()`](https://r-xla.github.io/pjrt/dev/reference/is_ready.md),
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md),
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)

## Examples

``` r
buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
result <- as_array_async(buf)
is_ready(result)
#> [1] TRUE
result
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```

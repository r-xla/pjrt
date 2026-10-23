# Check if an async operation is ready

Non-blocking check to see if an async operation has completed.

For a
[`PJRTBuffer`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md),
whether its computation or transfer has finished. For an array returned
by
[`as_array(x, await = FALSE)`](https://r-xla.github.io/pjrt/dev/reference/as_array.PJRTBuffer.md),
whether its transfer to the host has finished. Any other R object is
always ready.

## Usage

``` r
is_ready(x, ...)
```

## Arguments

- x:

  An async value object.

- ...:

  Additional arguments (unused).

## Value

`TRUE` if the operation has completed, `FALSE` otherwise.

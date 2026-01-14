# Convert buffer to R array asynchronously

Start an asynchronous transfer of buffer data from device to host.
Returns immediately with an async array promise object.

Use [`value()`](value.md) to get the R array (blocks if not ready). Use
[`is_ready()`](is_ready.md) to check if transfer has completed
(non-blocking). Use
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
as an alias for [`value()`](value.md).

This function also accepts `pjrt_buffer_promise` objects (from
[`pjrt_execute_async()`](pjrt_execute_async.md) or
[`pjrt_buffer_async()`](pjrt_buffer_async.md)), enabling fully async
pipelines. The transfer is chained to the previous operation without
blocking - PJRT handles the dependency internally.

## Usage

``` r
as_array_async(x)
```

## Arguments

- x:

  A `PJRTBuffer` or `pjrt_buffer_promise` object.

## Value

A `pjrt_array_promise` object. Call [`value()`](value.md) to get the R
array.

## See also

[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html),
[`value()`](value.md), [`is_ready()`](is_ready.md),
[`pjrt_execute_async()`](pjrt_execute_async.md)

## Examples

``` r
# Create a buffer
buf <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")

# Start async transfer
result <- as_array_async(buf)

# Check if ready (non-blocking)
is_ready(result)
#> [1] TRUE

# Get the R array (blocks if not ready)
value(result)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```

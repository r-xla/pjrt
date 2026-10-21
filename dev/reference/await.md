# Await an async operation

Block until the async operation is complete and return the object.

For a
[`PJRTBuffer`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md),
waits for its computation or transfer. For an array returned by
[`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md),
waits for its transfer to the host and materializes the array. Any other
R object is returned as is.

## Usage

``` r
await(x, ...)
```

## Arguments

- x:

  An async value object.

- ...:

  Additional arguments (unused).

## Value

The awaited object (invisibly).

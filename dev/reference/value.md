# Get the value of an async operation

Materialize and return the result of an async operation. Blocks until
the operation is complete if it hasn't finished yet.

For `PJRTArrayPromise`, returns the materialized R array. For
`PJRTBuffer`, use
[`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md) to
block until ready.

## Usage

``` r
value(x, ...)
```

## Arguments

- x:

  An async value object.

- ...:

  Additional arguments (unused).

## Value

The materialized value.

# Get the value of an async operation

Materialize and return the result of an async operation. Blocks until
the operation is complete if it hasn't finished yet.

Returns `PJRTBuffer` for buffers or an R array for `PJRTArrayPromise`.

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

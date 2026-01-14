# Create a PJRT Buffer Promise (internal)

Internal constructor for async buffer results. Users should not call
this directly - use [`pjrt_execute_async()`](pjrt_execute_async.md) or
[`pjrt_buffer_async()`](pjrt_buffer_async.md) instead.

The buffer is valid immediately and can be used in subsequent operations
(PJRT handles dependencies internally). Call [`value()`](value.md) to
block until the operation is complete.

## Usage

``` r
pjrt_buffer_promise(buffer, event, data_holder = NULL)
```

## Arguments

- buffer:

  A PJRTBuffer external pointer (valid immediately).

- event:

  PJRTEvent external pointer (or NULL if already complete).

- data_holder:

  Optional XPtr keeping host data alive until transfer completes.

## Value

A `pjrt_buffer_promise` object.

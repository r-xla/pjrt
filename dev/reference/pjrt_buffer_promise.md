# Create a PJRT Buffer Promise (internal)

Internal constructor for async buffer results. Users should not call
this directly - use
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
or
[`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)
instead.

The buffer is valid immediately and can be used in subsequent operations
(PJRT handles dependencies internally). Call
[`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md) to
block until the operation is complete.

## Usage

``` r
pjrt_buffer_promise(buffer, event, data_holder = NULL, events = list())
```

## Arguments

- buffer:

  A PJRTBuffer external pointer (valid immediately).

- event:

  PJRTEvent external pointer (or NULL if already complete).

- data_holder:

  Optional XPtr keeping host data alive until transfer completes.

- events:

  List of ancestor events to check for errors (for chained operations).

## Value

A `PJRTBufferPromise` object.

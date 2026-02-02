# Create a PJRT Array Promise (internal)

Internal constructor for async device-to-host transfer results. Users
should not call this directly - use
[`as_array_async()`](as_array_async.md) instead.

## Usage

``` r
PJRTArrayPromise(data, event, dtype, shape, events = list())
```

## Arguments

- data:

  XPtr to std::vector\<uint8_t\> holding raw bytes (row-major).

- event:

  PJRTEvent external pointer (or NULL).

- dtype:

  Element type string (e.g., "f32", "i32").

- shape:

  Integer vector of dimensions.

- events:

  List of ancestor events to check for errors (for chained operations).

## Value

A `PJRTArrayPromise` object.

# Create a PJRT Array Promise (internal)

Internal constructor for async device-to-host transfer results. Users
should not call this directly - use
[`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md)
instead.

## Usage

``` r
pjrt_array_promise(data, dtype, shape)
```

## Arguments

- data:

  XPtr to PJRTHostData holding raw bytes and completion event.

- dtype:

  Element type string (e.g., "f32", "i32").

- shape:

  Integer vector of dimensions.

## Value

A `PJRTArrayPromise` object.

# Create a PJRT Array Promise (internal)

Internal constructor for async device-to-host transfer results. Users
should not call this directly - use
[`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md)
instead.

## Usage

``` r
pjrt_array_promise(data, dtype, shape, minor_to_major)
```

## Arguments

- data:

  XPtr to PJRTHostData holding raw bytes and completion event.

- dtype:

  Element type string (e.g., "f32", "i32").

- shape:

  Integer vector of dimensions.

- minor_to_major:

  Integer vector giving the device buffer's layout (minor-to-major
  dimension order), used to reorder the bytes on readback.

## Value

A `PJRTArrayPromise` object.

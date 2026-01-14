# Create a PJRT Buffer asynchronously

Create a PJRT Buffer from R data asynchronously. Returns immediately
with a `pjrt_buffer_promise` object.

The buffer is valid immediately and can be used as input to operations.
The event signals when PJRT is done reading from host memory.

Use [`value()`](value.md) to get the `PJRTBuffer` (blocks if not ready).
Use [`is_ready()`](is_ready.md) to check if transfer has completed
(non-blocking).

## Usage

``` r
pjrt_buffer_async(data, dtype = NULL, device = NULL, shape = NULL, ...)
```

## Arguments

- data:

  (any)  
  Data to convert to a `PJRTBuffer`.

- dtype:

  (`NULL` \| `character(1)`)  
  The type of the buffer. Currently supported types are:

  - `"pred"`: predicate (i.e. a boolean)

  - `"{s,u}{8,16,32,64}"`: Signed and unsigned integer (for `integer`
    data).

  - `"f{32,64}"`: Floating point (for `double` or `integer` data). The
    default (`NULL`) depends on the method:

  - `logical` -\> `"pred"`

  - `integer` -\> `"i32"`

  - `double` -\> `"f32"`

  - `raw` -\> must be supplied

- device:

  (`NULL` \| `PJRTDevice` \| `character(1)`)  
  A `PJRTDevice` object or the name of the platform to use ("cpu",
  "cuda", ...), in which case the first device for that platform is
  used. The default is to use the CPU platform, but this can be
  configured via the `PJRT_PLATFORM` environment variable.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The dimensions of the buffer. The default (`NULL`) is to infer them
  from the data if possible. The default (`NULL`) depends on the method.

- ...:

  (any)  
  Additional arguments. For `raw` types, this includes:

  - `row_major`: Whether to read the data in row-major format or
    column-major format. R uses column-major format.

## Value

A `pjrt_buffer_promise` object. Call [`value()`](value.md) to get the
`PJRTBuffer`.

## See also

[`pjrt_buffer()`](pjrt_buffer.md), [`value()`](value.md),
[`is_ready()`](is_ready.md)

## Examples

``` r
# Create a buffer asynchronously
x <- pjrt_buffer_async(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")

# Check if ready (non-blocking)
is_ready(x)
#> [1] TRUE

# Get the buffer (blocks if not ready)
buf <- value(x)
```

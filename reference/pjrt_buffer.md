# Create a PJRT Buffer

Create a PJRT Buffer from an R object. Any numeric PJRT buffer is an
array and 0-dimensional arrays are used as scalars. `pjrt_buffer` will
create a array with dimensions `(1)` for a vector of length 1, while
`pjrt_scalar` will create a 0-dimensional array for an R vector of
length 1.

To create an empty buffer (at least one dimension must be 0), use
`pjrt_empty`.

**Important**: No checks are performed when creating the buffer, so you
need to ensure that the data fits the selected element type (e.g., to
prevent buffer overflow) and that no NA values are present.

## Usage

``` r
pjrt_buffer(data, dtype = NULL, device = NULL, shape = NULL, ...)

pjrt_scalar(data, dtype = NULL, device = NULL, ...)

pjrt_empty(dtype, shape, device = NULL)
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

`PJRTBuffer`

## Extractors

- [`platform()`](platform.md) -\> `character(1)`: for the platform name
  of the buffer (`"cpu"`, `"cuda"`, ...).

- [`device()`](https://r-xla.github.io/tengen/reference/device.html) -\>
  `PJRTDevice`: for the device of the buffer (also includes device
  number)

- [`elt_type()`](elt_type.md) -\> `PJRTElementType`: for the element
  type of the buffer.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html) -\>
  [`integer()`](https://rdrr.io/r/base/integer.html): for the shape of
  the buffer.

## Converters

- [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
  -\> `array` \| `vector`: for converting back to R (`vector` is only
  used for shape [`integer()`](https://rdrr.io/r/base/integer.html)).

- [`as_raw()`](https://r-xla.github.io/tengen/reference/as_raw.html) -\>
  `raw` for a raw vector.

## Reading and Writing

- [`safetensors::safe_save_file`](https://mlverse.github.io/safetensors/reference/safe_save_file.html)
  for writing to a safetensors file.

- [`safetensors::safe_load_file`](https://mlverse.github.io/safetensors/reference/safe_load_file.html)
  for reading from a safetensors file.

## Scalars

When calling this function on a vector of length 1, the resulting shape
is `1L`. To create a 0-dimensional buffer, use `pjrt_scalar` where the
resulting shape is [`integer()`](https://rdrr.io/r/base/integer.html).

## Examples

``` r
# Create a buffer from a numeric vector
buf <- pjrt_buffer(c(1, 2, 3, 4))
buf
#> PJRTBuffer 
#>  1.0000
#>  2.0000
#>  3.0000
#>  4.0000
#> [ CPUf32{4} ] 

# Create a buffer from a matrix
mat <- matrix(1:6, nrow = 2)
buf <- pjrt_buffer(mat)
buf
#> PJRTBuffer 
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2x3} ] 

# Create an integer buffer from an array
arr <- array(1:8, dim = c(2, 2, 2))
buf <- pjrt_buffer(arr)
# Create a scalar (0-dimensional array)
scalar <- pjrt_scalar(42, dtype = "f32")
scalar
#> PJRTBuffer 
#>  42.0000
#> [ CPUf32{} ] 
# Create an empty buffer
empty <- pjrt_empty(dtype = "f32", shape = c(0, 3))
empty
#> PJRTBuffer 
#> [ CPUf32{0x3} ] 
```

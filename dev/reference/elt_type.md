# Element Type

Get the element type of a buffer.

## Usage

``` r
elt_type(x)
```

## Arguments

- x:

  ([`PJRTBuffer`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)
  or `PJRTBufferPromise`)  
  Buffer.

## Examples

``` r
buf <- pjrt_buffer(c(1.0, 2.0, 3.0))
elt_type(buf)
#> <f32>
```

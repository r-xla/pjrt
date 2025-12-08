# Platform Name

Get the platform name of a PJRT buffer.

## Usage

``` r
platform(x, ...)
```

## Arguments

- x:

  (`PJRTBuffer`)  
  The buffer.

- ...:

  Additional arguments (unused).

## Value

`character(1)`

## Examples

``` r
if (FALSE) { # plugin_is_downloaded()
buf <- pjrt_buffer(c(1, 2, 3))
platform(buf)
}
```

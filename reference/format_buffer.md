# Format Buffer Data

Formats buffer data into a character vector of string representations of
individual elements suitable for stableHLO.

## Usage

``` r
format_buffer(buffer)
```

## Arguments

- buffer:

  ([`PJRTBuffer`](pjrt_buffer.md))  
  A PJRT buffer object.

## Value

[`character()`](https://rdrr.io/r/base/character.html) A character
vector containing the formatted elements.

## Examples

``` r
buf <- pjrt_buffer(c(1.5, 2.5, 3.5))
format_buffer(buf)
#> [1] "1.50000000e+00" "2.50000000e+00" "3.50000000e+00"
```

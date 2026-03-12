# Convert to PJRT Plugin

Convert a platform name to a PJRT plugin or verify that an object is
already a plugin.

## Usage

``` r
as_pjrt_plugin(x)
```

## Arguments

- x:

  (any)  
  Object to convert to a PJRT plugin. Currently supports `PJRTPlugin`
  and `character(1)`.

## Value

`PJRTPlugin`

## Examples

``` r
# Convert from platform name
plugin <- as_pjrt_plugin("cpu")
plugin
#> <PJRTPlugin:cpu>
```

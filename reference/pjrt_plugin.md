# Create PJRT Plugin

Create a PJRT plugin for a specific platform.

## Usage

``` r
pjrt_plugin(platform)
```

## Arguments

- platform:

  (`character(1)`)  
  Platform name (e.g., "cpu", "cuda", "metal").

## Value

`PJRTPlugin`

## Extractors

- [`plugin_attributes()`](plugin_attributes.md) -\>
  [`list()`](https://rdrr.io/r/base/list.html): for the attributes of
  the plugin.

## Examples

``` r
plugin <- pjrt_plugin("cpu")
plugin
#> <PJRTPlugin:cpu>
```

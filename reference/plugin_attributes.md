# Get Plugin Attributes

Get the attributes of a PJRT plugin. This commonly includes:

- `xla_version`

- `stablehlo_current_version`

- `stablehlo_minimum_version`

But the implementation depends on the plugin.

## Usage

``` r
plugin_attributes(plugin)
```

## Arguments

- plugin:

  (`PJRTPlugin` \| `character(1)`)  
  The plugin (or platform name) to get the attributes of.

## Value

named [`list()`](https://rdrr.io/r/base/list.html)

## Examples

``` r
if (FALSE) { # plugin_is_downloaded("cpu")
plugin_attributes("cpu")
}
```

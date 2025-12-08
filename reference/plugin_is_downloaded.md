# Check if Plugin is Downloaded

Check if a plugin is downloaded.

## Usage

``` r
plugin_is_downloaded(platform = NULL)
```

## Arguments

- platform:

  (`character(1)`)  
  Platform name.

## Value

`logical(1)`

## Examples

``` r
# Check if CPU plugin is downloaded
plugin_is_downloaded("cpu")
#> [1] TRUE
```

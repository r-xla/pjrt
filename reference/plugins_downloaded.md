# Check if Plugin is Downloaded

Check if one more more plugin is already downloaded.

## Usage

``` r
plugins_downloaded(platforms = NULL)
```

## Arguments

- platforms:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Platform names.

## Value

`logical(1)`

## Examples

``` r
# Check if CPU plugin is downloaded
plugins_downloaded("cpu")
#> [1] TRUE
```

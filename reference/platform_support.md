# Platform Support

Which PJRT backends are available on which operating system and CPU
architecture.

The possible support levels are:

- `"yes"`: The plugin is available and fully supported.

- `"wsl2"`: Only available via the Windows Subsystem for Linux (WSL2),
  i.e. by running R inside Linux.

- `"no"`: No plugin is available.

## Usage

``` r
platform_support()
```

## Value

([`data.frame()`](https://rdrr.io/r/base/data.frame.html))  
One row per operating system and architecture, with character columns
`os`, `arch`, and one column per backend (`cpu`, `cuda`) holding the
support level.

## Examples

``` r
platform_support()
#>        os   arch cpu cuda
#> 1   Linux x86_64 yes  yes
#> 2   Linux  arm64 yes  yes
#> 3 Windows x86_64 yes wsl2
#> 4 Windows  arm64  no   no
#> 5   macOS x86_64 yes   no
#> 6   macOS  arm64 yes   no
```

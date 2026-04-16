# Create a PJRT Device

Create a PJRT Device from an R object.

## Usage

``` r
pjrt_device(device)
```

## Arguments

- device:

  (any)  
  The device.

## Value

`PJRTDevice`

## Extractors

- [`platform()`](https://r-xla.github.io/pjrt/dev/reference/platform.md)
  for a `character(1)` representation of the platform.

## Examples

``` r
if (FALSE) { # plugins_downloaded("cpu")
# Show available devices for CPU client
devices(pjrt_client("cpu"))
# Create device 0 for CPU client
dev <- pjrt_device("cpu:0")
dev
}
```

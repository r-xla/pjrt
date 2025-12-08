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

- [`platform()`](platform.md) for a `character(1)` representation of the
  platform.

## Examples

``` r
# Show available devices for CPU client
devices(pjrt_client("cpu"))
#> [[1]]
#> <CpuDevice(id=0)>
#> 
# Create device 0 for CPU client
dev <- pjrt_device("cpu:0")
dev
#> <CpuDevice(id=0)>
```

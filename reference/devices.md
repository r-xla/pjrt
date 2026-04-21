# Devices

Get the addressable devices.

## Usage

``` r
devices(x = NULL, ...)
```

## Arguments

- x:

  An object to get devices from: a
  [`PJRTClient`](https://r-xla.github.io/pjrt/reference/pjrt_client.md),
  a
  [`PJRTLoadedExecutable`](https://r-xla.github.io/pjrt/reference/pjrt_compile.md),
  or `NULL` (default client).

- ...:

  Additional arguments (currently unused).

## Value

`list` of `PJRTDevice`

## Examples

``` r
# Create client (defaults to CPU)
client <- pjrt_client()
devices(client)
#> [[1]]
#> <CpuDevice(id=0)>
#> 
```

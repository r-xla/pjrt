# Create a Client

Create a PJRT client for a specific device.

## Usage

``` r
pjrt_client(platform = NULL, ...)
```

## Arguments

- platform:

  (`character(1)` \| `NULL`)  
  Platform name (e.g., "cpu", "cuda", "metal"). If `NULL`, use
  `PJRT_PLATFORM` environment variable or default to "cpu".

- ...:

  Additional options passed to the PJRT client creation. For CPU
  clients, you can pass `cpu_device_count` to specify the number of CPU
  devices. You can also configure this via `PJRT_CPU_DEVICE_COUNT`
  environment variable.

## Value

`PJRTClient`

## Extractors

- [`platform()`](platform.md) for a `character(1)` representation of the
  platform.

- [`devices()`](devices.md) for a `list` of `PJRTDevice` objects.

## Examples

``` r
# Create a client (defaults to CPU)
client <- pjrt_client()
client
#> <PJRTClient:cpu>
```

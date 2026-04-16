# Devices

Get the addressable devices for a PJRT client.

## Usage

``` r
devices(client = NULL)
```

## Arguments

- client:

  ([`PJRTClient`](https://r-xla.github.io/pjrt/dev/reference/pjrt_client.md))  
  Object convertible to a `PJRTClient`.

## Value

`list` of `PJRTDevice`

## Examples

``` r
if (FALSE) { # plugins_downloaded()
# Create client (defaults to CPU)
client <- pjrt_client()
devices(client)
}
```

# Create PJRT Client

Create a PJRT client for a specific plugin and platform.

## Usage

``` r
plugin_client_create(plugin, platform, options = list())
```

## Arguments

- plugin:

  (`PJRTPlugin`)  
  The plugin to create a client for.

- platform:

  (`character(1)`)  
  The platform to create a client for.

- options:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Additional options to pass to the client.

## Value

`PJRTClient`

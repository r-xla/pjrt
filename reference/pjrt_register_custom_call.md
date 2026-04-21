# Register a Custom Call Handler

Register an XLA FFI handler for use with `stablehlo.custom_call`.

Handlers are C/C++ functions defined using the XLA FFI API (see
`xla/ffi/api/ffi.h` shipped in pjrt's `inst/include/`). They are passed
to this function as external pointers.

Registration is deferred: if the PJRT plugin for a given platform is not
yet loaded, the handler is queued and registered automatically when
[`pjrt_plugin()`](https://r-xla.github.io/pjrt/reference/pjrt_plugin.md)
loads it.

## Usage

``` r
pjrt_register_custom_call(target_name, handler, .package = NULL)
```

## Arguments

- target_name:

  (`character(1)`)  
  The target name used in `stablehlo.custom_call @target_name(...)`.

- handler:

  A named list of external pointers (`externalptr`) to
  `XLA_FFI_Handler`s, keyed by PJRT platform name (e.g.,
  `list(host = ptr)` or `list(host = cpu_ptr, cuda = cuda_ptr)`).

- .package:

  (`character(1)` or `NULL`)  
  The package registering this handler. When provided, handlers are
  automatically removed from the registry when the package unloads.

## Value

`NULL` (invisibly).

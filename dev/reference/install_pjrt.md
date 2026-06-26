# Install PJRT Plugins

Download and cache the PJRT plugins needed to run `pjrt`. The CPU plugin
is always installed. The CUDA plugin is installed in addition when a
CUDA-capable GPU is detected, or when `cuda = TRUE` is passed
explicitly.

Plugins are otherwise downloaded lazily the first time a client is
created, but the download requires user confirmation, unless the
environment variable `PJRT_INSTALL=1`.

## Usage

``` r
install_pjrt(cuda = NULL)
```

## Arguments

- cuda:

  (`logical(1)` \| `NULL`)  
  Whether to also install the CUDA plugin. When `NULL` (the default),
  CUDA support is auto-detected: the CUDA plugin is installed when an
  NVIDIA GPU is available on a Linux x86_64 machine.

## Value

([`character()`](https://rdrr.io/r/base/character.html))  
The platforms that were installed, invisibly.

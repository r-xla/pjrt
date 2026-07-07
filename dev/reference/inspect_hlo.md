# Inspect the HLO Source of a Program

The result contains the program before and after optimization.

## Usage

``` r
inspect_hlo(program, device = NULL)
```

## Arguments

- program:

  (`PJRTProgram`)  
  The program to inspect.

- device:

  (`NULL` \| `PJRTDevice` \| `character(1)`)  
  Device to compile for.

## Value

A `PJRTHloModuleSrcs`

## Enabling HLO dumping

Dumping is driven by the XLA compiler's own dump mechanism, enabled
through the `XLA_FLAGS` environment variable. **Two** flags are
required:

- `--xla_dump_hlo_as_text` – dump the HLO in text form.

- `--xla_dump_to=<dir>` – write the dump into `<dir>`. `inspect_hlo()`
  reads the HLO back from the files XLA writes here.

XLA reads `XLA_FLAGS` **once, before the first compilation in an R
process**, so both flags must be set at the very start of a fresh
session, before any
[`pjrt_compile()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_compile.md)
or
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
call:

    Sys.setenv(XLA_FLAGS = "--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text")
    library(pjrt)
    # ... build a program ...
    inspect_hlo(prog)

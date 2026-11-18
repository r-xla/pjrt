# Compile a Program

Compile a `PJRTProgram` program into a `PJRTExecutable`.

## Usage

``` r
pjrt_compile(
  program,
  compile_options = new_compile_options(),
  device = NULL,
  xla_options = list()
)
```

## Arguments

- program:

  (`character(1)`)  
  A program to compile.

- compile_options:

  (`PJRTCompileOptions`)  
  Compile options.

- device:

  (`NULL` \| `PJRTDevice` \| `character(1)`)  
  A `PJRTDevice` object or the name of the platform to use ("cpu",
  "cuda", ...), in which case the first device for that platform is
  used. The default is to use the CPU platform, but this can be
  configured via the `PJRT_PLATFORM` environment variable.

- xla_options:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  XLA compiler options for this compilation only. Each is named after a
  field of XLA's `DebugOptions` message (`xla.proto`), the names
  `XLA_FLAGS` takes without the leading `--`, and XLA applies it on top
  of its defaults and of `XLA_FLAGS`. The type of a value decides how
  XLA reads it: a `logical(1)` sets a boolean option, an `integer(1)` an
  integer option (or an enum by number), a `double(1)` a floating-point
  option, and a `character(1)` is parsed by the option's type – `"True"`
  or `"False"`, a number, the name of an enum value, or a
  comma-separated list for a repeated option, which replaces its whole
  value. So an integer option takes `3L` or `"3"`, but not `3`, and a
  single-precision option takes a `double(1)` only if it is positive, so
  `0` or a negative value has to be given as a string, e.g. `"0"`. XLA
  reports an unknown name or a value of the wrong type when compiling.

  An option of map type, such as `xla_backend_extra_options`, takes a
  `character(1)` in `XLA_FLAGS` syntax: comma-separated entries
  `key=value`, or `key` for an empty value, e.g.
  `"xla_cpu_use_multi_output_fusion"`. Its entries are added to those
  the plugin applies by default, so entries set through `XLA_FLAGS` are
  kept. XLA does not check the entries: it ignores a key it does not
  know, and some keys count by their presence alone, so
  `"xla_cpu_use_multi_output_fusion=false"` turns that option on as
  well. On the CPU, XLA hands a key that does not start with `xla_` to
  LLVM, which ends the R process if it does not know the key.

  XLA's overrides cannot set a map, so pjrt reads the plugin's
  `DebugOptions`, with `XLA_FLAGS` applied, back from a small program it
  compiles once per client, adds the entries and sends the whole set.
  That differs from a compilation without map options in a few ways: XLA
  reads some options only when a compilation sends such a set (with
  `xla_dump_to`, it then also writes the compilation's inputs, and it
  dumps the small program too), and with `--xla_flags_reset=true` a
  later change to `XLA_FLAGS` does not reach compilations with map
  options. Reading the set back needs a plugin that reports the compile
  options it applies, which the CPU plugin does and the CUDA plugin does
  not; there, set the option through `XLA_FLAGS` before the plugin is
  loaded.

## Value

`PJRTExecutable`

## Examples

``` r
# Create a simple program
src <- r"(
func.func @main(\%arg0: tensor<2xf32>) -> tensor<2xf32> {
  return \%arg0 : tensor<2xf32>
}
)"
prog <- pjrt_program(src = src)
exec <- pjrt_compile(prog)
#> Error: -:2:17: error: unexpected character
#> <unknown>:0: error: Failed to parse using StableHLO v1.23.0, this could indicate forward incompatibility, >12w old unsupported plugin, or a portable artifact that needs to be further downgraded.

# Write the optimized HLO of this compilation to a directory
dir <- tempfile()
exec <- pjrt_compile(prog, xla_options = list(xla_dump_to = dir))
#> Error: -:2:17: error: unexpected character
#> <unknown>:0: error: Failed to parse using StableHLO v1.23.0, this could indicate forward incompatibility, >12w old unsupported plugin, or a portable artifact that needs to be further downgraded.
list.files(dir)
#> character(0)

# Keep XLA:CPU from duplicating cheap operations into the fusions that use them
exec <- pjrt_compile(
  prog,
  device = "cpu",
  xla_options = list(xla_backend_extra_options = "xla_cpu_use_multi_output_fusion")
)
#> Error: -:2:17: error: unexpected character
#> <unknown>:0: error: Failed to parse using StableHLO v1.23.0, this could indicate forward incompatibility, >12w old unsupported plugin, or a portable artifact that needs to be further downgraded.
```

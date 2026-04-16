# Compile a Program

Compile a `PJRTProgram` program into a `PJRTExecutable`.

## Usage

``` r
pjrt_compile(program, compile_options = new_compile_options(), device = NULL)
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
#> <unknown>:0: error: Failed to parse using StableHLO v1.13.2, this could indicate forward incompatibility, >12w old unsupported plugin, or a portable artifact that needs to be further downgraded.
```

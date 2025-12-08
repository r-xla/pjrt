# Compile a Program

Compile a `PJRTProgram` program into a `PJRTExecutable`.

## Usage

``` r
pjrt_compile(
  program,
  compile_options = new_compile_options(),
  client = pjrt_client()
)
```

## Arguments

- program:

  (`character(1)`)  
  A program to compile.

- compile_options:

  (`PJRTCompileOptions`)  
  Compile options.

- client:

  (`PJRTClient` \| `character(1)`)  
  A PJRT client object or the name of the platform to use ("cpu",
  "cuda", ...), from which the client will be created.

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
#> <unknown>:0: error: Failed to parse using StableHLO v1.12.1, this could indicate forward incompatibility, >12w old unsupported plugin, or a portable artifact that needs to be further downgraded.
```

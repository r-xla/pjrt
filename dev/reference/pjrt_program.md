# Create a `PJRTProgram`

Create a program from a string or file path.

## Usage

``` r
pjrt_program(src = NULL, path = NULL, format = c("mlir", "hlo"))
```

## Arguments

- src:

  (`character(1)`) Source code.

- path:

  (`character(1)`) Path to the program file.

- format:

  (`character(1)`) One of "mlir" or "hlo".

## Value

`PJRTProgram`

## Examples

``` r
# Create a program from source
src <- "
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  return %arg0 : tensor<2xf32>
}
"
prog <- pjrt_program(src = src)
prog
#> PJRTProgram(format=mlir, code_size=92)
#> 
#> func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
#>   return %arg0 : tensor<2xf32>
#> }
#> 
#>  
```

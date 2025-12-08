# Execute a PJRT program

Execute a PJRT program with the given inputs and execution options.

**Important:** Arguments are passed by position and names are ignored.

## Usage

``` r
pjrt_execute(executable, ..., execution_options = NULL, simplify = TRUE)
```

## Arguments

- executable:

  (`PJRTLoadedExecutable`)  
  A PJRT program.

- ...:

  (`PJRTBuffer)`  
  Inputs to the program. Named are ignored and arguments are passed in
  order.

- execution_options:

  (`PJRTExecuteOptions`)  
  Optional execution options for configuring buffer donation and other
  settings.

- simplify:

  (`logical(1)`)  
  If `TRUE` (default), a single output is returned as a `PJRTBuffer`. If
  `FALSE`, a single output is returned as a `list` of length 1
  containing a `PJRTBuffer`.

## Value

`PJRTBuffer` \| `list` of `PJRTBuffer`s

## Examples

``` r
if (FALSE) { # plugin_is_downloaded()
# Create and compile a simple identity program
src <- r"(
func.func @main(
  \%x: tensor<2x2xf32>,
  \%y: tensor<2x2xf32>
) -> tensor<2x2xf32> {
  \%0 = "stablehlo.add"(\%x, \%y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(\%0): (tensor<2x2xf32>) -> ()
}
)"
prog <- pjrt_program(src = src)
exec <- pjrt_compile(prog)

# Execute with input
x <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
y <- pjrt_buffer(c(5, 6, 7, 8), shape = c(2, 2), dtype = "f32")
pjrt_execute(exec, x, y)
}
```

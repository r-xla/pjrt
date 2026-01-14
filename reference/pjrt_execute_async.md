# Execute a PJRT program asynchronously

Execute a PJRT program asynchronously with the given inputs. Returns
immediately with buffer promise(s) that can be awaited later.

Use [`value()`](value.md) to get the result (blocks if not ready). Use
[`is_ready()`](is_ready.md) to check if execution has completed
(non-blocking). Use [`as_array_async()`](as_array_async.md) to chain
async buffer-to-host transfer.

Inputs can be `PJRTBuffer` objects or buffer promises
(`pjrt_buffer_promise`). Buffer promises are resolved automatically
before execution.

## Usage

``` r
pjrt_execute_async(executable, ..., execution_options = NULL, simplify = TRUE)
```

## Arguments

- executable:

  (`PJRTLoadedExecutable`)  
  A PJRT program.

- ...:

  (`PJRTBuffer` \| `pjrt_buffer_promise`)  
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

A `pjrt_buffer_promise` object (or list of them if multiple outputs).
Call [`value()`](value.md) to get the `PJRTBuffer`.

## See also

[`pjrt_execute()`](pjrt_execute.md), [`value()`](value.md),
[`is_ready()`](is_ready.md), [`as_array_async()`](as_array_async.md),
[`pjrt_buffer_async()`](pjrt_buffer_async.md)

## Examples

``` r
# Create and compile a simple program
src <- r"(
func.func @main(\%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(\%x): (tensor<2x2xf32>) -> ()
}
)"
prog <- pjrt_program(src = src)
exec <- pjrt_compile(prog)
#> Error: -:2:17: error: unexpected character
#> <unknown>:0: error: Failed to parse using StableHLO v1.13.2, this could indicate forward incompatibility, >12w old unsupported plugin, or a portable artifact that needs to be further downgraded.

# Execute asynchronously
x <- pjrt_buffer(c(1.0, 2.0, 3.0, 4.0), shape = c(2, 2), dtype = "f32")
result <- pjrt_execute_async(exec, x)
#> Error: object 'exec' not found

# Check if ready (non-blocking)
is_ready(result)
#> Error: object 'result' not found

# Get the result (blocks if not ready)
value(result)
#> Error: object 'result' not found

# Chain with async buffer-to-host transfer
arr <- as_array_async(result)
#> Error: object 'result' not found
value(arr)
#> Error: object 'arr' not found
```

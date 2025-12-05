# Execute a PJRT program asynchronously

Returns lazy buffers immediately; execution continues in the background.

## Usage

``` r
pjrt_execute_lazy(executable, ..., execution_options = NULL, simplify = TRUE)
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

`PJRTLazyBuffer` or list of them.

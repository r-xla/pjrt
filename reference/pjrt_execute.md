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

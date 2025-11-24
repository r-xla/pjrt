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

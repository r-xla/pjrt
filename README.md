
<!-- README.md is generated from README.Rmd. Please edit that file -->

# pjrt

<!-- badges: start -->

[![R-CMD-check](https://github.com/r-xla/pjrt/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/r-xla/pjrt/actions/workflows/R-CMD-check.yaml)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

The {pjrt} package provides an R interface to the
[PJRT](https://github.com/openxla/pjrt) (Pretty much Just another
RunTime), which allows you to run XLA or stableHLO programs on a variety
of hardware backends. For a low-level interface to write stableHLO
programs, you can use the
[stablehlo](https://github.com/r-xla/stablehlo) package.

## Installation

You can install the development version of pjrt from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("r-xla/pjrt")
```

## Example

Below, we create a simple program in the stableHLO MLIR dialect that
adds two tensors of shape `(2, 2)` and returns the result.

``` r
library(pjrt)
src <- r"(
func.func @main(
  %x: tensor<2x2xf32>,
  %y: tensor<2x2xf32>
) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0): (tensor<2x2xf32>) -> ()
}
)"
```

The {pjrt} package now allows to create a PJRT program from this string.

``` r
program <- pjrt_program(src, format = "mlir")
program
#> PJRTProgram(format=mlir, code_size=221)
#> 
#> func.func @main(
#>   %x: tensor<2x2xf32>,
#>   %y: tensor<2x2xf32>
#> ) -> tensor<2x2xf32> {
#> ...
```

This program can now be compiled using a client that wraps a plugin,
which is a dynamic library that implements the PJRT API. The plugin is
the part that is hardware-specific, i.e. different hardware backends
provide their own plugin. Without specifying it explicitly, we use the
default client, which uses the CPU plugin.

``` r
program_compiled <- pjrt_compile(program)
```

To run this program, we need to create a so PJRT Buffer, which
represents the data, which are mostly tensors.

To create such a tensor, we can use the `pjrt_buffer()` function.
Because the program above expects `2x2` tensors with element type `f32`,
we need to create buffers with this type.

``` r
x <- matrix(as.double(1:4), nrow = 2)
y <- matrix(as.double(5:8), nrow = 2)
x_buffer <- pjrt_buffer(x, type = "f32")
y_buffer <- pjrt_buffer(y, type = "f32")
```

Having created these buffers, we can now run the program.

``` r
result <- pjrt_execute(program_compiled, x_buffer, y_buffer)
```

We can convert this result back to R:

``` r
as_array(result)
#>      [,1] [,2]
#> [1,]    6   10
#> [2,]    8   12
```

## Platform Support

- **Linux**
  - :white_check_mark: CPU backend is fully supported.
  - :white_check_mark: CUDA (NVIDIA GPU) backend is fully supported.
- **Windows**
  - :warning: Currently only supported via Windows Subsystem for Linux
    (WSL2).
- **macOS**
  - :white_check_mark: CPU backend is supported.
  - :warning: Metal (Apple GPU) backend is available but not fully
    functional.

## Acknowledgements

The design of the {pjrt} package was inspired by the
[gopjrt](https://github.com/gomlx/gopjrt) implementation.

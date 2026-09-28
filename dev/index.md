# pjrt

Package website: [release](https://r-xla.github.io/pjrt/) \|
[dev](https://r-xla.github.io/pjrt/dev/)

{pjrt} is an R interface to [PJRT](https://openxla.org/xla/pjrt), the
runtime API of [OpenXLA](https://openxla.org/). It compiles
[StableHLO](https://openxla.org/stablehlo) (or HLO) programs with the
XLA compiler into executables for a specific device and runs them there.
StableHLO programs do not depend on the framework that produced them, so
a program exported from, e.g., JAX can be run by {pjrt}. To *create*
StableHLO programs from R, see the
[stablehlo](https://github.com/r-xla/stablehlo) package.

{pjrt} is a low-level runtime. For array computing in R with JIT
compilation and automatic differentiation, use
[anvl](https://github.com/r-xla/anvl), which builds on {pjrt}.

## Installation

``` r

install.packages("pjrt", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
pjrt::install_pjrt()
```

Each backend (CPU or CUDA) is a PJRT *plugin*: a shared library that is
not bundled with the package but downloaded separately.
[`install_pjrt()`](https://r-xla.github.io/pjrt/dev/reference/install_pjrt.md)
downloads the CPU plugin and, if an NVIDIA GPU is detected, the CUDA
plugin together with the {pjrt.cuda} package, which provides the CUDA
runtime libraries. Without
[`install_pjrt()`](https://r-xla.github.io/pjrt/dev/reference/install_pjrt.md),
a plugin is downloaded the first time its platform is used. See
[`?pjrt`](https://r-xla.github.io/pjrt/dev/reference/pjrt-package.md)
for the environment variables that control this.

## Quick Start

Below, we compile and run a StableHLO program that adds two `f32`
tensors of shape `(2x2)`.

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
program <- pjrt_program(src, format = "mlir")
program
#> PJRTProgram(format=mlir, code_size=221)
#> 
#> func.func @main(
#>   %x: tensor<2x2xf32>,
#>   %y: tensor<2x2xf32>
#> ) -> tensor<2x2xf32> {
#> ...
executable <- pjrt_compile(program, device = "cpu")

x <- pjrt_buffer(c(1, 2, 3, 4), shape = c(2, 2), dtype = "f32")
x
#> PJRTBuffer 
#>  1 3
#>  2 4
#> [ CPUf32{2x2} ]
y <- pjrt_buffer(c(5, 6, 7, 8), shape = c(2, 2), dtype = "f32")
y
#> PJRTBuffer 
#>  5 7
#>  6 8
#> [ CPUf32{2x2} ]

pjrt_execute(executable, x, y)
#> PJRTBuffer 
#>   6 10
#>   8 12
#> [ CPUf32{2x2} ]
```

## Main Features

- Compile StableHLO and HLO programs into device-specific executables.
- Execute them asynchronously on CPU or CUDA devices.
- Convert between R arrays and device buffers, with range checks and
  `NA` handling.
- Register custom XLA FFI handlers.
- Read and write buffers using the
  [safetensors](https://github.com/mlverse/safetensors) format.

## Platform Support

| OS      | Architecture | CPU | CUDA |
|:--------|:-------------|:---:|:----:|
| Linux   | x86_64       |  ✓  |  ✓   |
| Linux   | arm64        |  ✓  |  ✓   |
| Windows | x86_64       |  ✓  | WSL2 |
| Windows | arm64        |  ✗  |  ✗   |
| macOS   | x86_64       |  ✓  |  ✗   |
| macOS   | arm64        |  ✓  |  ✗   |

✓ supported  ·  WSL2: via the Windows Subsystem for Linux  ·  ✗ not
supported

## Acknowledgements

- The development of this package is supported by
  [MaRDI](https://www.mardi4nfdi.de/about/mission).
- Without [OpenXLA](https://openxla.org/), none of this would be
  possible.
- The design of the {pjrt} package was inspired by the
  [gopjrt](https://github.com/gomlx/gopjrt) implementation.
- The project also uses various components from OpenXLA:
  - [PJRT C
    API](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h).
  - [PJRT C API FFI
    Extension](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_ffi_extension.h).
  - Various protobuf files, see `./tools/copy-proto.R` for which ones.
  - Plugin implementations for CPU and CUDA (we are using the builds
    from [zml/pjrt-artifacts](https://github.com/zml/pjrt-artifacts/),
    and our own [Windows build](https://github.com/r-xla/pjrt-builds)).

# Get Started

## Main Classes

The {pjrt} package provides an R interface to
[PJRT](https://openxla.org/xla/pjrt). PJRT is part of the
[OpenXLA](https://openxla.org/) project, which is an open source
compiler ecosystem for machine learning or more generally Accelerated
Linear Algebra (XLA). It defines a standardized interface which serves
as a portability layer that allows a single framework to be used with
different hardware backends. PJRT defines what a concrete “backend”
needs to be able to do in order to be usable with a frontend (like
[JAX](https://jax.readthedocs.io/)) using PJRT. Anyone can implement a
specific PJRT plugin (`PJRTPlugin`) for a specific hardware backend.
When using PJRT, such a plugin is loaded as a shared library and is
wrapped in a PJRT client (`PJRTClient`). In this package, we currently
support CPU plugins, NVIDIA GPU plugins, and Metal (Apple GPU;
experimental) plugins.

When creating a plugin upon first use, the shared library is downloaded
and cached.

``` r
library(pjrt)
plugin <- pjrt_plugin("cpu")
plugin
#> <PJRTPlugin:cpu>
```

Such a plugin can be used to create a client for a specific platform.

``` r
client <- plugin_client_create(plugin, "cpu")
client
#> <PJRTClient:cpu>
```

Currently, there will be exactly one client per platform and they are
stored in a global cache.

``` r
the <- getFromNamespace("the", "pjrt")
the[["clients"]][["cpu"]]
#> <PJRTClient:cpu>
the[["plugins"]][["cpu"]]
#> <PJRTPlugin:cpu>
```

A client can have one or more devices. This is not relevant for CPU
clients, but is in principle useful for GPUs, although this is not fully
supported yet.

``` r
cpu_devices <- devices(client)
cpu_device <- cpu_devices[[1L]]
```

The most important operations that a client supports are data handling,
compilation and execution.

### Data Handling

Using a client, we can move data from the host (standard R objects) to a
specific device to create a buffer (`PJRTBuffer`) (also known as a
tensor or array). A buffer has a type, which consists of the types of
the (homogenous) elements (`PJRTElementType`) and the shape. Scalars are
represented as 0-dimensional buffers and dimensions can be 0.

``` r
host_data <- 1:4
device_buffer <- pjrt_buffer(host_data, device = cpu_device, shape = c(2L, 2L), dtype = "f32")
device_buffer
#> PJRTBuffer 
#> 1.0000 3.0000
#> 2.0000 4.0000
#> [ CPUf32{2x2} ]
```

After creation, we can inspect the shape and element type of the buffer.

``` r
shape(device_buffer)
#> [1] 2 2
elt_type(device_buffer)
#> <f32>
device(device_buffer)
#> <CpuDevice(id=0)>
```

We can also move the data back to the host from a buffer.

``` r
host_data_from_buffer <- as_array(device_buffer)
host_data_from_buffer
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```

In principle, it is also supported to move data from one device to
another, but this is not supported yet.

### Compilation

PJRT offers an interface to compile programs into executables. These
programs can be written in HLO or the newer StableHLO. The
[stablehlo](https://github.com/r-xla/stablehla) package allows you to
easily create StableHLO programs in R. Below, we define a simple program
that adds two `f32` tensors of shape `(2, 2)`.

``` r
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

We first create a program (`PJRTProgram`) object from the source code.

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

We can’t really do anything with this, except for compiling it into an
executable (`PJRTLoadedExecutable`). For compilation, we need to specify
the client to use, because the executable depends on the platform as the
GPU binary will differ from the CPU binary.

``` r
executable <- pjrt_compile(program, client = client)
executable
#> <PJRTLoadedExecutable>
```

### Execution

With buffers in place and an executable, we can execute the program. We
simply pass the same buffer twice as input. By default, single outputs
are unpacked into a single buffer instead of a list of buffers, but we
can disable this by setting `simplify = FALSE`.

``` r
result <- pjrt_execute(executable, device_buffer, device_buffer)
```

It would have also been possible to pass execution options
(`PJRTExecuteOptions`) to the execution.

## Serialization

We support reading and writing buffers using the
[safetensors](https://github.com/mlverse/safetensors) format, which
allows storing named lists of buffers.

``` r
tmp <- tempfile(fileext = ".safetensors")
safetensors::safe_save_file(list(x = result), tmp, framework = "pjrt")
reloaded <- safetensors::safe_load_file(tmp, framework = "pjrt")
reloaded$x
#> PJRTBuffer 
#> 2.0000 6.0000
#> 4.0000 8.0000
#> [ CPUf32{2x2} ]
```

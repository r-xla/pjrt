# Get Started

## Overview

The {pjrt} package provides an R interface to
[PJRT](https://openxla.org/xla/pjrt), part of the
[OpenXLA](https://openxla.org/) project. PJRT is a portability layer
that allows frameworks to work with different hardware backends through
a standardized interface. Anyone can implement a PJRT plugin for a
specific hardware backend, and this package currently supports CPU,
NVIDIA GPU, and Metal (Apple GPU; experimental) plugins.

A key design principle of pjrt is **asynchronous dispatch**: operations
like buffer creation and program execution return immediately with
`PJRTBuffer` objects that may not be ready yet, allowing R to continue
preparing work while the device computes in the background. This enables
efficient overlap of host and device work, which is critical for
performance on accelerators.

## Plugins and Clients

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

## Data Handling

Using a client, we can move data from the host (standard R objects) to a
specific device to create a buffer.
[`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)
returns a `PJRTBuffer` immediately — the buffer may not be ready yet
(the host-to-device transfer happens asynchronously), but R does not
block.

``` r
host_data <- 1:4
buf <- pjrt_buffer(host_data, device = cpu_device, shape = c(2L, 2L), dtype = "f32")
buf
#> PJRTBuffer 
#>  1 3
#>  2 4
#> [ CPUf32{2x2} ]
```

Even before the transfer finishes, you can inspect buffer metadata —
these operations are non-blocking because the metadata is available
immediately:

``` r
shape(buf)
#> [1] 2 2
elt_type(buf)
#> <f32>
device(buf)
#> <CpuDevice(id=0)>
```

To move data back to the host, use
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html).
This blocks until the data is available:

``` r
as_array(buf)
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```

## Compilation

PJRT compiles programs written in HLO or the newer StableHLO into
device-specific executables. The
[stablehlo](https://github.com/r-xla/stablehla) package allows you to
easily create StableHLO programs in R. Below, we define a simple program
that adds two `f32` tensors of shape `2x2`.

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

First create a program object, then compile it into an executable. Note
that compilation depends on the specific device.

``` r
program <- pjrt_program(src, format = "mlir")
executable <- pjrt_compile(program, device = cpu_device)
executable
#> <PJRTLoadedExecutable>
```

## Execution

With buffers and an executable ready, we can run the program.
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
also returns a `PJRTBuffer` immediately — R gets control back while the
device computes in the background.

``` r
x <- pjrt_buffer(c(1, 2, 3, 4), shape = c(2L, 2L), dtype = "f32")
y <- pjrt_buffer(c(5, 6, 7, 8), shape = c(2L, 2L), dtype = "f32")
result <- pjrt_execute(executable, x, y)
result
#> PJRTBuffer 
#>   6 10
#>   8 12
#> [ CPUf32{2x2} ]
```

Single outputs are unpacked by default; set `simplify = FALSE` to always
get a list.

To retrieve the result as an R array, call
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html):

``` r
as_array(result)
#>      [,1] [,2]
#> [1,]    6   10
#> [2,]    8   12
```

## Asynchronous Dispatch

### Why async?

When running computations on accelerators, a synchronous approach
creates **synchronization bubbles** — periods where either the host or
the device sits idle waiting for the other:

    Host:   [prepare data] [wait...] [prepare data] [wait...] [prepare data]
    Device: [wait.........] [compute] [wait.........] [compute] [wait.........]

With async dispatch, the host and device can work in parallel:

    Host:   [prepare batch 1] [prepare batch 2] [prepare batch 3] [prepare batch 4]
    Device:                   [compute 1......] [compute 2......] [compute 3......]

In pjrt, this happens automatically. Both
[`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)
and
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
return `PJRTBuffer` objects immediately without blocking R. The buffer
may not be ready yet, but you can pass it directly to other operations.
You only pay the synchronization cost when you actually need the
host-side result (e.g. calling
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)).

### Async types

| Operation                                                                          | Returns            | Description             |
|------------------------------------------------------------------------------------|--------------------|-------------------------|
| [`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)       | `PJRTBuffer`       | Host-to-device transfer |
| [`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)     | `PJRTBuffer`       | Program execution       |
| [`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md) | `PJRTArrayPromise` | Device-to-host transfer |

`PJRTBuffer` is a transparent async type — it can be used directly in
operations even if the underlying computation hasn’t completed yet. PJRT
handles the dependencies internally.

`PJRTArrayPromise` represents data in transit from device to host.
Because R arrays are plain R objects (not under our control like
`PJRTBuffer`), we need a separate promise type to represent “array that
isn’t ready yet.” This is why there are two functions for
synchronization:

- `await(x)` — block until a `PJRTBuffer` is ready on the device. Pure
  synchronization; no data leaves the device. Useful for timing or error
  checking, but **not required** before passing a buffer to
  [`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md).
- `value(x)` — block until a `PJRTArrayPromise` completes, then
  materialize and return the R array. This is how you get data out of
  the device.

Both types also support `is_ready(x)` for non-blocking readiness checks.

### Chaining operations

Buffers can be passed directly to other operations without waiting —
PJRT handles the dependencies internally:

``` r
# Both transfers start immediately
buf1 <- pjrt_buffer(matrix(1:4, 2, 2), dtype = "f32")
buf2 <- pjrt_buffer(matrix(5:8, 2, 2), dtype = "f32")

# Execute with promise inputs — no need to wait for transfers
result <- pjrt_execute(executable, buf1, buf2)

# Only block when we need the R array
as_array(result)
#>      [,1] [,2]
#> [1,]    6   10
#> [2,]    8   12
```

You can also chain execution with async device-to-host transfer:

``` r
src2 <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
exec2 <- pjrt_compile(pjrt_program(src2))

# Full async pipeline: transfer -> execute -> transfer back
buf <- pjrt_buffer(matrix(runif(1e6), 1000, 1000), dtype = "f32")
result <- pjrt_execute(exec2, buf)
async_output <- as_array_async(result)   # returns immediately

# R can do other work here...

# Block only when we need the final R array
output <- value(async_output)
dim(output)
#> [1] 1000 1000
```

### When does blocking happen?

Blocking occurs when:

1.  Calling
    [`await()`](https://r-xla.github.io/pjrt/dev/reference/await.md) on
    a buffer
2.  Calling
    [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
    on a buffer
3.  Calling
    [`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md) on
    a `PJRTArrayPromise`
4.  Printing a buffer (needs to read values to display them)

Operations like
[`shape()`](https://r-xla.github.io/tengen/reference/shape.html),
[`elt_type()`](https://r-xla.github.io/pjrt/dev/reference/elt_type.md),
and [`device()`](https://r-xla.github.io/tengen/reference/device.html)
do **not** block — buffer metadata is available immediately.

### Writing efficient loops

**Bad** — blocking every iteration:

``` r
for (i in seq_len(n_iterations)) {
  result <- pjrt_execute(executable, input)
  metrics <- as_array(result)  # blocks! device idles during transfer
  cat("Step", i, "loss:", metrics[1], "\n")
  input <- prepare_next_batch()  # device still idle
}
```

**Better** — log previous iteration’s metrics while device computes the
next:

``` r
prev_result <- NULL
for (i in seq_len(n_iterations)) {
  result <- pjrt_execute(executable, input)

  if (!is.null(prev_result)) {
    metrics <- as_array(prev_result)  # previous result likely ready
    cat("Step", i - 1, "loss:", metrics[1], "\n")
  }

  input <- prepare_next_batch()  # overlap with device work
  prev_result <- result
}
```

### Full pipeline example

``` r
# Multiply-add program
src3 <- r"(
func.func @main(%x: tensor<100x100xf32>, %y: tensor<100x100xf32>) -> tensor<100x100xf32> {
  %0 = "stablehlo.multiply"(%x, %y) : (tensor<100x100xf32>, tensor<100x100xf32>) -> tensor<100x100xf32>
  %1 = "stablehlo.add"(%0, %x) : (tensor<100x100xf32>, tensor<100x100xf32>) -> tensor<100x100xf32>
  "func.return"(%1): (tensor<100x100xf32>) -> ()
}
)"
exec3 <- pjrt_compile(pjrt_program(src3))

# Mini computation loop with chaining
shp <- c(100L, 100L)
x <- pjrt_buffer(matrix(runif(prod(shp)), shp[1], shp[2]), dtype = "f32")
y <- pjrt_buffer(matrix(runif(prod(shp)), shp[1], shp[2]), dtype = "f32")

for (step in seq_len(5)) {
  # Returns immediately — auto-waits for x and y if needed
  result <- pjrt_execute(exec3, x, y)
  # Chain: use result as input to next iteration
  x <- result
}

# Only retrieve the final value
final <- as_array(result)
cat("Final result shape:", dim(final), "\n")
#> Final result shape: 100 100
cat("Final result[1,1]:", final[1, 1], "\n")
#> Final result[1,1]: 22.14824
```

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
#> Columns 1 to 7
#>  2.2148e+01 1.6243e+00 3.1992e+00 3.3661e+00 3.2027e+00 3.7586e+00 2.0813e+00
#>  1.6443e+01 2.4477e-01 4.1511e+00 4.1295e+00 2.4273e+00 1.6176e+00 3.6497e-01
#>  1.0526e+00 1.4976e+00 9.9318e-01 1.9173e+00 8.9408e+00 1.2844e-01 1.1338e+01
#>  1.7938e+00 1.2325e-01 1.1881e+00 2.4539e+00 2.9107e-02 1.1639e+01 3.7073e+00
#>  2.5915e+01 2.1596e+00 3.3475e-02 4.2064e+00 1.1248e+00 1.4394e+01 9.0609e-01
#>  2.0549e+00 1.1823e+01 2.4401e+00 2.6452e+00 1.1399e+00 1.6505e+01 1.7988e+00
#>  5.0847e+00 2.3894e+00 6.9374e-01 7.1540e+00 1.9468e+00 1.2250e+01 1.5661e+00
#>  1.7365e+00 7.0727e-01 1.1450e+01 2.8378e-02 7.8192e+00 2.5113e+00 1.8746e+01
#>  6.6238e+00 3.4759e+00 1.2597e+00 1.2197e+01 1.0617e+00 4.2360e-01 8.6550e+00
#>  1.8511e-01 3.8913e+00 2.9864e+00 1.2368e+01 3.8097e+00 7.9126e-01 6.6415e-01
#>  1.9836e-01 1.7327e+00 1.0690e+01 1.2025e+00 7.5129e-01 1.8660e-01 3.4132e+00
#>  7.9593e-01 4.0803e+00 3.1893e+00 2.4257e-02 5.9005e+00 1.9903e+01 9.4508e+00
#>  4.5504e+00 2.0023e+00 1.6533e+01 3.2972e+00 4.3016e+00 1.1033e+00 5.8061e-01
#>  4.6889e+00 8.7448e+00 7.7285e+00 6.5496e-01 8.9090e+00 2.4281e+00 6.8213e+00
#>  4.0073e-01 4.9153e+00 1.9894e+00 4.4955e-01 8.8144e+00 4.8487e+00 3.4146e-01
#>  4.8680e-01 1.5711e+01 2.7894e+00 2.6457e+01 3.9455e+00 1.7037e+00 6.1513e+00
#>  2.6114e+00 8.1016e-01 1.2146e-01 1.6320e+00 7.8187e-03 2.5365e+00 1.5105e+01
#>  4.4820e-01 4.7786e-01 1.0679e+00 2.8630e-01 3.1332e+00 3.0947e-02 8.6495e+00
#>  4.6614e-01 3.0826e-01 7.0557e+00 2.1700e+00 1.5732e+00 2.2271e-01 4.7378e-02
#>  1.3440e+00 2.9234e+00 1.1559e+01 8.8668e+00 1.8729e+00 1.3320e+00 1.4763e+01
#>  6.0223e+00 5.5660e+00 7.3217e+00 8.8343e-01 2.4898e+00 2.2978e+00 1.2654e+00
#>  5.0453e+00 1.7959e+00 1.1050e+00 6.8665e+00 1.0883e+01 9.7641e+00 6.3046e-01
#>  6.3220e-01 1.5343e+00 3.7788e-02 1.0085e+01 1.0526e+00 3.9324e+00 1.2135e+00
#>  3.9624e+00 1.4638e+00 5.2830e-01 5.1944e-01 1.7337e+00 2.7231e-01 6.3342e+00
#>  7.0109e-01 9.1790e-01 1.1373e+01 1.7708e+01 5.8359e+00 5.4447e+00 1.4719e+00
#>  1.8467e+01 1.9650e+00 5.8906e+00 1.1622e+01 2.3799e+01 1.9868e+00 4.0531e-01
#>  4.5296e+00 2.0595e+01 6.0703e+00 1.1153e+00 2.1703e-01 4.6344e+00 3.5762e-01
#>  1.3230e+01 6.8999e+00 2.0481e+00 1.7136e+00 1.0243e+01 2.6142e+00 1.1486e+00
#>  5.5286e+00 4.1406e-01 1.0848e+01 4.4146e-01 7.3570e+00 1.2660e+00 8.3911e+00
#>  2.5474e+01 3.8595e+00 8.9820e+00 1.1383e+00 2.4344e+00 1.5862e+00 1.2519e+01
#>  ... [output was truncated, set max_rows = -1 to see all]
#> [ CPUf32{100x100} ]
```

## Further Reading

The async dispatch design in pjrt is inspired by JAX’s approach:

- [Asynchronous dispatch in
  JAX](https://docs.jax.dev/en/latest/async_dispatch.html)
- [The Training Cookbook: Efficiency via Asynchronous
  Dispatch](https://docs.jax.dev/en/latest/the-training-cookbook.html#efficiency-via-asynchronous-dispatch)

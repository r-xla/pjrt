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
*promises*, allowing R to continue preparing work while the device
computes in the background. This enables efficient overlap of host and
device work, which is critical for performance on accelerators.

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
returns a `PJRTBufferPromise` — a promise that resolves to a
`PJRTBuffer` once the host-to-device transfer completes. R does not
block; it returns immediately.

``` r
host_data <- 1:4
buf <- pjrt_buffer(host_data, device = cpu_device, shape = c(2L, 2L), dtype = "f32")
buf
#> <PJRTBufferPromise>
#> Status: Not awaited
#> Events: 1 
#> (Call value() to await and retrieve the buffer)
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

First create a program object, then compile it into an executable.
Compilation depends on the platform — a GPU binary differs from a CPU
binary.

``` r
program <- pjrt_program(src, format = "mlir")
executable <- pjrt_compile(program, client = client)
executable
#> <PJRTLoadedExecutable>
```

## Execution

With buffers and an executable ready, we can run the program.
[`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)
also returns a `PJRTBufferPromise` — R gets control back immediately
while the device computes.

``` r
x <- pjrt_buffer(c(1, 2, 3, 4), shape = c(2L, 2L), dtype = "f32")
y <- pjrt_buffer(c(5, 6, 7, 8), shape = c(2L, 2L), dtype = "f32")
result <- pjrt_execute(executable, x, y)
result
#> <PJRTBufferPromise>
#> Status: Not awaited
#> Events: 1 
#> (Call value() to await and retrieve the buffer)
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
return promises and don’t block R. You only pay the synchronization cost
when you actually need the result (e.g. calling
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
or [`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md)).

### Promise types

| Operation                                                                          | Returns             | Description             |
|------------------------------------------------------------------------------------|---------------------|-------------------------|
| [`pjrt_buffer()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_buffer.md)       | `PJRTBufferPromise` | Host-to-device transfer |
| [`pjrt_execute()`](https://r-xla.github.io/pjrt/dev/reference/pjrt_execute.md)     | `PJRTBufferPromise` | Program execution       |
| [`as_array_async()`](https://r-xla.github.io/pjrt/dev/reference/as_array_async.md) | `PJRTArrayPromise`  | Device-to-host transfer |

All promises support:

- `value(x)` — block until ready and return the result (`PJRTBuffer` or
  R array)
- `is_ready(x)` — non-blocking readiness check
- `as_array(x)` — convert to R array (blocks if needed)

### Chaining operations

Promises can be passed directly to other operations without calling
[`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md) first —
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
    [`value()`](https://r-xla.github.io/pjrt/dev/reference/value.md) on
    a promise
2.  Calling
    [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
    on a promise or buffer
3.  Printing a buffer (needs to read values to display them)

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
#> <PJRTBufferPromise>
#> Status: Not awaited
#> Events: 0 
#> (Call value() to await and retrieve the buffer)
```

## Further Reading

The async dispatch design in pjrt is inspired by JAX’s approach:

- [Asynchronous dispatch in
  JAX](https://docs.jax.dev/en/latest/async_dispatch.html)
- [The Training Cookbook: Efficiency via Asynchronous
  Dispatch](https://docs.jax.dev/en/latest/the-training-cookbook.html#efficiency-via-asynchronous-dispatch)

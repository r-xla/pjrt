# Asynchronous Dispatch

## The Problem: Host-Device Synchronization

When running computations on accelerators (GPUs, TPUs, or even optimized
CPU backends), there’s a fundamental tension between the **host** (R)
and the **device** (accelerator). By default, every operation follows a
synchronous pattern:

1.  R sends data to the device

2.  R waits for the transfer to complete

3.  R tells the device to execute a computation

4.  R waits for the computation to complete

5.  R retrieves the result

6.  R waits for the transfer to complete

This creates **synchronization bubbles**—periods where either the host
or the device sits idle waiting for the other. In a training loop, these
bubbles can significantly reduce throughput:

    Host:   [prepare batch] [wait...] [prepare batch] [wait...] [prepare batch]
    Device:        [wait...] [compute]        [wait...] [compute]        [wait...]

The ideal scenario overlaps host and device work:

    Host:   [prepare batch 1] [prepare batch 2] [prepare batch 3] [prepare batch 4]
    Device:          [compute 1] [compute 2] [compute 3] [compute 4]

## Asynchronous Dispatch in pjrt

The {pjrt} package provides async variants of key operations that return
immediately, allowing R to continue while the device works in the
background. These operations return **promise** objects that can be
awaited later.

### The Async API

| Sync Operation                                                         | Async Operation                                              | Returns               |
|------------------------------------------------------------------------|--------------------------------------------------------------|-----------------------|
| [`pjrt_buffer()`](../reference/pjrt_buffer.md)                         | [`pjrt_buffer_async()`](../reference/pjrt_buffer_async.md)   | `pjrt_buffer_promise` |
| [`pjrt_execute()`](../reference/pjrt_execute.md)                       | [`pjrt_execute_async()`](../reference/pjrt_execute_async.md) | `pjrt_buffer_promise` |
| [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html) | [`as_array_async()`](../reference/as_array_async.md)         | `pjrt_array_promise`  |

All async objects support:

- `value(x)` - Block until ready and return the result
- `is_ready(x)` - Check if complete (non-blocking)
- `as_array(x)` - Convert to R array (blocks if needed)

### Basic Example

``` r
library(pjrt)

# Compile a simple program
src <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
executable <- pjrt_compile(pjrt_program(src))
```

#### Synchronous execution (blocking)

``` r
# Each step blocks until complete
data <- matrix(runif(1000 * 1000), nrow = 1000)
buf <- pjrt_buffer(data, dtype = "f32")       # Blocks: wait for transfer
result <- pjrt_execute(executable, buf)        # Blocks: wait for computation
output <- as_array(result)                     # Blocks: wait for transfer back
```

#### Asynchronous execution (non-blocking)

``` r
# Operations return immediately
data <- matrix(runif(1000 * 1000), nrow = 1000)
transfer <- pjrt_buffer_async(data, dtype = "f32")  # Returns immediately
result <- pjrt_execute_async(executable, transfer)   # Returns immediately (auto-waits for transfer)
async_output <- as_array_async(result)               # Returns immediately

# R can do other work here while device computes...

# Only block when we actually need the value
output <- value(async_output)
```

## Smart Input Handling

A key feature of the async API is **smart input handling**. PJRT manages
dependencies internally, so you can chain async operations without
explicit waits:

- **Buffer promises** (`pjrt_buffer_promise`): The buffer is valid
  immediately - PJRT handles dependencies internally. R returns
  immediately without blocking.
- Buffer promises can be passed to
  [`as_array_async()`](../reference/as_array_async.md) or used as inputs
  to other executions - PJRT handles the dependency.

This enables true pipelining where transfers and execution overlap:

``` r
# Start two async transfers in parallel
buf1 <- pjrt_buffer_async(matrix(1:4, 2, 2), dtype = "f32")
buf2 <- pjrt_buffer_async(matrix(5:8, 2, 2), dtype = "f32")

# Execute with async inputs - auto-waits internally
src2 <- r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0): (tensor<2x2xf32>) -> ()
}
)"
exec2 <- pjrt_compile(pjrt_program(src2))

# Pass async transfers directly - no explicit value() needed
result <- pjrt_execute_async(exec2, buf1, buf2)
as_array(result)
#>      [,1] [,2]
#> [1,]    6   10
#> [2,]    8   12
```

### Chaining Execution with Device-to-Host Transfer

You can also chain async execution with async device-to-host transfer
without blocking:

``` r
# Start async execution
result <- pjrt_execute_async(executable, pjrt_buffer(matrix(runif(1e6), 1000, 1000), dtype = "f32"))

# Chain with async device-to-host transfer - returns immediately!
# PJRT internally waits for execution to complete before starting transfer
async_output <- as_array_async(result)

# R can do other work here...

# Only block when we need the final R array
output <- value(async_output)
```

## When Blocking Happens

Understanding when R blocks is crucial for writing efficient code.
Blocking occurs when:

1.  **Calling [`value()`](../reference/value.md)** - Explicitly waiting
    for an async operation
2.  **Calling
    [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
    on a buffer** - Transfers data from device to host
3.  **Printing a buffer** - Needs to read values to display them
4.  **Any operation that needs the actual data**

### Avoid Unnecessary Blocking

**Bad pattern** - Blocking in every iteration:

``` r
for (i in seq_len(n_iterations)) {
  result <- pjrt_execute(executable, input)
  metrics <- as_array(result)  # BLOCKS! Device waits while we transfer
  cat("Step", i, "loss:", metrics[1], "\n")
  input <- prepare_next_batch()  # Device idle during data prep
}
```

**Better pattern** - Log previous iteration’s metrics:

``` r
prev_result <- NULL
for (i in seq_len(n_iterations)) {
  # Start this iteration's computation
  result <- pjrt_execute_async(executable, input)

  # While device computes, log PREVIOUS iteration's metrics
  if (!is.null(prev_result)) {
    metrics <- as_array(prev_result)  # Previous result likely ready
    cat("Step", i - 1, "loss:", metrics[1], "\n")
  }

  # Prepare next batch while device still computing
  input <- prepare_next_batch()
  prev_result <- result
}
```

## Checking Readiness

Use [`is_ready()`](../reference/is_ready.md) for non-blocking status
checks:

``` r
# Start async operation
transfer <- pjrt_buffer_async(matrix(runif(1e6), 1000, 1000), dtype = "f32")

# Check without blocking
if (is_ready(transfer)) {
  message("Transfer complete!")
} else {
  message("Still transferring...")
}
#> Transfer complete!

# Eventually get the value (blocks if needed)
buf <- value(transfer)
```

## Full Async Pipeline Example

Here’s a complete example showing the full async pipeline:

``` r
library(pjrt)

# Simple multiply-add program
src <- r"(
func.func @main(%x: tensor<100x100xf32>, %y: tensor<100x100xf32>) -> tensor<100x100xf32> {
  %0 = "stablehlo.multiply"(%x, %y) : (tensor<100x100xf32>, tensor<100x100xf32>) -> tensor<100x100xf32>
  %1 = "stablehlo.add"(%0, %x) : (tensor<100x100xf32>, tensor<100x100xf32>) -> tensor<100x100xf32>
  "func.return"(%1): (tensor<100x100xf32>) -> ()
}
)"
executable <- pjrt_compile(pjrt_program(src))

# Simulate a mini training loop
n_steps <- 5
shape <- c(100L, 100L)

# Initialize with async transfer
x <- pjrt_buffer_async(matrix(runif(prod(shape)), shape[1], shape[2]), dtype = "f32")
y <- pjrt_buffer_async(matrix(runif(prod(shape)), shape[1], shape[2]), dtype = "f32")

for (step in seq_len(n_steps)) {
  # Execute asynchronously (auto-waits for x and y if needed)
  result <- pjrt_execute_async(executable, x, y)

  # Use result as input to next iteration (chaining)
  x <- result

  # Only on last iteration, get the actual value
  if (step == n_steps) {
    final <- as_array(result)
    cat("Final result shape:", dim(final), "\n")
    cat("Final result[1,1]:", final[1, 1], "\n")
  }
}
#> Final result shape: 100 100 
#> Final result[1,1]: 4.000973
```

## Benchmark: Sync vs Async

Let’s compare synchronous and asynchronous execution patterns. We’ll
simulate a training loop where each iteration involves preparing input
data (simulated with
[`Sys.sleep()`](https://rdrr.io/r/base/Sys.sleep.html) to represent
loading from disk or preprocessing).

**Note:** On the CPU backend used in this vignette, the async benefit is
minimal because execution is very fast and happens on the same CPU as R.
The real benefit appears on GPU/TPU backends where:

- Device computation takes significant time (hundreds of milliseconds)
- The device operates independently from the host CPU
- True parallelism is possible between host data preparation and device
  execution

``` r
library(pjrt)

# Compile a matrix multiply program
src <- r"(
func.func @main(%x: tensor<1000x1000xf32>) -> tensor<1000x1000xf32> {
  %0 = "stablehlo.dot"(%x, %x) : (tensor<1000x1000xf32>, tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  "func.return"(%0): (tensor<1000x1000xf32>) -> ()
}
)"
executable <- pjrt_compile(pjrt_program(src))

n_iterations <- 5
data_prep_time <- 0.02  # Simulate 20ms data preparation

prepare_data <- function() {
  Sys.sleep(data_prep_time)
  matrix(runif(1000 * 1000), 1000, 1000)
}
```

### Synchronous Pattern

Each operation blocks before the next can start:

``` r
sync_time <- system.time({
  for (i in seq_len(n_iterations)) {
    data <- prepare_data()
    buf <- pjrt_buffer(data, dtype = "f32")
    result <- pjrt_execute(executable, buf)
  }
  final <- as_array(result)
})

cat("Synchronous total time:", round(sync_time["elapsed"], 3), "seconds\n")
#> Synchronous total time: 0.197 seconds
```

### Asynchronous Pattern

R prepares the next batch while the device computes:

``` r
async_time <- system.time({
  data <- prepare_data()
  buf <- pjrt_buffer_async(data, dtype = "f32")
  result <- pjrt_execute_async(executable, buf)

  for (i in seq_len(n_iterations - 1)) {
    data <- prepare_data()
    buf <- pjrt_buffer_async(data, dtype = "f32")
    result <- pjrt_execute_async(executable, buf)
  }
  final <- as_array(result)
})

cat("Asynchronous total time:", round(async_time["elapsed"], 3), "seconds\n")
#> Asynchronous total time: 0.197 seconds
```

### Results

``` r
cat("Sync:", round(sync_time["elapsed"], 3), "s, Async:", round(async_time["elapsed"], 3), "s\n")
#> Sync: 0.197 s, Async: 0.197 s
```

On CPU, both patterns have similar performance because:

- CPU execution is very fast (a few milliseconds)
- The same CPU handles both R code and PJRT execution
- Little opportunity for true parallelism

On **GPU/TPU backends**, the async pattern shines because:

1.  **Data preparation overlaps with execution**: While the GPU runs
    iteration N, R prepares data for N+1
2.  **Transfers overlap with execution**: Host-to-device transfers
    happen while previous computations run
3.  **No idle time**: The device stays busy instead of waiting for R

In production GPU training loops, async dispatch can provide **1.2-2x
speedup** depending on the ratio of data preparation time to computation
time.

## Summary

| Pattern                                                                | When to Use                                                 |
|------------------------------------------------------------------------|-------------------------------------------------------------|
| [`pjrt_buffer()`](../reference/pjrt_buffer.md)                         | Simple scripts, when you need the buffer immediately        |
| [`pjrt_buffer_async()`](../reference/pjrt_buffer_async.md)             | Training loops, when overlapping transfers with computation |
| [`pjrt_execute()`](../reference/pjrt_execute.md)                       | Simple inference, when you need results immediately         |
| [`pjrt_execute_async()`](../reference/pjrt_execute_async.md)           | Training loops, pipelining multiple operations              |
| [`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html) | When you need R values now                                  |
| [`as_array_async()`](../reference/as_array_async.md)                   | When you want to overlap device-to-host transfer            |

The key insight is that **blocking should happen as late as possible**.
By using async operations and delaying
[`value()`](../reference/value.md) calls, you allow the device to stay
busy while R prepares the next batch of work.

## Further Reading

The concepts in this vignette are inspired by JAX’s approach to
asynchronous dispatch:

- [Asynchronous dispatch in
  JAX](https://docs.jax.dev/en/latest/async_dispatch.html)
- [The Training Cookbook: Efficiency via Asynchronous
  Dispatch](https://docs.jax.dev/en/latest/the-training-cookbook.html#efficiency-via-asynchronous-dispatch)

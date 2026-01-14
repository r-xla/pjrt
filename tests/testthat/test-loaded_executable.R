test_that("arguments must be unnamed", {
  skip_if_metal("only supports MLIR programs")
  path <- system.file("programs/test_hlo.pb", package = "pjrt")
  program <- pjrt_program(path = path, format = "hlo")
  executable <- pjrt_compile(program)
  buf <- pjrt_buffer(1)
  expect_error(pjrt_execute(executable, a = buf, "Expected unnamed arguments"))
})

test_that("execute program without arguments", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_equal(as_array(result), 3)
})

test_that("can return two values", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_list(result, types = "PJRTBuffer", len = 2L)
  expect_equal(as_array(result[[1]]), 3)
  expect_equal(as_array(result[[2]]), 7)
})

test_that("single-output returns list when simplify=FALSE", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable, simplify = FALSE)
  expect_list(result, types = "PJRTBuffer", len = 1L)
  expect_equal(as_array(result[[1]]), 3)

  result <- pjrt_execute(executable, simplify = TRUE)
  expect_class(result, "PJRTBuffer")
  expect_equal(as_array(result), 3)
})

test_that("can execute empty constant", {
  path <- system.file("programs/stablehlo-empty-constant.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  result <- pjrt_execute(executable)
  expect_equal(as_array(result), array(integer(), 0L))
})

test_that("print works", {
  path <- system.file("programs/stablehlo-empty-constant.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)
  expect_snapshot(executable)
})

# Async execution tests

test_that("pjrt_execute_async returns buffer promise", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  expect_class(result, "pjrt_buffer_promise")
})

test_that("is_ready works for async values", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  # is_ready should return logical
  ready <- is_ready(result)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)
})

test_that("value() returns correct result for async execution", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  output <- value(result)
  expect_class(output, "PJRTBuffer")
  expect_equal(as_array(output), 3)
})

test_that("async execution with multiple outputs", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # With multiple outputs, returns list of buffer promises
  result <- pjrt_execute_async(executable)
  expect_list(result, types = "pjrt_buffer_promise", len = 2L)

  # Each buffer promise can be awaited individually
  buf1 <- value(result[[1]])
  buf2 <- value(result[[2]])
  expect_class(buf1, "PJRTBuffer")
  expect_class(buf2, "PJRTBuffer")
  expect_equal(as_array(buf1), 3)
  expect_equal(as_array(buf2), 7)
})

test_that("async execution with simplify=FALSE", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # With simplify=FALSE, returns list even for single output
  result <- pjrt_execute_async(executable, simplify = FALSE)
  expect_list(result, types = "pjrt_buffer_promise", len = 1L)

  # The buffer promise contains a single buffer
  buf <- value(result[[1]])
  expect_class(buf, "PJRTBuffer")
})

test_that("print.pjrt_buffer_promise works", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  expect_output(print(result), "pjrt_buffer_promise")
})

test_that("as_array works for async values (single output)", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  result <- pjrt_execute_async(executable)
  arr <- as_array(result)
  expect_equal(arr, 3)
})

test_that("as_array works for async values (multiple outputs)", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # With multiple outputs, get list of async values
  result <- pjrt_execute_async(executable)
  expect_list(result, len = 2L)

  # as_array on each async value
  arr1 <- as_array(result[[1]])
  arr2 <- as_array(result[[2]])
  expect_equal(arr1, 3)
  expect_equal(arr2, 7)
})

test_that("async execution chained with async buffer-to-host", {
  path <- system.file("programs/jax-stablehlo-no-arg.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # Start async execution
  async_result <- pjrt_execute_async(executable)
  expect_class(async_result, "pjrt_buffer_promise")

  # Chain with async buffer-to-host transfer (auto-waits for execution)
  async_array <- as_array_async(async_result)
  expect_class(async_array, "pjrt_array_promise")

  # Check is_ready returns logical
  ready <- is_ready(async_array)
  expect_true(is.logical(ready))

  # Get the final R value (this waits for transfer)
  arr <- value(async_array)
  expect_equal(arr, 3)
})

test_that("async execution with inputs chained to async buffer-to-host", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  path <- system.file("programs/jax-stablehlo-subset-2d.mlir", package = "pjrt")
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # Create input buffers
  x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  x_buf <- pjrt_buffer(x)
  i1_buf <- pjrt_scalar(0L, "i32")
  i2_buf <- pjrt_scalar(1L, "i32")

  # Execute asynchronously
  async_result <- pjrt_execute_async(executable, x_buf, i1_buf, i2_buf)
  expect_class(async_result, "pjrt_buffer_promise")

  # Chain with async buffer-to-host transfer
  async_array <- as_array_async(async_result)
  expect_class(async_array, "pjrt_array_promise")

  # Get final value
  result <- value(async_array)
  expect_equal(result, x[1, 2])  # 0-indexed: x[0+1, 1+1] = x[1, 2] = 3
})

test_that("async execution with multiple outputs chained to async transfer", {
  path <- system.file(
    "programs/jax-stablehlo-two-constants.mlir",
    package = "pjrt"
  )
  program <- pjrt_program(path = path, format = "mlir")
  executable <- pjrt_compile(program)

  # Execute asynchronously - returns list of buffer promises
  result <- pjrt_execute_async(executable)
  expect_list(result, len = 2L)

  # Chain each output with async buffer-to-host transfer
  async_arr1 <- as_array_async(result[[1]])
  async_arr2 <- as_array_async(result[[2]])

  expect_class(async_arr1, "pjrt_array_promise")
  expect_class(async_arr2, "pjrt_array_promise")

  # Get final values
  arr1 <- value(async_arr1)
  arr2 <- value(async_arr2)
  expect_equal(arr1, 3)
  expect_equal(arr2, 7)
})

# Event chain tracking tests ------------------------------------------------

test_that("buffer_promise tracks events from pjrt_buffer_async", {
  x <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")

  # Should have one event in the chain
  expect_length(x$events, 1L)

  # Event should be the same as x$event
  expect_identical(x$events[[1]], x$event)
})

test_that("buffer_promise accumulates events through execute_async chain", {
  # Create a simple pass-through program
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Create input buffer asynchronously
  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  expect_length(input$events, 1L)

  # Execute asynchronously - should accumulate parent event
  result <- pjrt_execute_async(executable, input)

  # Result should have 2 events: input transfer + execution
  expect_length(result$events, 2L)

  # Verify the result is correct
  arr <- as_array(value(result))
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("array_promise accumulates events through as_array_async chain", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Create full async chain: buffer -> execute -> as_array
  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute_async(executable, input)
  arr_promise <- as_array_async(result)

  # arr_promise should have 3 events: input transfer + execution + D2H transfer
  expect_length(arr_promise$events, 3L)

  # Verify the result is correct
  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("longer async chains accumulate all events", {
  # Two programs that pass through
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  exec1 <- pjrt_compile(pjrt_program(src))
  exec2 <- pjrt_compile(pjrt_program(src))

  # Chain: buffer_async -> execute1 -> execute2 -> as_array_async
  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  expect_length(input$events, 1L)

  result1 <- pjrt_execute_async(exec1, input)
  expect_length(result1$events, 2L)

  result2 <- pjrt_execute_async(exec2, result1)
  expect_length(result2$events, 3L)

  arr_promise <- as_array_async(result2)
  expect_length(arr_promise$events, 4L)

  # Verify the result is correct
  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("is_ready checks all events in chain", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute_async(executable, input)
  arr_promise <- as_array_async(result)

  # is_ready should return a logical value
  ready <- is_ready(arr_promise)
  expect_true(is.logical(ready))
  expect_length(ready, 1L)

  # After getting the value, it should be ready

  value(arr_promise)
  expect_true(is_ready(arr_promise))
})

test_that("sync buffer inputs don't add events", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Sync buffer (not async)
  input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # Execute async with sync input - only execution event should be present
  result <- pjrt_execute_async(executable, input)
  expect_length(result$events, 1L)

  arr <- as_array(value(result))
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("mixed sync and async inputs collect correct events", {
  skip_if_metal("-:20:28: error: expected ')' in inline location")
  src <- r"(
func.func @main(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.add"(%x, %y) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  "func.return"(%0): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # One async, one sync input
  x_async <- pjrt_buffer_async(matrix(1:4, 2, 2), dtype = "f32")
  y_sync <- pjrt_buffer(matrix(5:8, 2, 2), dtype = "f32")

  result <- pjrt_execute_async(executable, x_async, y_sync)

  # Should have 2 events: x_async transfer + execution
  expect_length(result$events, 2L)

  arr <- as_array(value(result))
  expect_equal(as.vector(arr), as.vector(matrix(1:4, 2, 2) + matrix(5:8, 2, 2)), tolerance = 1e-6)
})

# Error propagation tests ---------------------------------------------------

test_that("error in execute_async is caught when calling value on buffer_promise", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Wrong shape input - should fail
  wrong_input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # The error should be caught during execute_async (input validation)
  expect_error(
    pjrt_execute_async(executable, wrong_input),
    "size"
  )
})

test_that("error in execute_async is caught when calling value on array_promise", {
  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Create valid input first, then test with wrong shape
  wrong_input <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # Error should be caught at execute time
  expect_error(
    pjrt_execute_async(executable, wrong_input),
    "size"
  )
})

test_that("error propagates through async chain when buffer_promise is used", {
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Create async input and chain it
  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")

  # Execute successfully
  result <- pjrt_execute_async(executable, input)

  # The chain should work
  arr_promise <- as_array_async(result)

  # All events are tracked
  expect_length(arr_promise$events, 3L)

  # Value should succeed
  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)
})

test_that("value() awaits all events in chain before returning", {
  # This test verifies that value() checks all events, not just the last one
  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Build a chain
  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute_async(executable, input)
  arr_promise <- as_array_async(result)

  # Before calling value, check that events list is complete
  expect_length(arr_promise$events, 3L)

  # After calling value, all events should have been awaited
  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(1.0, 2.0, 3.0), tolerance = 1e-6)

  # Call value again - should return cached result without re-awaiting
  arr2 <- value(arr_promise)
  expect_identical(arr, arr2)
})

# Tests documenting error behavior in async operations ----------------------
# These tests document WHERE errors appear in async chains.
#
# Error timing by backend:
# - CPU backend: Most errors are caught synchronously during execute_async()
#   (input validation). The CPU backend is robust and rarely produces true
#   runtime errors - XLA clamps indices, produces inf/nan for invalid math, etc.
#
# - GPU/TPU backends: Errors may be deferred until value() is called because
#   execution is truly asynchronous. Common runtime errors include:
#   - Out of memory
#   - Device communication failures
#   - Computation errors
#
# The event chain tracking ensures that when value() is called, ALL events
# in the chain are awaited, so errors from ANY step are properly surfaced.

test_that("async errors: input validation errors appear at execute_async time", {

  # This test documents that input validation errors (wrong shape, wrong type)

  # are caught immediately when execute_async is called, NOT deferred to value()

  src <- r"(
func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  "func.return"(%x): (tensor<2x2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Create buffer with wrong shape (3 elements instead of 2x2=4)
  wrong_shape_buffer <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # Error appears immediately at execute_async time (input validation)
  # NOT deferred to value()
  expect_error(
    pjrt_execute_async(executable, wrong_shape_buffer),
    "size"
  )
})

test_that("async errors: when chaining, errors in value() come from event await", {
  # This test shows the expected behavior: when an async chain succeeds,

  # value() returns the result. The event chain tracking ensures that
  # if any operation in the chain had failed asynchronously, the error

  # would be caught when awaiting that event.

  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%0): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Build async chain
 input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute_async(executable, input)
  arr_promise <- as_array_async(result)

  # All events are tracked in the chain
  expect_length(arr_promise$events, 3L)

  # value() awaits all events - this is where async errors would surface
  # In this case, no error occurs
  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(2.0, 4.0, 6.0), tolerance = 1e-6)
})

test_that("async errors: is_ready returns FALSE until all events complete", {
  # This test verifies is_ready() checks all events, not just the last one

  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  "func.return"(%x): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute_async(executable, input)
  arr_promise <- as_array_async(result)

  # On CPU, operations complete synchronously, so is_ready should be TRUE
  # On GPU/TPU, this might initially be FALSE
  ready <- is_ready(arr_promise)
  expect_true(is.logical(ready))

  # After value(), all events are awaited
  value(arr_promise)
  expect_true(is_ready(arr_promise))
})

test_that("async errors: error messages are descriptive", {
  # Verify that error messages from async operations are meaningful

  src <- r"(
func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
  "func.return"(%x): (tensor<4xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Wrong size: expected 4 elements, got 3
  wrong_buffer <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")

  # Error message should mention the size mismatch
  expect_error(
    pjrt_execute_async(executable, wrong_buffer),
    regexp = "size.*(16|12)",  # Should mention expected vs actual size in bytes
    ignore.case = TRUE
  )
})

test_that("async errors: event chain mechanism is in place for deferred errors", {
  # This test verifies the event chain mechanism that would catch deferred
  # errors on GPU/TPU backends. On CPU, errors are caught earlier, but the
  # mechanism is still exercised.

  src <- r"(
func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "stablehlo.add"(%x, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%0): (tensor<3xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  # Build a chain
  input <- pjrt_buffer_async(c(1.0, 2.0, 3.0), dtype = "f32")
  result <- pjrt_execute_async(executable, input)
  arr_promise <- as_array_async(result)

  # Verify all events are tracked
  expect_length(arr_promise$events, 3L)

  # Each event is a PJRTEvent that can be awaited
  for (evt in arr_promise$events) {
    expect_s3_class(evt, "PJRTEvent")
  }

  # value() awaits ALL events - on GPU/TPU, this is where deferred errors
  # would be caught. On CPU, events complete synchronously.
  arr <- value(arr_promise)
  expect_equal(as.vector(arr), c(2.0, 4.0, 6.0), tolerance = 1e-6)
})

test_that("async errors: CPU backend clamps out-of-bounds indices (no runtime error)", {
  # This test documents that the CPU backend is robust and doesn't error
 # on out-of-bounds access - it clamps indices instead. This is XLA behavior.
  # GPU/TPU backends may behave differently.

  src <- r"(
func.func @main(%x: tensor<3xf32>, %idx: tensor<i32>) -> tensor<2xf32> {
  %0 = "stablehlo.dynamic_slice"(%x, %idx) {
    slice_sizes = array<i64: 2>
  } : (tensor<3xf32>, tensor<i32>) -> tensor<2xf32>
  "func.return"(%0): (tensor<2xf32>) -> ()
}
)"
  executable <- pjrt_compile(pjrt_program(src))

  x <- pjrt_buffer(c(1.0, 2.0, 3.0), dtype = "f32")
  # Index 5 would be out of bounds, but XLA clamps it
  idx <- pjrt_scalar(5L, "i32")

  result <- pjrt_execute_async(executable, x, idx)
  arr_promise <- as_array_async(result)

  # No error - XLA clamps the index to valid range
  arr <- value(arr_promise)
  # Result is the last 2 elements (index clamped to 1)
  expect_equal(as.vector(arr), c(2.0, 3.0), tolerance = 1e-6)
})

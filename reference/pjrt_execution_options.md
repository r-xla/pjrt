# Create Execution Options

Create execution options for configuring how a PJRT program is executed,
including buffer donation settings. **Important**: It is not enough to
only mark a buffer as donatable (not not donatable) during runtime. The
program also needs to specify this during compile-time via input-output
aliasing (stableHLO attribute `tf.aliasing_output`).

## Usage

``` r
pjrt_execution_options(non_donatable_input_indices = integer(), launch_id = 0L)
```

## Arguments

- non_donatable_input_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  A vector of input buffer indices that should not be donated during
  execution (0-based). Buffer donation allows the runtime to reuse input
  buffers for outputs when possible, which can improve performance.
  However, if an input buffer is referenced multiple times or needs to
  be preserved, it should be marked as non-donatable.

- launch_id:

  (`integer(1)`)  
  An optional launch identifier for multi-device executions. This can be
  used to detect scheduling errors in multi-host programs.

## Value

`PJRTExecuteOptions`

## Examples

``` r
if (FALSE) { # plugin_is_downloaded()
# Create default execution options
opts <- pjrt_execution_options()

# Mark buffer 0 as non-donatable
opts <- pjrt_execution_options(non_donatable_input_indices = 0L)
}
```

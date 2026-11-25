# a bare R input with a missing value / warns where i32 keeps it, naming the input

    Code
      invisible(run(NA_integer_, "i32"))
    Condition
      Warning:
      Input `x` contains at least one "NA", which "i32" stores as "-2147483648".
      i It materializes as "NA" again in R, which `as_array()` reports on the way back.
      i Set `check = "err"` to make this an error, or `check = FALSE` to silence it; the option `pjrt.upload_check` sets the default.

# a bare R input with a missing value / is reported at the level of the option `pjrt.upload_check`

    Code
      run(NA_integer_, "i32")
    Condition
      Error:
      ! Input `x` contains at least one "NA", which "i32" stores as "-2147483648".
      i It materializes as "NA" again in R, which `as_array()` reports on the way back.
      i Set `check = FALSE` or the option `pjrt.upload_check` to accept it.


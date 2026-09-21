# an integer and a double agree at an integer dtype

    Code
      pjrt_buffer(300L, dtype = "ui8")
    Condition
      Error:
      ! Value 300 cannot be converted to "ui8" without overflow.

---

    Code
      pjrt_buffer(300, dtype = "ui8")
    Condition
      Error:
      ! Value 300 cannot be converted to "ui8" without overflow.

---

    Code
      pjrt_buffer(200L, dtype = "i8")
    Condition
      Error:
      ! Value 200 cannot be converted to "i8" without overflow.

---

    Code
      pjrt_buffer(200, dtype = "i8")
    Condition
      Error:
      ! Value 200 cannot be converted to "i8" without overflow.

---

    Code
      pjrt_buffer(-5L, dtype = "ui32")
    Condition
      Error:
      ! Value -5 cannot be converted to "ui32" without overflow.

---

    Code
      pjrt_buffer(-5, dtype = "ui32")
    Condition
      Error:
      ! Value -5 cannot be converted to "ui32" without overflow.

# a double an integer dtype cannot hold is rejected

    Code
      pjrt_buffer(1e+30, dtype = "i64")
    Condition
      Error:
      ! Value 1e+30 cannot be converted to "i64" without overflow.

---

    Code
      pjrt_buffer(-1, dtype = "ui8")
    Condition
      Error:
      ! Value -1 cannot be converted to "ui8" without overflow.

---

    Code
      pjrt_buffer(300, dtype = "ui8")
    Condition
      Error:
      ! Value 300 cannot be converted to "ui8" without overflow.

---

    Code
      pjrt_buffer(2^31, dtype = "i32")
    Condition
      Error:
      ! Value 2147483648 cannot be converted to "i32" without overflow.

---

    Code
      pjrt_buffer(2^31 + 1000, dtype = "i32")
    Condition
      Error:
      ! Value 2147484648 cannot be converted to "i32" without overflow.

---

    Code
      pjrt_buffer(Inf, dtype = "i32")
    Condition
      Error:
      ! Value Inf cannot be converted to "i32" without overflow.

---

    Code
      pjrt_buffer(NA_real_, dtype = "i64")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "i64".

---

    Code
      pjrt_buffer(NaN, dtype = "i32")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "i32".

---

    Code
      pjrt_buffer(c(1, 300), dtype = "ui8")
    Condition
      Error:
      ! Value 300 cannot be converted to "ui8" without overflow (element 2).

---

    Code
      pjrt_buffer(-1L, dtype = "ui32")
    Condition
      Error:
      ! Value -1 cannot be converted to "ui32" without overflow.

---

    Code
      pjrt_buffer(NA_integer_, dtype = "ui8")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "ui8".

---

    Code
      pjrt_buffer(NA_integer_, dtype = "i64")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "i64".

# a double at the edge of an integer dtype's range is accepted

    Code
      as_array(suppressWarnings(pjrt_buffer(NA_integer_, dtype = "i32")), check = "err")
    Condition
      Error in `as_array()`:
      ! Materialized <i32> buffer contains a value that R cannot distinguish from "NA".
      i "i32" reserves the bit pattern "-2147483648" (`INT_MIN`); "i64" reserves "-9223372036854775808" (`INT64_MIN`).
      i Set `check = FALSE` to skip this check.

# a logical uploads at any element type, not just pred

    Code
      pjrt_buffer(NA, dtype = "ui8")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "ui8".

---

    Code
      pjrt_buffer(TRUE, dtype = "nope")
    Condition
      Error:
      ! Unsupported type: nope

# a missing value at pred / is rejected, whatever the source type it arrives as

    Code
      pjrt_buffer(NA)
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "pred".

---

    Code
      pjrt_buffer(c(TRUE, NA, FALSE))
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "pred" (element 2).

---

    Code
      pjrt_scalar(NA)
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "pred".

---

    Code
      pjrt_buffer(NA_integer_, dtype = "pred")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "pred".

---

    Code
      pjrt_buffer(NA_real_, dtype = "pred")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "pred".

# a missing bit64::integer64 value / is rejected at ui64, which has no missing value to land on

    Code
      pjrt_buffer(bit64::NA_integer64_, dtype = "ui64")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "ui64".

---

    Code
      pjrt_buffer(bit64::as.integer64(c(1, NA)), dtype = "ui64")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "ui64" (element 2).

---

    Code
      pjrt_scalar(bit64::NA_integer64_, dtype = "ui64")
    Condition
      Error:
      ! Missing value (NA/NaN) cannot be converted to "ui64".

# device works

    Code
      as.character(device(buf))
    Output
      [1] "CpuDevice(id=0)"

# device print

    Code
      print(device(pjrt_buffer(1)))
    Output
      <CpuDevice(id=0)>


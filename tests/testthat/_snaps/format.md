# format_buffer works for floats

    Code
      format_buffer(pjrt_buffer(1.23, "f32"))
    Output
      [1] "1.23000002e+00"

---

    Code
      format_buffer(pjrt_buffer(1.23, "f64"))
    Output
      [1] "1.2300000000000000e+00"

---

    Code
      format_buffer(pjrt_buffer(-3.33, "f64"))
    Output
      [1] "-3.3300000000000001e+00"

---

    Code
      format_buffer(pjrt_buffer(NaN, "f32"))
    Output
      [1] "0x7FC00000"

---

    Code
      format_buffer(pjrt_buffer(Inf, "f32"))
    Output
      [1] "0x7F800000"

---

    Code
      format_buffer(pjrt_buffer(-Inf, "f32"))
    Output
      [1] "0xFF800000"


describe("is_ready", {
  it("returns a flag for a PJRTBuffer", {
    ready <- is_ready(pjrt_buffer(c(1.0, 2.0), dtype = "f32"))
    expect_true(is.logical(ready))
    expect_length(ready, 1L)
  })

  it("is TRUE for objects that are not async values", {
    expect_true(is_ready(1:3))
    expect_true(is_ready(NULL))
  })
})

describe("await", {
  it("returns a PJRTBuffer once it is ready", {
    original <- c(1.0, 2.0, 3.0, 4.0)
    buf <- await(pjrt_buffer(original, shape = c(2L, 2L), dtype = "f32"))
    expect_class(buf, "PJRTBuffer")
    expect_equal(as.vector(as_array(buf)), original, tolerance = 1e-6)
  })

  it("returns objects that are not async values unchanged", {
    expect_identical(await(1:3), 1:3)
  })
})

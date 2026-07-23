skip_if_not_installed("safetensors")

test_that("can write a safetensors file (pjrt)", {
  buffers <- list(
    x = pjrt_buffer(array(rnorm(100), dim = c(10, 10)), dtype = "f32"),
    y = pjrt_buffer(array(1:20, dim = c(4, 5)), dtype = "i32")
  )

  tmp <- tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(buffers, tmp)

  reloaded <- safetensors::safe_load_file(tmp, framework = "pjrt")

  expect_true(identical(as_array(buffers$x), as_array(reloaded$x)))
  expect_true(identical(as_array(buffers$y), as_array(reloaded$y)))
})

test_that("Can write safetensors (different data types)", {
  types <- list(
    list(pjrt_type = "f32", rtype = "double"),
    list(pjrt_type = "f64", rtype = "double"),
    list(pjrt_type = "i8", rtype = "integer"),
    list(pjrt_type = "i16", rtype = "integer"),
    list(pjrt_type = "i32", rtype = "integer"),
    list(pjrt_type = "i64", rtype = "integer"),
    list(pjrt_type = "ui8", rtype = "integer"),
    list(pjrt_type = "ui16", rtype = "integer"),
    list(pjrt_type = "ui32", rtype = "integer"),
    list(pjrt_type = "ui64", rtype = "integer"),
    list(pjrt_type = "pred", rtype = "logical")
  )

  dat <- c(0L, 1:8, 0L)
  for (type in types) {
    x <- switch(
      type$rtype,
      double = as.double(dat),
      integer = as.integer(dat),
      logical = as.logical(dat),
      cli_abort()
    )

    x <- list(
      x = pjrt_buffer(array(x, dim = c(5, 2)), dtype = type$pjrt_type)
    )

    tmp <- tempfile(fileext = ".safetensors")
    safetensors::safe_save_file(x, tmp)

    reloaded <- safetensors::safe_load_file(tmp, framework = "pjrt")

    expect_true(identical(as_array(x$x), as_array(reloaded$x)))
  }
})

test_that("f16 safetensors payloads load raw into f16 buffers", {
  # Hand-crafted file, independent of pjrt's own writer: an 8-byte header
  # length, the JSON header, then the packed row-major halfs
  # 1.0 (0x3C00), -2.5 (0xC100), 1/3 (0x3555 = 0.333251953125),
  # 65504 (0x7BFF), Inf (0x7C00) and 2^-24 (0x0001) as little-endian bytes.
  header <- '{"x":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}'
  path <- tempfile(fileext = ".safetensors")
  con <- file(path, "wb")
  writeBin(nchar(header), con, size = 8L, endian = "little")
  writeBin(charToRaw(header), con)
  writeBin(
    as.raw(c(0x00, 0x3c, 0x00, 0xc1, 0x55, 0x35, 0xff, 0x7b, 0x00, 0x7c, 0x01, 0x00)),
    con
  )
  close(con)

  dict <- safetensors::safe_load_file(path, framework = "pjrt")
  expect_equal(as.character(elt_type(dict$x)), "f16")
  expect_identical(shape(dict$x), c(2L, 3L))
  expected <- matrix(
    c(1, -2.5, 0.333251953125, 65504, Inf, 2^-24),
    nrow = 2,
    byrow = TRUE
  )
  expect_identical(as_array(dict$x), expected)
})

test_that("f16 buffers round-trip through safetensors write/read", {
  x <- list(
    x = pjrt_buffer(array(c(0.5, 1.5, -2.25, 4), dim = c(2, 2)), dtype = "f16")
  )
  tmp <- tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(x, tmp)
  reloaded <- safetensors::safe_load_file(tmp, framework = "pjrt")
  expect_equal(as.character(elt_type(reloaded$x)), "f16")
  expect_identical(as_array(x$x), as_array(reloaded$x))
})

test_that("load a file (pjrt)", {
  path <- test_path("_safetensors", "hello.safetensors")
  dict <- safetensors::safe_load_file(
    path,
    framework = "pjrt"
  )
  expect_equal(names(dict), c("hello", "world"))

  expect_equal(shape(dict$hello), c(10, 10))
  expect_true(all(as_array(dict$hello) == 1))

  expect_equal(shape(dict$world), c(5, 10))
  expect_true(all(as_array(dict$world) == 0))
})

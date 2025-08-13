skip_if_not_installed("safetensors")

test_that("can write a safetensors file (pjrt)", {
  buffers <- list(
    x = pjrt_buffer(array(rnorm(100), dim = c(10, 10)), elt_type = "f32"),
    y = pjrt_buffer(array(1:20, dim = c(4, 5)), elt_type = "i32")
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
      stop()
    )

    x <- list(
      x = pjrt_buffer(array(x, dim = c(5, 2)), elt_type = type$pjrt_type)
    )

    tmp <- tempfile(fileext = ".safetensors")
    safetensors::safe_save_file(x, tmp)

    reloaded <- safetensors::safe_load_file(tmp, framework = "pjrt")

    expect_true(identical(as_array(x$x), as_array(reloaded$x)))
  }
})

test_that("load a file (pjrt)", {
  path <- test_path("_safetensors", "hello.safetensors")
  dict <- safetensors::safe_load_file(
    path,
    framework = "pjrt"
  )
  expect_equal(names(dict), c("hello", "world"))

  expect_equal(dim(dict$hello), c(10, 10))
  expect_true(all(as_array(dict$hello) == 1))

  expect_equal(dim(dict$world), c(5, 10))
  expect_true(all(as_array(dict$world) == 0))
})

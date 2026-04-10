test_that("pjrt_register_custom_call validates inputs", {
  ptr <- get_print_handler()
  expect_error(pjrt_register_custom_call(123, list(host = ptr)))
  expect_error(pjrt_register_custom_call("foo", "not_a_ptr"))
  expect_error(pjrt_register_custom_call("foo", list(ptr)))
  expect_error(pjrt_register_custom_call("foo", ptr))
})

test_that("pjrt_register_custom_call accepts named list", {
  withr::defer({
    the[["custom_calls"]][["test_list"]] <- NULL
  })
  ptr_host <- get_print_handler()
  ptr_cuda <- get_print_handler_cuda()
  pjrt_register_custom_call("test_list", list(host = ptr_host, cuda = ptr_cuda))
  entry <- the[["custom_calls"]][["test_list"]]
  expect_named(entry$handler, c("host", "cuda"))
})

test_that("pjrt_register_custom_call maps cpu to host", {
  withr::defer({
    the[["custom_calls"]][["test_cpu_map"]] <- NULL
  })
  ptr <- get_print_handler()
  pjrt_register_custom_call("test_cpu_map", list(cpu = ptr))
  entry <- the[["custom_calls"]][["test_cpu_map"]]
  expect_named(entry$handler, "host")
})

test_that("duplicate registration overwrites", {
  withr::defer({
    the[["custom_calls"]][["test_dup"]] <- NULL
  })
  ptr <- get_print_handler()
  handler <- list(host = ptr)
  pjrt_register_custom_call("test_dup", handler)
  expect_no_error(pjrt_register_custom_call("test_dup", handler))
})

test_that("PJRT API silently overwrites when registering the same handler name twice", {
  skip_if_metal("FFI extension not available on metal")
  platform <- Sys.getenv("PJRT_PLATFORM", "cpu")
  plugin <- pjrt_plugin(platform)
  pjrt_platform <- ifelse(platform != "cuda", "host", "cuda")
  ptr <- get_print_handler()
  # "print_tensor" is already registered during .onLoad
  expect_no_error(
    impl_register_custom_call(plugin, "print_tensor", ptr, pjrt_platform)
  )
})

test_that("pjrt_unregister_custom_calls removes entries for a package", {
  withr::defer({
    the[["custom_calls"]][c("cleanup_1", "cleanup_2", "keep_this")] <- NULL
  })
  ptr <- get_print_handler()
  handler <- list(host = ptr)
  pjrt_register_custom_call("cleanup_1", handler, .package = "fake_pkg")
  pjrt_register_custom_call("cleanup_2", handler, .package = "fake_pkg")
  pjrt_register_custom_call("keep_this", handler, .package = "other_pkg")

  pjrt_unregister_custom_calls("fake_pkg")

  expect_false("cleanup_1" %in% names(the[["custom_calls"]]))
  expect_false("cleanup_2" %in% names(the[["custom_calls"]]))
  expect_true("keep_this" %in% names(the[["custom_calls"]]))
})

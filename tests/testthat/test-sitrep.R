test_that("sitrep with no plugins and no env vars", {
  # skip this test if not on met
  skip_if(!is_metal())
  withr::local_envvar(
    PJRT_PLATFORM = NA,
    PJRT_CPU_DEVICE_COUNT = NA,
    PJRT_ZML_ARTIFACT_VERSION = NA,
    PJRT_PLUGIN_PATH_CPU = NA,
    PJRT_PLUGIN_PATH_CUDA = NA,
    PJRT_PLUGIN_PATH_METAL = NA,
    PJRT_PLUGIN_URL_CPU = NA,
    PJRT_PLUGIN_URL_CUDA = NA,
    PJRT_PLUGIN_URL_METAL = NA,
    TF_CPP_MIN_LOG_LEVEL = NA,
    XLA_FLAGS = NA
  )
  expect_snapshot(pjrt_sitrep())
})

test_that("sitrep with cpu plugin", {
  local_mocked_bindings(
    plugin_os = function() "linux",
    plugin_arch = function() "x86_64",
    plugin_is_downloaded = function(p) p == "cpu",
    collect_plugin_info = function(platforms) {
      list(list(
        platform = "cpu",
        attrs = list(xla_version = "1.0.0"),
        n_devices = 2L
      ))
    }
  )
  withr::local_envvar(
    PJRT_PLATFORM = NA,
    PJRT_CPU_DEVICE_COUNT = "2",
    PJRT_ZML_ARTIFACT_VERSION = NA,
    PJRT_PLUGIN_PATH_CPU = NA,
    PJRT_PLUGIN_PATH_CUDA = NA,
    PJRT_PLUGIN_PATH_METAL = NA,
    PJRT_PLUGIN_URL_CPU = NA,
    PJRT_PLUGIN_URL_CUDA = NA,
    PJRT_PLUGIN_URL_METAL = NA,
    TF_CPP_MIN_LOG_LEVEL = NA,
    XLA_FLAGS = NA
  )
  expect_snapshot(pjrt_sitrep())
})

test_that("sitrep with cuda plugin shows recommended versions", {
  local_mocked_bindings(
    plugin_os = function() "linux",
    plugin_arch = function() "x86_64",
    plugin_is_downloaded = function(p) p == "cuda",
    collect_plugin_info = function(platforms) {
      list(list(
        platform = "cuda",
        attrs = list(xla_version = "1.0.0"),
        n_devices = 1L
      ))
    }
  )
  withr::local_envvar(
    PJRT_PLATFORM = "cuda",
    PJRT_CPU_DEVICE_COUNT = NA,
    PJRT_ZML_ARTIFACT_VERSION = NA,
    PJRT_PLUGIN_PATH_CPU = NA,
    PJRT_PLUGIN_PATH_CUDA = NA,
    PJRT_PLUGIN_PATH_METAL = NA,
    PJRT_PLUGIN_URL_CPU = NA,
    PJRT_PLUGIN_URL_CUDA = NA,
    PJRT_PLUGIN_URL_METAL = NA,
    TF_CPP_MIN_LOG_LEVEL = NA,
    XLA_FLAGS = NA
  )
  expect_snapshot(pjrt_sitrep())
})

test_that("sitrep with plugin error", {
  local_mocked_bindings(
    plugin_os = function() "linux",
    plugin_arch = function() "x86_64",
    plugin_is_downloaded = function(p) p == "cpu",
    collect_plugin_info = function(platforms) {
      list(list(
        platform = "cpu",
        error = "Failed to load plugin"
      ))
    }
  )
  withr::local_envvar(
    PJRT_PLATFORM = NA,
    PJRT_CPU_DEVICE_COUNT = NA,
    PJRT_ZML_ARTIFACT_VERSION = NA,
    PJRT_PLUGIN_PATH_CPU = NA,
    PJRT_PLUGIN_PATH_CUDA = NA,
    PJRT_PLUGIN_PATH_METAL = NA,
    PJRT_PLUGIN_URL_CPU = NA,
    PJRT_PLUGIN_URL_CUDA = NA,
    PJRT_PLUGIN_URL_METAL = NA,
    TF_CPP_MIN_LOG_LEVEL = NA,
    XLA_FLAGS = NA
  )
  expect_snapshot(pjrt_sitrep())
})

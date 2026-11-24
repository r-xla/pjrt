test_that("creation works", {
  expect_class(pjrt_plugin("cpu"), "PJRTPlugin")
})

test_that("attributes work", {
  attrs <- plugin_attributes(pjrt_plugin("cpu"))
  expect_subset(
    c("xla_version", "stablehlo_current_version", "stablehlo_minimum_version"),
    names(attrs)
  )
})

test_that("print works", {
  expect_snapshot(pjrt_plugin("cpu"))
})


test_that("invalid plugin name does not create folder", {
  expect_error(pjrt_plugin("abc"), "Invalid platform")
  expect_false(any(dir.exists(plugin_dirs("abc"))))
})

# Points the package directory and the user cache at temporary directories and
# returns the CPU plugin directories within them: the package one first.
local_plugin_dirs <- function(env = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = env)
  pkg_dir <- file.path(root, "pkg")
  dir.create(pkg_dir)
  withr::local_envvar(
    R_USER_CACHE_DIR = file.path(root, "cache"),
    PJRT_PLUGIN_PATH_CPU = NA,
    PJRT_PLUGIN_URL_CPU = NA,
    .local_envir = env
  )
  local_mocked_bindings(installed_package_dir = function() pkg_dir, .env = env)
  c(file.path(pkg_dir, "plugins", "cpu"), file.path(root, "cache", "R", "pjrt", "cpu"))
}

# Writes a fake CPU plugin into `dir` and returns its path.
write_plugin <- function(dir, url = plugin_url("cpu")) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(rlang::hash(url), file.path(dir, "hash"))
  path <- file.path(dir, "libpjrt_cpu.so")
  file.create(path)
  path
}

describe("plugin_path()", {
  it("prefers the plugin in the package directory over the user cache", {
    dirs <- local_plugin_dirs()
    pkg_plugin <- write_plugin(dirs[[1L]])
    write_plugin(dirs[[2L]])
    expect_equal(plugin_path("cpu"), pkg_plugin)
  })

  it("uses the plugin in the user cache when the package directory has none", {
    dirs <- local_plugin_dirs()
    cache_plugin <- write_plugin(dirs[[2L]])
    expect_equal(plugin_path("cpu"), cache_plugin)
  })

  it("ignores a plugin downloaded from another URL", {
    dirs <- local_plugin_dirs()
    write_plugin(dirs[[1L]], url = "https://example.com/old-plugin.tar.gz")
    cache_plugin <- write_plugin(dirs[[2L]])
    expect_equal(plugin_path("cpu"), cache_plugin)
  })

  it("downloads into the package directory when it is writable", {
    dirs <- local_plugin_dirs()
    local_mocked_bindings(plugin_download = function(dir, platform) write_plugin(dir))
    expect_equal(plugin_path("cpu"), file.path(dirs[[1L]], "libpjrt_cpu.so"))
  })

  it("downloads into the user cache when the package directory is not writable", {
    dirs <- local_plugin_dirs()
    local_mocked_bindings(
      plugin_download = function(dir, platform) write_plugin(dir),
      dir_writable = function(path) !startsWith(path, dirname(dirname(dirs[[1L]])))
    )
    expect_equal(plugin_path("cpu"), file.path(dirs[[2L]], "libpjrt_cpu.so"))
  })

  it("only uses the user cache when pjrt is loaded with load_all()", {
    dirs <- local_plugin_dirs()
    local_mocked_bindings(installed_package_dir = function() NULL)
    expect_equal(plugin_dirs("cpu"), dirs[[2L]])
  })
})

describe("installed_package_dir()", {
  it("is the package directory when pjrt is installed", {
    local_mocked_bindings(is_dev_package = function(name) FALSE, .package = "pkgload")
    expect_equal(installed_package_dir(), system.file(package = "pjrt"))
  })

  it("is NULL when pjrt is loaded with load_all()", {
    local_mocked_bindings(is_dev_package = function(name) name == "pjrt", .package = "pkgload")
    expect_null(installed_package_dir())
  })
})

describe("plugins_downloaded()", {
  it("finds a plugin in the package directory or the user cache", {
    dirs <- local_plugin_dirs()
    expect_false(plugins_downloaded("cpu"))
    write_plugin(dirs[[1L]])
    expect_true(plugins_downloaded("cpu"))
    unlink(dirs[[1L]], recursive = TRUE)
    write_plugin(dirs[[2L]])
    expect_true(plugins_downloaded("cpu"))
  })
})

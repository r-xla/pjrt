plugin_client_create <- function(plugin, ...) {
  check_plugin(plugin)
  impl_plugin_client_create(plugin)
}

check_plugin <- function(plugin) {
  stopifnot(inherits(plugin, "PJRTPlugin"))
  invisible(NULL)
}

plugin_load <- function() {
  impl_plugin_load(plugin_path())
}

plugin_path <- function() {
  cache_dir <- tools::R_user_dir("pjrt", which = "cache")

  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE)
  }

  plugin_hash_path <- file.path(cache_dir, "hash")

  if (!file.exists(plugin_hash_path)) {
    plugin_download(cache_dir)
  } else {
    plugin_hash <- readLines(plugin_hash_path)
    expected_hash <- rlang::hash(plugin_url())

    if (plugin_hash != expected_hash) {
      message("Plugin hash mismatch. Re-downloading...")
      plugin_download(cache_dir)
    }
  }

  list.files(cache_dir, pattern = "pjrt", full.names = TRUE)
}

plugin_download <- function(cache_dir) {
  plugin_hash_path <- file.path(cache_dir, "hash")

  url <- plugin_url()
  tempfile <- tempfile(fileext = ".tar.gz")
  download.file(url, tempfile)

  plugin_hash <- rlang::hash(url)
  writeLines(plugin_hash, plugin_hash_path)
  utils::untar(tempfile, exdir = cache_dir)
}

plugin_url <- function() {
  if (Sys.getenv("PJRT_PLUGIN_URL") != "") {
    return(Sys.getenv("PJRT_PLUGIN_URL"))
  }

  os <- plugin_os()
  arch <- plugin_arch()
  device <- plugin_device()
  zml_version <- plugin_version()

  glue::glue(
    "https://github.com/zml/pjrt-artifacts/releases/download/v{zml_version}/pjrt-{device}_{os}-{arch}.tar.gz"
  )
}

plugin_version <- function() {
  if (Sys.getenv("PJRT_ZML_ARTIFACT_VERSION") != "") {
    return(Sys.getenv("PJRT_ZML_ARTIFACT_VERSION"))
  }

  "9.0.1"
}

plugin_os <- function() {
  if (Sys.info()[["sysname"]] == "Darwin") {
    return("darwin")
  } else if (Sys.info()[["sysname"]] == "Linux") {
    return("linux")
  } else {
    stop("Unsupported OS: ", Sys.info()[["sysname"]])
  }
}

plugin_device <- function() {
  if (Sys.getenv("PJRT_DEVICE") != "") {
    return(Sys.getenv("PJRT_DEVICE"))
  }

  "cpu"
}

plugin_arch <- function() {
  if (Sys.info()["machine"] == "x86_64") {
    return("amd64")
  } else if (Sys.info()["machine"] == "arm64") {
    return("arm64")
  } else {
    stop("Unsupported architecture: ", .Platform$r_arch)
  }
}

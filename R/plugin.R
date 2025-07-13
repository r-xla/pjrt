the <- new.env(parent = emptyenv())

# Initialize named lists for plugins and clients
the$plugins <- list()
the$clients <- list()

plugin_client_create <- function(plugin, platform, ...) {
  # Check if client already exists for this device
  if (platform %in% names(the$clients)) {
    return(the$clients[[platform]])
  }

  check_plugin(plugin)
  the$clients[[platform]] <- impl_plugin_client_create(plugin)
}

check_plugin <- function(plugin) {
  stopifnot(inherits(plugin, "PJRTPlugin"))
  invisible(NULL)
}

plugin_load <- function(platform) {
  if (platform %in% names(the$plugins)) {
    return(the$plugins[[platform]])
  }

  # Load plugin for the specified device
  the$plugins[[platform]] <- impl_plugin_load(plugin_path(platform))
}

plugin_path <- function(platform) {
  # Create device-specific cache directory
  cache_dir <- tools::R_user_dir("pjrt", which = "cache")

  device_cache_dir <- file.path(cache_dir, platform)

  if (!dir.exists(device_cache_dir)) {
    dir.create(device_cache_dir, recursive = TRUE)
  }

  plugin_hash_path <- file.path(device_cache_dir, "hash")

  if (!file.exists(plugin_hash_path)) {
    plugin_download(device_cache_dir, platform)
  } else {
    plugin_hash <- readLines(plugin_hash_path)
    expected_hash <- rlang::hash(as.character(plugin_url(platform)))

    if (plugin_hash != expected_hash) {
      message("Plugin hash mismatch. Re-downloading...")
      plugin_download(device_cache_dir, platform)
    }
  }

  list.files(device_cache_dir, pattern = "pjrt", full.names = TRUE)
}

plugin_download <- function(cache_dir, platform = NULL) {
  plugin_hash_path <- file.path(cache_dir, "hash")

  url <- plugin_url(platform)
  tempfile <- tempfile(fileext = ".tar.gz")
  utils::download.file(url, tempfile)

  fs::dir_delete(cache_dir)
  fs::dir_create(cache_dir, recurse = TRUE)

  plugin_hash <- rlang::hash(as.character(url))
  writeLines(plugin_hash, plugin_hash_path)

  if (is.function(attr(url, "extract"))) {
    return(attr(url, "extract")(tempfile, cache_dir))
  }

  utils::untar(tempfile, exdir = cache_dir)
}

plugin_url <- function(platform) {
  # Check if plugin already exists for this device
  env_var <- paste0("PJRT_PLUGIN_URL_", toupper(platform))
  if (Sys.getenv(env_var) != "") {
    return(Sys.getenv(env_var))
  }

  os <- plugin_os()
  arch <- plugin_arch()
  zml_version <- plugin_version()

  if (platform == "metal") {
    stopifnot(os == "darwin")
    url <- if (arch == "arm64") {
      "https://files.pythonhosted.org/packages/09/dc/6d8fbfc29d902251cf333414cf7dcfaf4b252a9920c881354584ed36270d/jax_metal-0.1.1-py3-none-macosx_13_0_arm64.whl"
    } else {
      "https://files.pythonhosted.org/packages/87/ec/9bb7f7f0ffd06c3fb89813126b2f698636ac7a4263ed7bdd1ff7d7c94f8f/jax_metal-0.1.1-py3-none-macosx_10_14_x86_64.whl"
    }
    attr(url, "extract") <- function(path, cache_dir) {
      tmp <- tempfile()
      dir.create(tmp)
      utils::unzip(path, exdir = tmp)
      plugin_path <- list.files(
        file.path(tmp, "jax_plugins", "metal_plugin"),
        pattern = "*.dylib",
        full.names = TRUE
      )
      fs::file_move(plugin_path, cache_dir)
    }
    return(url)
  }

  sprintf(
    "https://github.com/zml/pjrt-artifacts/releases/download/v%s/pjrt-%s_%s-%s.tar.gz",
    zml_version, platform, os, arch
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

plugin_arch <- function() {
  if (Sys.info()["machine"] == "x86_64") {
    return("amd64")
  } else if (Sys.info()["machine"] == "arm64") {
    return("arm64")
  } else {
    stop("Unsupported architecture: ", .Platform$r_arch)
  }
}

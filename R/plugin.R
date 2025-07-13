the <- new.env(parent = emptyenv())

the$plugins <- list()
the$clients <- list()

plugin_client_create <- function(plugin, platform_name = NULL, ...) {
  check_plugin(plugin)

  # If no platform name specified, we'll create the client first and then get the name
  if (is.null(platform_name)) {
    # Create the client
    client <- impl_plugin_client_create(plugin)
    # Get the platform name from the client
    platform_name <- tolower(impl_client_platform_name(client))
    # Store the client with the correct platform name
    the$clients[[platform_name]] <- client
    return(client)
  }

  # Check if we already have a client for this platform
  if (platform_name %in% names(the$clients)) {
    return(the$clients[[platform_name]])
  }

  # Create new client and store it
  the$clients[[platform_name]] <- impl_plugin_client_create(plugin)
  the$clients[[platform_name]]
}

check_plugin <- function(plugin) {
  stopifnot(inherits(plugin, "PJRTPlugin"))
  invisible(NULL)
}

#' @title Load a PJRT Plugin
#' @description
#' Load a PJRT plugin for the specified device. If the plugin is already
#' loaded, it will be returned instead of loading it again.
#'
#' @param device (`character(1)`)\cr
#'   The device type (e.g., "cpu", "cuda", "metal"). If NULL, uses the default
#'   device from environment variables or "cpu".
#' @return `PJRTPlugin`
#' @export
plugin_load <- function(device = NULL) {
  # If no device specified, use the default
  if (is.null(device)) {
    device <- plugin_device()
  }

  # Check if we already have this plugin loaded
  if (device %in% names(the$plugins)) {
    return(the$plugins[[device]])
  }

  # Load the plugin for the specified device
  the$plugins[[device]] <- impl_plugin_load(plugin_path(device))
  the$plugins[[device]]
}

plugin_path <- function(device = NULL) {
  # If no device specified, use the default
  if (is.null(device)) {
    device <- plugin_device()
  }

  cache_dir <- tools::R_user_dir("pjrt", which = "cache")
  device_cache_dir <- file.path(cache_dir, device)

  if (!dir.exists(device_cache_dir)) {
    dir.create(device_cache_dir, recursive = TRUE)
  }

  plugin_hash_path <- file.path(device_cache_dir, "hash")

  if (!file.exists(plugin_hash_path)) {
    plugin_download(device_cache_dir, device)
  } else {
    plugin_hash <- readLines(plugin_hash_path)
    expected_hash <- rlang::hash(as.character(plugin_url(device)))

    if (plugin_hash != expected_hash) {
      message("Plugin hash mismatch. Re-downloading...")
      plugin_download(device_cache_dir, device)
    }
  }

  list.files(device_cache_dir, pattern = "pjrt", full.names = TRUE)
}

plugin_download <- function(cache_dir, device = NULL) {
  # If no device specified, use the default
  if (is.null(device)) {
    device <- plugin_device()
  }

  plugin_hash_path <- file.path(cache_dir, "hash")

  url <- plugin_url(device)
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

plugin_url <- function(device = NULL) {
  # If no device specified, use the default
  if (is.null(device)) {
    device <- plugin_device()
  }

  if (Sys.getenv("PJRT_PLUGIN_URL") != "") {
    return(Sys.getenv("PJRT_PLUGIN_URL"))
  }

  os <- plugin_os()
  arch <- plugin_arch()
  zml_version <- plugin_version()

  if (device == "metal") {
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

#' @title Get a PJRT Client by Platform Name
#' @description
#' Get a PJRT client for the specified platform. If no platform is specified,
#' returns the default client. If a client for the platform already exists,
#' it will be returned; otherwise, a new one will be created.
#'
#' @param platform_name (`character(1)`)\cr
#'   The platform name (e.g., "cpu", "gpu", "metal"). If NULL, uses the default
#'   platform.
#' @return `PJRTClient`
#' @export
get_client <- function(platform_name = NULL) {
  if (is.null(platform_name)) {
    # Use default device
    device <- plugin_device()
    plugin <- plugin_load(device)
    return(plugin_client_create(plugin))
  }

  # Check if we already have a client for this platform
  if (platform_name %in% names(the$clients)) {
    return(the$clients[[platform_name]])
  }

  # Try to find a plugin that matches this platform
  # For now, we'll try common mappings
  device_mapping <- list(
    "cpu" = "cpu",
    "gpu" = "cuda",  # Assuming CUDA for GPU
    "metal" = "metal"
  )

  if (platform_name %in% names(device_mapping)) {
    device <- device_mapping[[platform_name]]
    plugin <- plugin_load(device)
    return(plugin_client_create(plugin, platform_name))
  }

  # If we can't find a mapping, try to load a plugin with the platform name as device
  tryCatch({
    plugin <- plugin_load(platform_name)
    return(plugin_client_create(plugin, platform_name))
  }, error = function(e) {
    stop("Could not find or create client for platform: ", platform_name)
  })
}

#' @title List Available Platforms
#' @description
#' List all platforms for which clients have been created.
#'
#' @return A character vector of platform names.
#' @export
list_platforms <- function() {
  names(the$clients)
}

#' @title Clear All Plugins and Clients
#' @description
#' Clear all loaded plugins and clients from memory. This is useful for
#' testing or when you want to start fresh.
#'
#' @return NULL (invisibly)
#' @export
clear_plugins_and_clients <- function() {
  the$plugins <- list()
  the$clients <- list()
  invisible(NULL)
}


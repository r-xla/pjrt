the <- new.env(parent = emptyenv())

the$plugins <- list()
the$clients <- list()

plugin_client_create <- function(plugin, platform, ...) {
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

#' @title Create PJRT Plugin
#' @description
#' Create a PJRT plugin for a specific platform.
#'
#' @section Extrators:
#' * [`plugin_attributes()`] for a named `list()` of attributes.
#'
#' @param platform (`character(1)`)\cr
#'   Platform name (e.g., "cpu", "cuda", "metal").
#' @return `PJRTPlugin`
#' @export
pjrt_plugin <- function(platform) {
  if (platform %in% names(the$plugins)) {
    return(the$plugins[[platform]])
  }

  the$plugins[[platform]] <- impl_plugin_load(plugin_path(platform))
}

plugin_path <- function(platform) {
  envvar <- Sys.getenv(paste0("PJRT_PLUGIN_PATH_", toupper(platform)), "")
  if (envvar != "") {
    return(envvar)
  }

  cache_dir <- tools::R_user_dir("pjrt", which = "cache")

  platform_cache_dir <- file.path(cache_dir, platform)

  if (!dir.exists(platform_cache_dir)) {
    dir.create(platform_cache_dir, recursive = TRUE)
  }

  plugin_hash_path <- file.path(platform_cache_dir, "hash")

  if (!file.exists(plugin_hash_path)) {
    plugin_download(platform_cache_dir, platform)
  } else {
    plugin_hash <- readLines(plugin_hash_path)
    expected_hash <- rlang::hash(as.character(plugin_url(platform)))

    if (plugin_hash != expected_hash) {
      message("Plugin hash mismatch. Re-downloading...")
      plugin_download(platform_cache_dir, platform)
    }
  }

  list.files(platform_cache_dir, pattern = "pjrt", full.names = TRUE)
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
    zml_version,
    platform,
    os,
    arch
  )
}

plugin_version <- function() {
  if (Sys.getenv("PJRT_ZML_ARTIFACT_VERSION") != "") {
    return(Sys.getenv("PJRT_ZML_ARTIFACT_VERSION"))
  }

  "11.0.0"
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

pjrt_api_version <- function(plugin = pjrt_plugin()) {
  v <- impl_plugin_pjrt_api_version(plugin)
  list(major = v[[1]], minor = v[[2]])
}

#' @title Get Plugin Attributes
#' @description
#' Get the attributes of a PJRT plugin.
#' This commonly includes:
#' - `xla_version`
#' - `stablehlo_current_version`
#' - `stablehlo_minimum_version`
#'
#' But the implementation depends on the plugin.
#'
#' @param plugin (`PJRTPlugin` | `character(1)`)\cr
#'   The plugin (or platform name) to get the attributes of.
#' @return named `list()`
#' @export
#' @examplesIf Sys.info()["sysname"] != "Windows"
#' plugin_attributes("cpu")
plugin_attributes <- function(plugin) {
  plugin <- as_pjrt_plugin(plugin)
  impl_plugin_attributes(plugin)
}

#' @title Convert to PJRT Plugin
#' @description
#' Convert a platform name to a PJRT plugin or verify that an object is already a plugin.
#'
#' @param x (any)\cr
#'   Object to convert to a PJRT plugin. Currently supports `PJRTPlugin` and `character(1)`.
#' @return `PJRTPlugin`
#' @export
as_pjrt_plugin <- function(x) {
  if (checkmate::test_string(x)) {
    pjrt_plugin(x)
  } else if (inherits(x, "PJRTPlugin")) {
    x
  } else {
    stop("Invalid plugin: ", class(x))
  }
}

the <- hashtab()

the[["plugins"]] <- hashtab()
the[["clients"]] <- hashtab()
the[["config"]] <- list(
  cpu_device_count = 1L
)

#' @title Create PJRT Client
#' @description
#' Create a PJRT client for a specific plugin and platform.
#'
#' @param plugin (`PJRTPlugin`)\cr
#'   The plugin to create a client for.
#' @param platform (`character(1)`)\cr
#'   The platform to create a client for.
#' @param options (`list()`)\cr
#'   Additional options to pass to the client.
#' @return `PJRTClient`
#' @export
plugin_client_create <- function(plugin, platform, options = list()) {
  client <- the[["clients"]][[platform]]
  if (!is.null(client)) {
    if (length(options)) {
      cli_abort("Can only specify client options the first time you create a client.")
    }
    return(client)
  }
  opts <- default_client_options(platform)
  if (length(options)) {
    opts[names(options)] <- options
  }

  check_plugin(plugin)

  opts <- if (length(opts) > 0) opts else NULL
  lapply(opts, function(x) {
    if (!is.integer(x) || length(x) != 1) {
      cli_abort("Client options must be integers of length 1")
    }
  })
  # Otherwise, we get startup message w.r.t. the number of cpu devices
  client <- withr::with_envvar(c(TF_CPP_MIN_LOG_LEVEL = "1"), {
    impl_plugin_client_create(plugin, opts)
  })
  the[["clients"]][[platform]] <- client
  # in order to go from device -> client, we go through the platform name, which might
  # not be the same as the "cuda" string, but might be "nvidia h100" etc.
  the[["clients"]][[platform(devices(client)[[1L]])]] <- client
  client
}

check_plugin <- function(plugin) {
  stopifnot(inherits(plugin, "PJRTPlugin"))
  invisible(NULL)
}

#' @title Create PJRT Plugin
#' @description
#' Create a PJRT plugin for a specific platform.
#'
#' @section Extractors:
#' * [`plugin_attributes()`] -> `list()`: for the attributes of the plugin.
#'
#' @param platform (`character(1)`)\cr
#'   Platform name (e.g., "cpu", "cuda", "metal").
#' @return `PJRTPlugin`
#' @examplesIf plugin_is_downloaded("cpu")
#' plugin <- pjrt_plugin("cpu")
#' plugin
#' @export
pjrt_plugin <- function(platform) {
  if (platform %in% names(the[["plugins"]])) {
    return(the[["plugins"]][[platform]])
  }

  plugin <- impl_plugin_load(plugin_path(platform))
  attributes(plugin) <- list(platform = platform)

  # register the print handler
  if (!ffi_register_print_tensor(plugin)) {
    cli::cli_warn(c(
      x = "Unable to register the print tensor handler.",
      i = "Using the {.fn print_tensor} custom call won't be possible."
    ))
  }

  class(plugin) <- "PJRTPlugin"
  the[["plugins"]][[platform]] <- plugin
  plugin
}

platform_cache_dir <- function(platform) {
  file.path(tools::R_user_dir("pjrt", which = "cache"), platform)
}

#' @title Check if Plugin is Downloaded
#' @description
#' Check if a plugin is downloaded.
#'
#' @param platform (`character(1)`)\cr
#'   Platform name.
#' @return `logical(1)`
#' @examplesIf plugin_is_downloaded("cpu")
#' # Check if CPU plugin is downloaded
#' plugin_is_downloaded("cpu")
#' @export
plugin_is_downloaded <- function(platform = NULL) {
  platform <- platform %||% Sys.getenv("PJRT_PLATFORM", "cpu")
  dir.exists(platform_cache_dir(platform))
}

plugin_path <- function(platform) {
  if (!(platform %in% c("cpu", "cuda", "metal"))) {
    cli_abort(c(
      i = "Invalid platform: {.val {platform}}",
      x = "Must be one of: {.val cpu}, {.val cuda}, {.val metal}"
    ))
  }
  envvar <- Sys.getenv(paste0("PJRT_PLUGIN_PATH_", toupper(platform)), "")
  if (envvar != "") {
    return(envvar)
  }

  platform_cache_dir <- platform_cache_dir(platform)

  plugin_hash_path <- file.path(platform_cache_dir, "hash")

  if (!dir.exists(platform_cache_dir)) {
    plugin_download(platform_cache_dir, platform)
  } else {
    plugin_hash <- readLines(plugin_hash_path)
    expected_hash <- rlang::hash(as.character(plugin_url(platform)))

    if (plugin_hash != expected_hash) {
      plugin_download(platform_cache_dir, platform)
    }
  }

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

  if (dir.exists(cache_dir)) {
    fs::dir_delete(cache_dir)
  }
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
      "https://files.pythonhosted.org/packages/09/dc/6d8fbfc29d902251cf333414cf7dcfaf4b252a9920c881354584ed36270d/jax_metal-0.1.1-py3-none-macosx_13_0_arm64.whl" # nolint
    } else {
      "https://files.pythonhosted.org/packages/87/ec/9bb7f7f0ffd06c3fb89813126b2f698636ac7a4263ed7bdd1ff7d7c94f8f/jax_metal-0.1.1-py3-none-macosx_10_14_x86_64.whl" # nolint
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

  if (os == "windows") {
    if (arch != "amd64") {
      cli_abort(
        "Unsupported architecture for Windows: ",
        arch,
        ". Only 'amd64' is supported."
      )
    }

    # on windows download from our pre-built artifacts
    # TODO make this versioned.
    url <- "https://github.com/r-xla/pjrt-builds/releases/download/pjrt/pjrt-a4df377-windows-x86_64.zip"
    # windows files are zipped
    attr(url, "extract") <- function(path, cache_dir) {
      tmp <- tempfile()
      dir.create(tmp)
      utils::unzip(path, exdir = tmp)
      plugin_path <- list.files(
        tmp,
        pattern = "*.dll",
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

  "14.0.1"
}

plugin_os <- function() {
  if (Sys.info()[["sysname"]] == "Darwin") {
    return("darwin")
  } else if (Sys.info()[["sysname"]] == "Linux") {
    return("linux")
  } else if (Sys.info()[["sysname"]] == "Windows") {
    return("windows")
  } else {
    cli_abort("Unsupported OS: ", Sys.info()[["sysname"]])
  }
}

plugin_arch <- function() {
  if (Sys.info()["machine"] == "x86_64") {
    return("amd64")
  } else if (Sys.info()["machine"] == "arm64") {
    return("arm64")
  } else if (.Platform$r_arch == "x64") {
    return("amd64")
  } else if (.Platform$r_arch == "arm64") {
    return("arm64")
  } else {
    cli_abort("Unsupported architecture: ", .Platform$r_arch)
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
#' @examplesIf plugin_is_downloaded("cpu")
#' plugin_attributes("cpu")
#' @export
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
#' @examplesIf plugin_is_downloaded("cpu")
#' # Convert from platform name
#' plugin <- as_pjrt_plugin("cpu")
#' plugin
#' @export
as_pjrt_plugin <- function(x) {
  if (checkmate::test_string(x)) {
    pjrt_plugin(x)
  } else if (inherits(x, "PJRTPlugin")) {
    x
  } else {
    cli_abort("Invalid plugin: ", class(x))
  }
}

#' @export
print.PJRTPlugin <- function(x, ...) {
  cat(sprintf("<PJRTPlugin:%s>\n", attr(x, "platform")))
  invisible(x)
}

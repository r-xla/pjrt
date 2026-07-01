the <- new.env(parent = emptyenv())

the[["plugins"]] <- new.env(parent = emptyenv())
the[["clients"]] <- new.env(parent = emptyenv())
the[["devices"]] <- new.env(parent = emptyenv())
# Memoization caches keyed by a device's string representation. `platforms`
# caches the parsed platform name (avoids a regex per call); `canonical_devices`
# maps a device string to the canonical cached `PJRTDevice` xptr (avoids
# rescanning the client's device list per call). Both domains are tiny.
the[["platforms"]] <- new.env(parent = emptyenv())
the[["canonical_devices"]] <- new.env(parent = emptyenv())
the[["custom_calls"]] <- list()
the[["config"]] <- list(
  cpu_device_count = 1L,
  cuda_r_package = "cuda12.8"
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
  client <- impl_plugin_client_create(plugin, opts)
  the[["clients"]][[platform]] <- client
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
#' @examplesIf plugins_downloaded("cpu")
#' plugin <- pjrt_plugin("cpu")
#' plugin
#' @export
pjrt_plugin <- function(platform) {
  if (exists(platform, envir = the[["plugins"]], inherits = FALSE)) {
    return(the[["plugins"]][[platform]])
  }

  path <- plugin_path(platform)

  if (platform == "cuda") {
    setup_cuda_env()
  }

  plugin <- tryCatch(
    impl_plugin_load(path),
    error = function(e) {
      if (platform == "cuda" && Sys.info()[["sysname"]] == "Linux") {
        cuda_pkg <- Sys.getenv("PJRT_CUDA_R_PACKAGE", cuda_r_package())
        if (!requireNamespace(cuda_pkg, quietly = TRUE)) {
          cli::cli_abort(c(
            conditionMessage(e),
            i = "CUDA R package {.pkg {cuda_pkg}} is not installed.",
            i = "Install it with {.code install.packages(\"{cuda_pkg}\", repos = \"https://mlverse.r-universe.dev\")}."
          ))
        }
      }
      stop(e)
    }
  )
  attributes(plugin) <- list(platform = platform)

  if (platform != "metal") {
    drain_custom_calls(plugin, platform)
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
#' Check if one more more plugin is already downloaded.
#'
#' @param platforms (`character()`)\cr
#'   Platform names.
#' @return `logical(1)`
#' @examplesIf plugins_downloaded("cpu")
#' # Check if CPU plugin is downloaded
#' plugins_downloaded("cpu")
#' @export
plugins_downloaded <- function(platforms = NULL) {
  platforms <- platforms %||% Sys.getenv("PJRT_PLATFORM", "cpu")
  all(vapply(platforms, \(p) dir.exists(platform_cache_dir(p)), logical(1)))
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

# Ask the user for permission before downloading a PJRT plugin, mirroring the
# behaviour of torch's auto-install prompt. The `PJRT_INSTALL` environment
# variable overrides the prompt:
#   - PJRT_INSTALL=1  download without asking (e.g. CI, scripts, Docker builds)
#   - PJRT_INSTALL=0  never download; abort with instructions instead
# When `PJRT_INSTALL` is unset we ask in an interactive session and abort in a
# non-interactive one (where there is no terminal to ask on), so a batch job or
# script never triggers a surprise download. This is never reached during
# `R CMD check` because examples are guarded behind `plugins_downloaded()` and
# the test suite only runs when `PJRT_TEST=1` (see tests/testthat.R).
confirm_plugin_install <- function(platform, url) {
  install <- Sys.getenv("PJRT_INSTALL", unset = "")

  if (install == "1") {
    return(invisible(TRUE))
  }

  if (install == "0") {
    cli_abort(c(
      "The {.val {platform}} PJRT plugin needs to be downloaded but automatic downloads are disabled.",
      i = "{.envvar PJRT_INSTALL} is set to {.val 0}.",
      i = "Set {.envvar PJRT_INSTALL} to {.val 1} to allow the download, or set {.envvar PJRT_PLUGIN_PATH_{toupper(platform)}} to a local plugin file."
    ))
  }

  # PJRT_INSTALL unset: only download if we can ask and the user agrees.
  if (!interactive()) {
    cli_abort(c(
      "The {.val {platform}} PJRT plugin needs to be downloaded for {.pkg pjrt} to work.",
      i = "Automatic downloads are not performed in non-interactive sessions.",
      i = "Set {.envvar PJRT_INSTALL} to {.val 1} to allow the download, or set {.envvar PJRT_PLUGIN_PATH_{toupper(platform)}} to a local plugin file."
    ))
  }

  cli::cli_inform(c(
    "The {.val {platform}} PJRT plugin needs to be {.strong downloaded} for {.pkg pjrt} to work.",
    i = "It will be downloaded from {.url {url}} and cached in {.path {platform_cache_dir(platform)}}.",
    i = "Set {.envvar PJRT_INSTALL} to {.val 1} to skip this prompt in the future."
  ))
  response <- utils::askYesNo("Do you want to download it now?")
  if (is.na(response) || !response) {
    cli_abort("Download of the {.val {platform}} PJRT plugin was declined.")
  }

  invisible(TRUE)
}

plugin_download <- function(cache_dir, platform = NULL) {
  plugin_hash_path <- file.path(cache_dir, "hash")

  url <- plugin_url(platform)
  confirm_plugin_install(platform, url)
  tempfile <- tempfile(fileext = ".tar.gz")
  cli::cli_inform("Downloading PJRT plugin from {.url {url}}")
  withr::local_options(timeout = max(getOption("timeout"), 600L))
  utils::download.file(url, tempfile, mode = "wb", quiet = FALSE)

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

  if (os == "linux" && arch == "aarch64") {
    # on linux arm download from our pre-built artifacts
    url <- "https://github.com/r-xla/pjrt-builds/releases/download/pjrt/pjrt-a4df377-linux-aarch64.tar.gz"
    return(url)
  }

  if (platform == "cuda" && !(os == "linux" && arch == "amd64")) {
    cli_abort(c(
      "The CUDA PJRT plugin is only available for Linux x86_64.",
      i = "Detected platform: {.val {os}-{arch}}.",
      i = "To override, set the {.envvar PJRT_PLUGIN_URL_CUDA} environment variable to a plugin URL, or {.envvar PJRT_PLUGIN_PATH_CUDA} to a local plugin file."
    ))
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

# nocov start
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
  } else if (Sys.info()["machine"] == "aarch64") {
    return("aarch64")
  } else if (.Platform$r_arch == "x64") {
    return("amd64")
  } else if (.Platform$r_arch == "arm64") {
    return("arm64")
  } else {
    cli_abort("Unsupported architecture: ", .Platform$r_arch)
  }
}
# nocov end

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
#' @examplesIf plugins_downloaded("cpu")
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
#' @examplesIf plugins_downloaded("cpu")
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

cuda_r_package <- function() {
  the[["config"]][["cuda_r_package"]]
}

# Discover installed cuda{X.Y} packages and pre-load CUDA shared libraries
# into the process so dlopen can find them when the PJRT plugin loads.
# Also adds ptxas to PATH for PTX compilation.
# Called once, right before the plugin is loaded.
setup_cuda_env <- function() {
  cuda_pkg <- Sys.getenv("PJRT_CUDA_R_PACKAGE", cuda_r_package())
  if (cuda_pkg != cuda_r_package()) {
    pjrt_debug("PJRT_CUDA_R_PACKAGE set to {.val {cuda_pkg}} (default: {cuda_r_package()})")
  }

  if (!requireNamespace(cuda_pkg, quietly = TRUE)) {
    return(invisible(NULL))
  }

  # Pre-load all .so files from the CUDA lib dir with RTLD_GLOBAL so they're
  # available when the PJRT plugin resolves its NEEDED entries via dlopen.
  # We can't use LD_LIBRARY_PATH because it's read once at process startup.
  lib_dir <- tryCatch(
    getExportedValue(cuda_pkg, "lib_path")(),
    error = function(e) {
      cli::cli_warn("Failed to get lib_path from {cuda_pkg}: {conditionMessage(e)}")
      NULL
    }
  )
  if (!is.null(lib_dir) && dir.exists(lib_dir)) {
    so_files <- list.files(lib_dir, pattern = "\\.so[.0-9]*$", full.names = TRUE)
    pjrt_debug("Loading {length(so_files)} .so files from {.path {lib_dir}}")
    for (so in so_files) {
      tryCatch(
        dyn.load(so, local = FALSE, now = TRUE),
        error = function(e) {
          pjrt_debug("dyn.load failed for {.path {so}}: {conditionMessage(e)}")
        }
      )
    }
  }

  # Add nvcc bin dir (ptxas) to PATH for PTX compilation
  # and set XLA_FLAGS to point to the CUDA data dir (for nvvm/libdevice)
  tryCatch(
    {
      cuda_nvcc_path <- getExportedValue(cuda_pkg, "cuda_path")("nvcc")
      nvcc_bin <- file.path(cuda_nvcc_path, "bin")
      if (dir.exists(nvcc_bin)) {
        pjrt_debug("Adding nvcc bin to PATH: {.path {nvcc_bin}}")
        current_path <- Sys.getenv("PATH", "")
        Sys.setenv(PATH = paste(nvcc_bin, current_path, sep = ":"))
      } else {
        pjrt_debug("nvcc bin dir does not exist: {.path {nvcc_bin}}")
      }
      # XLA needs nvvm/libdevice/libdevice.10.bc — point it to the nvcc dir
      current_flags <- Sys.getenv("XLA_FLAGS", "")
      xla_cuda_dir <- paste0("--xla_gpu_cuda_data_dir=", cuda_nvcc_path)
      if (!grepl("xla_gpu_cuda_data_dir", current_flags)) {
        new_flags <- if (nzchar(current_flags)) paste(current_flags, xla_cuda_dir) else xla_cuda_dir
        pjrt_debug("Setting XLA_FLAGS: {new_flags}")
        Sys.setenv(XLA_FLAGS = new_flags)
      } else {
        pjrt_debug("xla_gpu_cuda_data_dir already set in XLA_FLAGS, skipping")
      }
    },
    error = function(e) {
      pjrt_debug("Failed to configure nvcc/XLA_FLAGS: {conditionMessage(e)}")
    }
  )

  invisible(NULL)
}

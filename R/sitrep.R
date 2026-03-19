#' @title Situation Report
#' @description
#' Get a situation report on the pjrt package installation and configuration.
#' This function checks system information, package versions, plugin availability,
#' and device status to help diagnose configuration issues.
#'
#' @return Invisibly returns `NULL`. Called for its side effect of printing
#'   a diagnostic report.
#' @examplesIf plugin_is_downloaded("cpu")
#' pjrt_sitrep()
#' @export
pjrt_sitrep <- function() {
  all_platforms <- c("cpu", "cuda", "metal")
  downloaded <- Filter(plugin_is_downloaded, all_platforms)
  plugin_info <- collect_plugin_info(downloaded)

  cli::cli_h1("{.pkg pjrt} situation report")
  sitrep_system_info()
  sitrep_plugins(plugin_info)
  sitrep_env_vars()

  invisible(NULL)
}

sitrep_system_info <- function() {
  cli::cli_h2("System information")
  cli::cli_ul()
  cli::cli_li("OS: {plugin_os()}")

  cli::cli_li("arch: {plugin_arch()}")
  cli::cli_li("R version: {getRversion()}")
  cli::cli_end()
}

sitrep_env_vars <- function() {
  cli::cli_h2("Environment variables")
  env_vars <- c(
    "PJRT_PLATFORM",
    "PJRT_CPU_DEVICE_COUNT",
    "PJRT_ZML_ARTIFACT_VERSION",
    "PJRT_PLUGIN_PATH_CPU",
    "PJRT_PLUGIN_PATH_CUDA",
    "PJRT_PLUGIN_PATH_METAL",
    "PJRT_PLUGIN_URL_CPU",
    "PJRT_PLUGIN_URL_CUDA",
    "PJRT_PLUGIN_URL_METAL",
    "TF_CPP_MIN_LOG_LEVEL",
    "XLA_FLAGS"
  )
  set_vars <- Filter(function(v) nzchar(Sys.getenv(v)), env_vars)
  if (length(set_vars) == 0) {
    cli::cli_alert_info("No pjrt-related environment variables are set.")
  } else {
    cli::cli_ul()
    for (v in set_vars) {
      cli::cli_li("{.envvar {v}}: {.val {Sys.getenv(v)}}")
    }
    cli::cli_end()
  }
}

collect_plugin_info <- function(platforms) {
  lapply(platforms, function(p) {
    info <- list(platform = p)
    tryCatch({
      pjrt_client(p)
      info$ok <- TRUE
    }, error = function(e) {
      info$error <<- conditionMessage(e)
    })
    info
  })
}

sitrep_plugins <- function(plugin_info) {
  cli::cli_h2("Plugins")
  if (length(plugin_info) == 0) {
    cli::cli_alert_warning("No plugins downloaded.")
    return(invisible(NULL))
  }
  for (info in plugin_info) {
    p <- info$platform
    if (!is.null(info$error)) {
      cli::cli_alert_danger("{.val {p}}: {info$error}")
      next
    }
    cli::cli_alert_success("{.val {p}}: plugin found")
    if (p == "cuda") {
      sitrep_cuda_versions()
    }
  }
}

sitrep_cuda_versions <- function() {
  cuda <- detect_cuda_version()
  recommended <- pjrt_cuda_versions$cuda
  if (is.null(cuda)) {
    cli::cli_alert_danger("CUDA: not found (recommended: {recommended})")
  } else {
    cli::cli_alert_success("CUDA: {cuda} (recommended: {recommended})")
  }
}

detect_cuda_version <- function() {
  tryCatch({
    out <- system2("nvcc", "--version", stdout = TRUE, stderr = TRUE)
    m <- regmatches(out, regexpr("release [0-9]+\\.[0-9]+(\\.[0-9]+)?", out))
    m <- m[nzchar(m)]
    if (length(m)) sub("release ", "", m[[1]]) else NULL
  }, error = function(e) NULL)
}

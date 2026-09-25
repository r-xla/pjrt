# Prebuilt CUDA kernels.
#
# A package ships CUDA kernels as source in `inst/cuda/`, and creates their
# modules with `pjrt_cuda_module(..., package = pkgname)` in its `.onLoad()`.
# For each release, `pjrt_cuda_build_kernels()` compiles those modules ahead
# of time into a tarball of cubins (one per GPU architecture) and PTX, which
# is attached to the package's release and named by the package's DESCRIPTION
# field `Config/pjrt/cuda-kernels`.
#
# When the CUDA plugin loads, pjrt downloads each package's tarball once into
# its cache, verifies it, and attaches the images to the modules, so a kernel
# runs without being compiled on the user's machine. Without a tarball (a
# development version, no network, a GPU architecture it does not cover) the
# modules are compiled with NVRTC instead, as any other source module is.

#' @title Build Prebuilt CUDA Kernels
#' @description
#' Compiles the CUDA modules a package ships ahead of time, into the tarball
#' that pjrt downloads in place of compiling them on the user's machine.
#'
#' The modules are the ones the package creates with
#' `pjrt_cuda_module(..., package = pkgname)` when it is loaded, so they are
#' built exactly as they are used. Compiling needs NVRTC, from the CUDA R
#' package the CUDA plugin uses, but no GPU.
#'
#' Attach the tarball to the package's release, and name its URL in the
#' package's DESCRIPTION field `Config/pjrt/cuda-kernels`, where `{version}`
#' stands for the package version, e.g.
#' `https://github.com/org/pkg/releases/download/v{version}/pkg-cuda-kernels-{version}.tar.gz`.
#' @param package (`character(1)`)\cr
#'   The package, which must be installed.
#' @param dest (`character(1)`)\cr
#'   The directory to write the tarball to.
#' @param targets (`character()`)\cr
#'   What to compile each module for: `"sm_XY"` makes a cubin for GPUs of
#'   compute capability X.Y (which also runs on later X.* ones), and
#'   `"compute_XY"` PTX, which the driver compiles for any later GPU.
#' @return (`character(1)`)\cr
#'   The path of the tarball, invisibly.
#' @export
pjrt_cuda_build_kernels <- function(package, dest = ".", targets = cuda_default_targets()) {
  checkmate::assert_string(package)
  checkmate::assert_directory_exists(dest)
  checkmate::assert_character(targets, pattern = "^(sm|compute)_[0-9]+$", min.len = 1L, any.missing = FALSE)
  cuda_load_nvrtc()
  loadNamespace(package)
  modules <- the[["cuda_modules"]][[package]]
  if (!length(modules)) {
    cli_abort(
      "{.pkg {package}} creates no CUDA modules with {.code pjrt_cuda_module(..., package = {.str {package}})}."
    )
  }

  version <- as.character(utils::packageVersion(package))
  name <- sprintf("%s-cuda-kernels-%s", package, version)
  root <- file.path(tempfile(), name)
  dir.create(root, recursive = TRUE)

  rows <- lapply(modules, function(mod) {
    dir.create(file.path(root, mod$id))
    built <- lapply(targets, function(target) {
      cli::cli_inform("Compiling {.file {mod$filename}} for {.val {target}}.")
      impl_cuda_compile(mod$code, mod$filename, mod$options, mod$kernels, target)
    })
    lowered <- built[[1L]]$lowered
    utils::write.table(
      data.frame(expression = names(lowered), lowered = unname(lowered)),
      file.path(root, mod$id, "names.tsv"),
      sep = "\t",
      quote = FALSE,
      row.names = FALSE
    )
    files <- file.path(mod$id, paste0(targets, ifelse(startsWith(targets, "sm_"), ".cubin", ".ptx")))
    Map(function(b, f) writeBin(b$image, file.path(root, f)), built, files)
    data.frame(Module = mod$id, Target = targets, File = files, MD5 = unname(tools::md5sum(file.path(root, files))))
  })
  write.dcf(do.call(rbind, rows), file.path(root, "manifest.dcf"))

  tarball <- file.path(normalizePath(dest), paste0(name, ".tar.gz"))
  withr::with_dir(dirname(root), utils::tar(tarball, name, compression = "gzip", tar = "internal"))
  cli::cli_inform(c(v = "Wrote {.file {tarball}}."))
  invisible(tarball)
}

cuda_default_targets <- function() {
  c("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120", "compute_120")
}

# Makes NVRTC from the CUDA R package loadable without starting the plugin.
cuda_load_nvrtc <- function() {
  cuda_pkg <- Sys.getenv("PJRT_CUDA_R_PACKAGE", cuda_r_package())
  if (!requireNamespace(cuda_pkg, quietly = TRUE)) {
    cli_abort("Compiling CUDA kernels needs NVRTC from the {.pkg {cuda_pkg}} package, which is not installed.")
  }
  lib_dir <- getExportedValue(cuda_pkg, "lib_path")()
  for (so in list.files(lib_dir, pattern = "^libnvrtc.*\\.so[.0-9]*$", full.names = TRUE)) {
    dyn.load(so, local = FALSE, now = TRUE)
  }
}

# Remembers a module a package ships, and attaches its prebuilt images right
# away when the CUDA plugin is already running.
cuda_record_module <- function(module, package) {
  the[["cuda_modules"]][[package]][[module$id]] <- module
  if (!is.null(the[["plugins"]][["cuda"]])) {
    cuda_attach_prebuilt(package)
  }
}

# Called when the CUDA plugin loads.
cuda_attach_prebuilt_all <- function() {
  for (package in names(the[["cuda_modules"]])) {
    cuda_attach_prebuilt(package)
  }
}

# Attaches the prebuilt images of `package`'s modules, downloading them first
# if needed. Never fails: without them, modules are compiled from source.
cuda_attach_prebuilt <- function(package) {
  # pjrt's own kernels belong to its CUDA plugin: agreeing to download that
  # is agreeing to download them
  confirmed <- package == "pjrt" &&
    Sys.getenv("PJRT_INSTALL") != "0" &&
    plugins_downloaded("cuda")
  tryCatch(
    {
      dir <- cuda_prebuilt_fetch(package, confirmed = confirmed)
      if (is.null(dir)) {
        return(invisible(FALSE))
      }
      manifest <- as.data.frame(read.dcf(file.path(dir, "manifest.dcf")), stringsAsFactors = FALSE)
      for (mod in the[["cuda_modules"]][[package]]) {
        rows <- manifest[manifest$Module == mod$id, , drop = FALSE]
        if (!nrow(rows)) {
          # the source changed since the tarball was built
          pjrt_debug(
            "No prebuilt images of CUDA module {.val {mod$id}} ({.file {mod$filename}}) in {.pkg {package}}'s release."
          )
          next
        }
        names_tsv <- utils::read.delim(file.path(dir, mod$id, "names.tsv"), colClasses = "character")
        images <- lapply(file.path(dir, rows$File), function(f) readBin(f, "raw", file.size(f)))
        impl_cuda_module_add_prebuilt(
          mod$id,
          rows$Target,
          images,
          stats::setNames(names_tsv$lowered, names_tsv$expression)
        )
      }
      invisible(TRUE)
    },
    error = function(e) {
      pjrt_debug("Not using prebuilt CUDA kernels of {.pkg {package}}: {conditionMessage(e)}")
      invisible(FALSE)
    }
  )
}

# The URL of `package`'s prebuilt kernels, or NULL if it names none. The
# environment variable PJRT_CUDA_KERNELS_URL_<PACKAGE> overrides the
# DESCRIPTION field.
cuda_kernels_url <- function(package) {
  env <- Sys.getenv(paste0("PJRT_CUDA_KERNELS_URL_", toupper(gsub(".", "_", package, fixed = TRUE))), "")
  template <- if (nzchar(env)) env else utils::packageDescription(package, fields = "Config/pjrt/cuda-kernels")
  if (is.null(template) || is.na(template) || !nzchar(template)) {
    return(NULL)
  }
  gsub("{version}", as.character(utils::packageVersion(package)), template, fixed = TRUE)
}

cuda_prebuilt_dir <- function(package) {
  file.path(
    tools::R_user_dir("pjrt", "cache"),
    "cuda-prebuilt",
    package,
    as.character(utils::packageVersion(package))
  )
}

# Returns the directory holding `package`'s verified prebuilt kernels,
# downloading them if needed and allowed, or NULL. A failed download is
# remembered, so it is retried only once the URL changes.
cuda_prebuilt_fetch <- function(package, confirmed = FALSE) {
  url <- cuda_kernels_url(package)
  if (is.null(url)) {
    return(NULL)
  }
  dir <- cuda_prebuilt_dir(package)
  url_file <- file.path(dir, "url")
  have <- if (file.exists(url_file)) readLines(url_file, warn = FALSE) else ""
  if (identical(have, url) && file.exists(file.path(dir, "manifest.dcf"))) {
    return(dir)
  }
  if (identical(have, paste("unavailable", url))) {
    return(NULL)
  }
  if (!confirmed && !confirm_cuda_kernels_download(package, url)) {
    return(NULL)
  }

  unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  ok <- tryCatch(
    {
      tarball <- tempfile(fileext = ".tar.gz")
      withr::defer(unlink(tarball))
      pjrt_debug("Downloading prebuilt CUDA kernels of {.pkg {package}} from {.url {url}}")
      # a failure is reported through pjrt_debug(), and the error below
      suppressWarnings(utils::download.file(url, tarball, mode = "wb", quiet = TRUE))
      exdir <- tempfile()
      withr::defer(unlink(exdir, recursive = TRUE))
      utils::untar(tarball, exdir = exdir)
      root <- list.dirs(exdir, recursive = FALSE)
      if (length(root) != 1L) {
        cli_abort("Expected one directory in the tarball.")
      }
      manifest <- as.data.frame(read.dcf(file.path(root, "manifest.dcf")), stringsAsFactors = FALSE)
      md5 <- unname(tools::md5sum(file.path(root, manifest$File)))
      if (!identical(md5, manifest$MD5)) {
        cli_abort("Checksum mismatch.")
      }
      file.copy(list.files(root, full.names = TRUE), dir, recursive = TRUE)
      TRUE
    },
    error = function(e) {
      pjrt_debug("Could not fetch prebuilt CUDA kernels of {.pkg {package}}: {conditionMessage(e)}")
      FALSE
    }
  )
  writeLines(if (ok) url else paste("unavailable", url), url_file)
  if (ok) dir else NULL
}

# Downloading prebuilt kernels is an optimization, so unlike the plugin's,
# a declined or impossible download is not an error: the kernels are then
# compiled locally. Otherwise the same `PJRT_INSTALL` rules apply.
confirm_cuda_kernels_download <- function(package, url) {
  install <- Sys.getenv("PJRT_INSTALL", unset = "")
  if (install == "1") {
    return(TRUE)
  }
  if (install == "0" || !interactive()) {
    return(FALSE)
  }
  cli::cli_inform(c(
    "{.pkg {package}} ships prebuilt CUDA kernels at {.url {url}}.",
    i = "Without them, its kernels are compiled on this machine when first used.",
    i = "Set {.envvar PJRT_INSTALL} to {.val 1} to skip this prompt in the future."
  ))
  isTRUE(utils::askYesNo("Do you want to download them now?"))
}

#' @title Optimize a StableHLO Program
#' @description
#' Run the [`stablehlo-opt`](https://openxla.org/stablehlo) tool on a StableHLO
#' (MLIR) program and return the transformed program.
#'
#' The binary is not shipped with `pjrt`; it is downloaded and cached on first
#' use (see [stablehlo_opt_bin()]).
#'
#' @param program (`PJRTProgram` | `character(1)`)\cr
#'   The program to optimize, either a `PJRTProgram` in `"mlir"` format or
#'   MLIR source code.
#' @param passes (`character()`)\cr
#'   The passes to run, e.g. `"stablehlo-aggressive-folder"`. A leading `--` is
#'   optional. See [stablehlo_opt_passes()] for the available passes.
#' @return `PJRTProgram`
#' @examplesIf stablehlo_opt_available()
#' src <- "
#' func.func @main() -> tensor<2xf32> {
#'   %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
#'   %1 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
#'   %2 = stablehlo.add %0, %1 : tensor<2xf32>
#'   return %2 : tensor<2xf32>
#' }
#' "
#' pjrt_optimize(src)
#' @export
pjrt_optimize <- function(
  program,
  passes = "stablehlo-target-independent-optimization"
) {
  checkmate::assert_character(passes, any.missing = FALSE, min.len = 1L)

  src <- if (inherits(program, "PJRTProgram")) {
    if (program_format(program) != "mlir") {
      cli_abort(c(
        "Can only optimize programs in {.val mlir} format.",
        i = "Got a program in {.val {program_format(program)}} format."
      ))
    }
    program_code(program)
  } else {
    checkmate::assert_string(program)
    program
  }

  # `stablehlo-opt` expects pass names as command line flags.
  flags <- ifelse(startsWith(passes, "--"), passes, paste0("--", passes))

  input <- tempfile(fileext = ".mlir")
  on.exit(unlink(input), add = TRUE)
  writeLines(src, input)

  out <- stablehlo_opt_run(c(flags, input))

  pjrt_program(src = paste(out, collapse = "\n"), format = "mlir")
}

#' @title Concretize the Shapes of a StableHLO Program
#' @description
#' Replace the argument types of a program's `main` function with concrete ones
#' and propagate the refinement through the rest of the program.
#'
#' Programs exported with dynamic (polymorphic) shapes -- e.g. by
#' `jax.export` with a symbolic batch dimension -- cannot be compiled by PJRT,
#' because XLA needs static shapes. Refining them turns such a program into an
#' ordinary, compilable one.
#'
#' @param program (`PJRTProgram` | `character(1)`)\cr
#'   The program to refine, either a `PJRTProgram` in `"mlir"` format or MLIR
#'   source code.
#' @param types (`character()`)\cr
#'   The concrete MLIR types for the arguments of `main`, one per argument,
#'   e.g. `c("tensor<5x3xf32>", "tensor<3xf32>")`.
#' @return `PJRTProgram`
#' @examplesIf stablehlo_opt_available() && plugins_downloaded("cpu")
#' path <- system.file("programs/jax-mlp-dynamic.mlir", package = "pjrt")
#' program <- pjrt_program(path = path)
#' pjrt_refine_shapes(
#'   program,
#'   c(
#'     "tensor<3x4xf32>",
#'     "tensor<4xf32>",
#'     "tensor<4x2xf32>",
#'     "tensor<2xf32>",
#'     "tensor<5x3xf32>"
#'   )
#' )
#' @export
pjrt_refine_shapes <- function(program, types) {
  checkmate::assert_character(types, any.missing = FALSE, min.len = 1L)

  pjrt_optimize(
    program,
    passes = c(
      sprintf(
        "stablehlo-refine-arguments=types='%s'",
        paste(types, collapse = ",")
      ),
      "stablehlo-refine-shapes",
      "stablehlo-canonicalize-dynamism",
      # Frameworks emit `stablehlo.custom_call @shape_assertion` to guard the
      # polymorphic dimensions. Once the shapes are concrete, these can be
      # checked at compile time and erased -- PJRT has no such custom call.
      "stablehlo-check-shape-assertions"
    )
  )
}

#' @title Available `stablehlo-opt` Passes
#' @description
#' The StableHLO passes supported by the `stablehlo-opt` binary, extracted
#' from its `--help` output.
#' @return (`character()`)\cr
#'   Pass names without the leading `--`, e.g.
#'   `"stablehlo-aggressive-folder"`.
#' @examplesIf stablehlo_opt_available()
#' head(stablehlo_opt_passes())
#' @export
stablehlo_opt_passes <- function() {
  help <- stablehlo_opt_run("--help")
  # Pass flags are listed one per line as e.g. "  --stablehlo-refine-shapes"
  matches <- regmatches(help, regexpr("--stablehlo-[a-z0-9-]+", help))
  sort(unique(substring(unlist(matches), 3L)))
}

# Invoke the stablehlo-opt binary, turning a non-zero exit status into an R
# error. `stablehlo-opt` reports diagnostics (e.g. MLIR parse errors) on stderr.
# `system2()` passes the arguments through a shell without quoting them, so we
# quote them here -- MLIR types such as `tensor<2xf32>` would otherwise be read
# as shell redirections.
stablehlo_opt_run <- function(args) {
  err <- tempfile()
  on.exit(unlink(err), add = TRUE)

  out <- suppressWarnings(system2(
    stablehlo_opt_bin(),
    args = shQuote(args),
    stdout = TRUE,
    stderr = err
  ))

  status <- attr(out, "status") %||% 0L
  if (status != 0L) {
    # Interpolated rather than pasted into the template: MLIR diagnostics
    # contain braces, which cli would otherwise try to evaluate.
    diagnostics <- paste(readLines(err, warn = FALSE), collapse = "\n")
    cli_abort(c(
      "{.code stablehlo-opt} failed with exit status {status}.",
      i = "{diagnostics}"
    ))
  }

  out
}

#' @title Path to the `stablehlo-opt` Binary
#' @description
#' Return the path to the `stablehlo-opt` binary, downloading and caching it
#' first if it is not available yet.
#'
#' The download requires confirmation, see the `PJRT_INSTALL` environment
#' variable in [pjrt-package].
#'
#' @param install (`logical(1)`)\cr
#'   Whether to download the binary when it is missing. If `FALSE` and the
#'   binary is missing, an error is raised.
#' @return (`character(1)`)\cr
#'   Path to the binary.
#' @examplesIf stablehlo_opt_available()
#' stablehlo_opt_bin()
#' @export
stablehlo_opt_bin <- function(install = TRUE) {
  checkmate::assert_flag(install)

  path <- Sys.getenv("PJRT_STABLEHLO_OPT_PATH", "")
  if (path != "") {
    if (!file.exists(path)) {
      cli_abort(c(
        "{.envvar PJRT_STABLEHLO_OPT_PATH} points to a non-existing file.",
        x = "No file at {.path {path}}."
      ))
    }
    return(path)
  }

  bin <- file.path(stablehlo_opt_cache_dir(), stablehlo_opt_bin_name())
  if (file.exists(bin)) {
    return(bin)
  }

  if (!install) {
    cli_abort(c(
      "The {.code stablehlo-opt} binary is not downloaded yet.",
      i = "Call {.run pjrt::stablehlo_opt_bin()} to download it."
    ))
  }

  stablehlo_opt_download()
  bin
}

#' @title Check if `stablehlo-opt` is Downloaded
#' @description
#' Whether the `stablehlo-opt` binary is available locally, i.e. whether
#' [pjrt_optimize()] can be used without a download.
#' @return `logical(1)`
#' @examples
#' stablehlo_opt_available()
#' @export
stablehlo_opt_available <- function() {
  !inherits(
    try(stablehlo_opt_bin(install = FALSE), silent = TRUE),
    "try-error"
  )
}

stablehlo_opt_cache_dir <- function() {
  file.path(tools::R_user_dir("pjrt", which = "cache"), "stablehlo-opt")
}

stablehlo_opt_bin_name <- function() {
  if (plugin_os() == "windows") "stablehlo-opt.exe" else "stablehlo-opt"
}

stablehlo_opt_download <- function() {
  url <- stablehlo_opt_url()
  confirm_install(
    what = cli::format_inline("the {.code stablehlo-opt} binary"),
    url = url,
    dest = stablehlo_opt_cache_dir(),
    override = cli::format_inline(
      "{.envvar PJRT_STABLEHLO_OPT_PATH} to a local binary"
    )
  )

  archive <- tempfile(fileext = if (endsWith(url, ".zip")) ".zip" else ".tar.gz")
  on.exit(unlink(archive), add = TRUE)
  cli::cli_inform("Downloading {.code stablehlo-opt} from {.url {url}}")
  withr::local_options(timeout = max(getOption("timeout"), 3600L))
  utils::download.file(url, archive, mode = "wb", quiet = FALSE)

  tmp <- withr::local_tempdir()
  if (endsWith(url, ".zip")) {
    utils::unzip(archive, exdir = tmp)
  } else {
    utils::untar(archive, exdir = tmp)
  }

  bin_name <- stablehlo_opt_bin_name()
  src <- list.files(tmp, pattern = bin_name, recursive = TRUE, full.names = TRUE)
  if (!length(src)) {
    cli_abort("The downloaded archive does not contain a {.file {bin_name}} binary.")
  }

  cache_dir <- stablehlo_opt_cache_dir()
  fs::dir_create(cache_dir, recurse = TRUE)
  bin <- file.path(cache_dir, bin_name)
  fs::file_copy(src[[1L]], bin, overwrite = TRUE)
  Sys.chmod(bin, "0755")

  invisible(bin)
}

stablehlo_opt_url <- function() {
  url <- Sys.getenv("PJRT_STABLEHLO_OPT_URL", "")
  if (url != "") {
    return(url)
  }

  os <- plugin_os()
  if (os == "darwin") {
    os <- "mac"
  }
  arch <- switch(plugin_arch(), amd64 = "x86_64", arm64 = "arm64", "unsupported")

  # Only these combinations are built by r-xla/pjrt-builds.
  target <- paste0(os, "-", arch)
  if (!target %in% c("linux-x86_64", "mac-arm64", "windows-x86_64")) {
    cli_abort(c(
      "No {.code stablehlo-opt} binary is available for {.val {target}}.",
      i = "Available builds: {.val linux-x86_64}, {.val mac-arm64}, {.val windows-x86_64}.",
      i = "To override, set {.envvar PJRT_STABLEHLO_OPT_URL} to an archive URL, or {.envvar PJRT_STABLEHLO_OPT_PATH} to a local binary."
    ))
  }

  version <- Sys.getenv("PJRT_STABLEHLO_OPT_VERSION", "")
  if (version == "") {
    version <- "main"
  }
  ext <- if (os == "windows") ".zip" else ".tar.gz"

  sprintf(
    "https://github.com/r-xla/pjrt-builds/releases/download/stablehlo/stablehlo-opt-%s-%s%s",
    version,
    target,
    ext
  )
}

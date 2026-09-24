# A tarball in the layout pjrt_cuda_build_kernels() writes, with made-up images.
fake_kernels_tarball <- function(module_id, dir, corrupt = FALSE) {
  root <- file.path(dir, "fake-cuda-kernels-1.0")
  dir.create(file.path(root, module_id), recursive = TRUE)
  writeBin(as.raw(1:8), file.path(root, module_id, "sm_80.cubin"))
  writeLines("expression\tlowered", file.path(root, module_id, "names.tsv"))
  md5 <- unname(tools::md5sum(file.path(root, module_id, "sm_80.cubin")))
  write.dcf(
    data.frame(
      Module = module_id,
      Target = "sm_80",
      File = file.path(module_id, "sm_80.cubin"),
      MD5 = if (corrupt) "0" else md5
    ),
    file.path(root, "manifest.dcf")
  )
  tarball <- file.path(dir, "fake.tar.gz")
  withr::with_dir(dir, utils::tar(tarball, basename(root), compression = "gzip", tar = "internal"))
  paste0("file://", tarball)
}

describe("cuda_kernels_url", {
  it("reads the package's DESCRIPTION, with the version filled in", {
    url <- cuda_kernels_url("pjrt")
    expect_match(url, as.character(utils::packageVersion("pjrt")), fixed = TRUE)
    expect_false(grepl("{version}", url, fixed = TRUE))
  })

  it("is overridden by an environment variable, and NULL without either", {
    withr::local_envvar(PJRT_CUDA_KERNELS_URL_PJRT = "file:///k-{version}.tar.gz")
    expect_identical(cuda_kernels_url("pjrt"), sprintf("file:///k-%s.tar.gz", utils::packageVersion("pjrt")))
    expect_null(cuda_kernels_url("stats"))
  })
})

describe("cuda_prebuilt_fetch", {
  # every test gets its own cache, and a stand-in package entry pointing at it
  local_prebuilt <- function(envir = parent.frame()) {
    cache <- withr::local_tempdir(.local_envir = envir)
    withr::local_envvar(R_USER_CACHE_DIR = cache, .local_envir = envir)
  }

  it("downloads, verifies and keeps a package's kernels", {
    local_prebuilt()
    url <- fake_kernels_tarball("abc", withr::local_tempdir())
    withr::local_envvar(PJRT_CUDA_KERNELS_URL_PJRT = url, PJRT_INSTALL = "1")
    dir <- cuda_prebuilt_fetch("pjrt")
    expect_true(file.exists(file.path(dir, "abc", "sm_80.cubin")))
    # a second call finds them without downloading
    withr::local_envvar(PJRT_INSTALL = "0")
    expect_identical(cuda_prebuilt_fetch("pjrt"), dir)
  })

  it("rejects a tarball whose checksums do not match, and remembers that", {
    local_prebuilt()
    url <- fake_kernels_tarball("abc", withr::local_tempdir(), corrupt = TRUE)
    withr::local_envvar(PJRT_CUDA_KERNELS_URL_PJRT = url, PJRT_INSTALL = "1")
    expect_null(cuda_prebuilt_fetch("pjrt"))
    expect_match(readLines(file.path(cuda_prebuilt_dir("pjrt"), "url")), "^unavailable ")
    expect_null(cuda_prebuilt_fetch("pjrt"))
  })

  it("falls back quietly when the download is not allowed or fails", {
    local_prebuilt()
    withr::local_envvar(PJRT_CUDA_KERNELS_URL_PJRT = "file:///does/not/exist.tar.gz", PJRT_INSTALL = "0")
    expect_null(cuda_prebuilt_fetch("pjrt"))
    withr::local_envvar(PJRT_INSTALL = "1")
    expect_null(cuda_prebuilt_fetch("pjrt"))
  })
})

describe("pjrt_cuda_build_kernels", {
  it("builds pjrt's kernels, which a fresh session then loads instead of compiling", {
    skip_if(!is_cuda())
    dest <- withr::local_tempdir()
    tarball <- pjrt_cuda_build_kernels("pjrt", dest = dest, targets = c("sm_80", "compute_80"))
    expect_true(file.exists(tarball))
    files <- utils::untar(tarball, list = TRUE)
    expect_true(any(grepl("sm_80\\.cubin$", files)))
    expect_true(any(grepl("compute_80\\.ptx$", files)))

    origins <- callr::r(
      function(url, plugin) {
        Sys.setenv(
          PJRT_CUDA_KERNELS_URL_PJRT = url,
          PJRT_INSTALL = "1",
          PJRT_PLUGIN_PATH_CUDA = plugin,
          PJRT_CUDA_CACHE = ""
        )
        library(pjrt)
        src <- r"(func.func @main(%p: tensor<2xi32>) -> tensor<4xi32> {
  %0 = stablehlo.custom_call @lu_pivots_to_permutation(%p) {
    call_target_name = "lu_pivots_to_permutation",
    api_version = 4 : i32,
    operand_layouts = [dense<0> : tensor<1xindex>],
    result_layouts = [dense<0> : tensor<1xindex>]
  } : (tensor<2xi32>) -> tensor<4xi32>
  "func.return"(%0) : (tensor<4xi32>) -> ()
})"
        exec <- pjrt_compile(pjrt_program(src), device = "cuda")
        out <- as_array(pjrt_execute(exec, pjrt_buffer(c(2L, 2L), dtype = "i32", device = "cuda")))
        ns <- asNamespace("pjrt")
        id <- ns$the[["cuda_modules"]][["pjrt"]][[1L]]$id
        list(out = as.vector(out), origins = ns$impl_cuda_module_origins(id))
      },
      list(paste0("file://", tarball), plugin_path("cuda")),
      # an empty cache, but the CUDA libraries where they are
      env = c(
        callr::rcmd_safe_env(),
        R_USER_CACHE_DIR = withr::local_tempdir(),
        PJRT_CUDA_HOME = dirname(getExportedValue(cuda_r_package(), "lib_path")())
      )
    )
    expect_identical(origins$out, c(2L, 1L, 3L, 4L))
    expect_identical(origins$origins, "prebuilt sm_80")
  })
})

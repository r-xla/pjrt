# TEMPORARY (remove before merging): checks that the Windows build links the
# static LAPACK/BLAS from Rtools instead of R's Rlapack/Rblas DLLs, so that the
# f32 custom calls use the single-precision routines. The custom calls
# themselves, f32 included, are covered by test-linalg.R on every platform.
describe("Windows LAPACK linking", {
  it("links LAPACK and BLAS statically", {
    skip_if_not(.Platform$OS.type == "windows")
    objdump <- Sys.which("objdump")
    skip_if(!nzchar(objdump), "objdump is not on the PATH")
    dll <- getLoadedDLLs()[["pjrt"]][["path"]]
    imports <- system2(objdump, c("-p", shQuote(dll)), stdout = TRUE)
    dlls <- trimws(sub(".*DLL Name:", "", grep("DLL Name:", imports, value = TRUE)))
    # Visible in the CI log, whatever the outcome.
    message("pjrt.dll imports: ", paste(dlls, collapse = ", "))
    expect_false(
      any(grepl("lapack|blas", dlls, ignore.case = TRUE)),
      info = paste(dlls, collapse = ", ")
    )
  })

  it("runs an f32 Cholesky factorisation", {
    a <- matrix(c(4, 2, 2, 3), 2L, 2L)
    res <- run_linalg(
      "potrf",
      inputs = list(a),
      in_specs = list(list(dims = c(2L, 2L), dtype = "f32")),
      out_specs = list(list(dims = c(2L, 2L), dtype = "f32"), list(dims = integer(), dtype = "i32")),
      attrs = list(backend_config = "{lower = true}"),
      in_layouts = batched_matrix_layout(2L),
      out_layouts = c(batched_matrix_layout(2L), row_major_layout(0L))
    )
    expect_equal(res[[2L]], 0L)
    l <- res[[1L]]
    l[upper.tri(l)] <- 0
    expect_equal(l, t(chol(a)), tolerance = 1e-6)
  })
})

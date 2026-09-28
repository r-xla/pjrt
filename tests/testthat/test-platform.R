describe("platform_support", {
  it("lists every backend for each OS and architecture", {
    support <- platform_support()
    expect_data_frame(support, nrows = 6L)
    expect_named(support, c("os", "arch", "cpu", "cuda"))
    expect_false(anyDuplicated(support[c("os", "arch")]) > 0L)
  })

  it("uses only the documented support levels", {
    support <- platform_support()
    levels <- unlist(support[c("cpu", "cuda")], use.names = FALSE)
    expect_subset(levels, c("yes", "wsl2", "no"))
  })
})

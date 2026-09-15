describe("remove_pkg_hooks", {
  local_hook_event <- function(event) {
    old <- getHook(event)
    withr::defer(setHook(event, old, action = "replace"), envir = parent.frame())
    event
  }

  it("removes the hooks the package registered", {
    event <- local_hook_event(packageEvent("pjrtTestPkg", "onUnload"))
    mine <- function(...) NULL
    environment(mine) <- asNamespace("pjrt")
    setHook(event, mine, action = "append")

    remove_pkg_hooks(event, "pjrt")

    expect_length(getHook(event), 0L)
  })

  it("keeps hooks it cannot attribute to the package", {
    event <- local_hook_event(packageEvent("pjrtTestPkg", "onUnload"))
    # A hook registered by someone else: its environment is neither a namespace
    # nor carries a `pkgname`, so its owner is unknown.
    foreign <- function(...) NULL
    setHook(event, foreign, action = "append")

    remove_pkg_hooks(event, "pjrt")

    hooks <- getHook(event)
    expect_length(hooks, 1L)
    expect_false(any(vapply(hooks, is.null, logical(1))))
  })

  it("leaves a foreign hook runnable after removing the package's own", {
    # A NULL in the hook list only errors once the hooks are run, which is what
    # broke a repeated `devtools::load_all()`.
    event <- local_hook_event(packageEvent("pjrtTestPkg", "onUnload"))
    mine <- function(...) NULL
    environment(mine) <- asNamespace("pjrt")
    state <- new.env(parent = emptyenv())
    state$ran <- FALSE
    foreign <- function(...) {
      state$ran <- TRUE
    }
    setHook(event, mine, action = "append")
    setHook(event, foreign, action = "append")

    remove_pkg_hooks(event, "pjrt")
    for (hook in getHook(event)) {
      hook()
    }

    expect_true(state$ran)
  })
})

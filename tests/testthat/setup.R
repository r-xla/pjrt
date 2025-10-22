old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE
)

# https://github.com/HenrikBengtsson/Wishlist-for-R/issues/88
old_opts <- lapply(old_opts, function(x) if (is.null(x)) FALSE else x)
the[["clients"]] <- hashtab()
the[["plugins"]] <- hashtab()
old_pjrt_config <- the[["config"]]
# so we can test multiple devices.
pjrt_config(cpu_device_count = 2L)

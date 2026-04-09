options(old_opts)
added <- setdiff(names(the[["custom_calls"]]), custom_calls_before)
the[["custom_calls"]][added] <- NULL

is_lazy_buffer <- function(x) {
  inherits(x, "PJRTLazyBuffer")
}

check_lazy_buffer <- function(x) {
  if (!is_lazy_buffer(x)) {
    cli_abort("`x` must be a `PJRTLazyBuffer`.")
  }
}

#' Execute a PJRT program asynchronously
#'
#' Returns lazy buffers immediately; execution continues in the background.
#'
#' @inheritParams pjrt_execute
#' @return `PJRTLazyBuffer` or list of them.
#' @export
pjrt_execute_lazy <- function(executable, ..., execution_options = NULL, simplify = TRUE) {
  if (!is.null(...names())) {
    cli_abort("Expected unnamed arguments")
  }
  check_loaded_executable(executable)
  input <- list(...)
  lapply(input, check_buffer)

  if (is.null(execution_options)) {
    execution_options <- pjrt_execution_options()
  } else {
    check_execution_options(execution_options)
  }

  assert_flag(simplify)

  result <- impl_loaded_executable_execute_lazy(executable, input, execution_options)

  if (simplify && length(result) == 1L) {
    return(result[[1]])
  }

  result
}

#' @export
print.PJRTLazyBuffer <- function(x, ...) {
  ready <- impl_lazy_buffer_is_ready(x)
  state <- if (ready) "ready" else "pending"
  cat("<PJRTLazyBuffer ", state, ">\n", sep = "")
  invisible(x)
}

#' Check if a lazy buffer is ready
#' @export
lazy_buffer_ready <- function(x) {
  check_lazy_buffer(x)
  impl_lazy_buffer_is_ready(x)
}

#' Materialize a lazy buffer into a regular PJRTBuffer
#' @export
pjrt_lazy_buffer_materialize <- function(x) {
  check_lazy_buffer(x)
  impl_lazy_buffer_materialize(x)
}

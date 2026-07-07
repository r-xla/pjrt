#' @title Create Execution Options
#' @description
#' Create execution options for configuring how a PJRT program is executed,
#' including buffer donation settings.
#' **Important**:
#' It is not enough to only mark a buffer as donatable (not not donatable)
#' during runtime. The program also needs to specify this during compile-time
#' via input-output aliasing (stableHLO attribute `tf.aliasing_output`).
#'
#' @param non_donatable_input_indices (`integer()`)\cr
#'   A vector of input buffer indices that should not be donated during execution (0-based).
#'   Buffer donation allows the runtime to reuse input buffers for outputs when
#'   possible, which can improve performance. However, if an input buffer is
#'   referenced multiple times or needs to be preserved, it should be marked as
#'   non-donatable.
#' @param launch_id (`integer(1)`)\cr
#'   An optional launch identifier for multi-device executions. This can be used
#'   to detect scheduling errors in multi-host programs.
#'
#' @return `PJRTExecuteOptions`
#' @examplesIf plugins_downloaded()
#' # Create default execution options
#' opts <- pjrt_execution_options()
#'
#' # Mark buffer 0 as non-donatable
#' opts <- pjrt_execution_options(non_donatable_input_indices = 0L)
#' @export
pjrt_execution_options <- function(
  non_donatable_input_indices = integer(),
  launch_id = 0L
) {
  checkmate::assert_integer(non_donatable_input_indices, any.missing = FALSE)
  checkmate::assert_integer(launch_id, len = 1, any.missing = FALSE)

  impl_execution_options_create(non_donatable_input_indices, launch_id)
}

check_execution_options <- function(options) {
  stopifnot(inherits(options, "PJRTExecuteOptions"))
  invisible(NULL)
}

# The default execution options (no non-donatable indices, launch_id 0) are an
# immutable value reused on every donation-enabled execute. Building them per
# call (checkmate asserts + a C++ constructor) is pure overhead on the hot
# dispatch path, so we memoize a single shared instance.
default_execution_options <- function() {
  opts <- the[["default_execution_options"]]
  if (is.null(opts)) {
    opts <- pjrt_execution_options()
    the[["default_execution_options"]] <- opts
  }
  opts
}

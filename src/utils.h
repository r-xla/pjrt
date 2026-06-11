#pragma once
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "gc.h"
#include "xla/pjrt/c/pjrt_c_api.h"

void check_err(const PJRT_Api *api, PJRT_Error *err);

// Return the error code of a PJRT_Error. If querying the code itself fails, the
// inner error is destroyed and UNKNOWN is returned so the caller still surfaces
// the original error via check_err.
PJRT_Error_Code get_error_code(const PJRT_Api *api, PJRT_Error *err);

// Destroy a PJRT_Error. The error object is owned by the caller of the PJRT C
// API and must be freed once its message/code has been read; a null error is a
// no-op.
void destroy_error(const PJRT_Api *api, PJRT_Error *err);

namespace rpjrt {

// Temporarily redirects stderr (fd 2) into a reusable sink so that XLA's native
// logging emitted during a single allocation/execution attempt can be dropped
// when we recover from it. Process-global and not thread-safe; intended only
// for the main-thread allocation calls wrapped by try_alloc(). The sink itself
// is a process-wide singleton (see utils.cpp); this handle only carries the
// per-call dup of the original stderr.
struct StderrCapture {
  int saved_fd = -1;  // dup of the original stderr; -1 means capture disabled
};

// Begin capturing stderr. On any failure capture is silently disabled
// (saved_fd == -1) and the real stderr is left untouched.
StderrCapture begin_stderr_capture();

// Restore stderr. If `replay` is true the captured bytes are written back to
// the real stderr first (so non-OOM diagnostics are never hidden); otherwise
// they are discarded.
void end_stderr_capture(StderrCapture &cap, bool replay);

// Write `msg` (followed by a newline) to R's stderr, but only when the
// PJRT_DEBUG environment variable is set to a non-empty value. Used to make
// the otherwise-silent gc-on-OOM retry observable.
void debug_inform(const char *msg);

}  // namespace rpjrt

// Run a PJRT allocation call. If it returns RESOURCE_EXHAUSTED, call R's gc()
// (via rpjrt::call_r_gc) and retry the call once. Any other error, or a
// still-failing retry, is surfaced via check_err. `alloc_fn` must be callable
// twice and return a PJRT_Error*; the underlying *_Args struct is filled in
// each call so it is safe to reuse.
//
// `suppress_logs` gates the stderr capture, which must wrap *every* call (XLA
// writes the OOM diagnostic during alloc_fn, before returning, so it can't be
// suppressed after the fact). Callers pass false on CPU, where allocation
// failure is rare and the gc-and-retry is largely moot, so the hot path (buffer
// upload, execute) pays nothing. It is left on for the backends where
// OOM-and-recover actually happens (CUDA), so the recovered first attempt stays
// quiet there.
template <typename F>
void try_alloc(const PJRT_Api *api, F &&alloc_fn, bool suppress_logs = true) {
  rpjrt::StderrCapture cap;
  if (suppress_logs) cap = rpjrt::begin_stderr_capture();
  PJRT_Error *err = alloc_fn();
  bool is_oom = err != nullptr &&
                get_error_code(api, err) == PJRT_Error_Code_RESOURCE_EXHAUSTED;
  if (suppress_logs) rpjrt::end_stderr_capture(cap, /*replay=*/!is_oom);

  if (is_oom) {
    destroy_error(api, err);
    rpjrt::call_r_gc();
    rpjrt::debug_inform("pjrt: RESOURCE_EXHAUSTED — ran R gc, retrying");
    err = std::forward<F>(alloc_fn)();
  }
  check_err(api, err);
}

std::vector<int64_t> dims2strides(std::vector<int64_t> dims, bool row_major);

template <typename src_type, typename dst_type>
void row_to_col_order(const std::vector<src_type> &src, dst_type *dst,
                      const std::vector<int64_t> &dims) {
  if (dims.empty() || dims.size() == 1) {
    std::copy(src.begin(), src.end(), dst);
    return;
  }

  int64_t total = 1, n = dims.size();
  for (int64_t d : dims) total *= d;

  for (int64_t d : dims) {
    if (total == d) {
      std::copy(src.begin(), src.end(), dst);
      return;
    }
  }

  std::vector<int64_t> row_strides(n, 1), col_strides(n, 1);
  row_strides = dims2strides(dims, true);
  col_strides = dims2strides(dims, false);

  for (auto i = 0; i < total; ++i) {
    int64_t tmp = i, idx_row = i, idx_col = 0;
    for (auto j = 0; j < n; ++j) {
      int64_t coord = tmp / row_strides[j];
      tmp %= row_strides[j];
      idx_col += coord * col_strides[j];
    }
    dst[idx_col] = static_cast<dst_type>(src[idx_row]);
  }
}

// True if `minor_to_major` is the dense row-major order [n-1, ..., 1, 0]
// (last logical dimension fastest-varying), the layout our readback assumes.
inline bool is_row_major(const std::vector<int64_t> &minor_to_major) {
  const int64_t n = minor_to_major.size();
  for (int64_t k = 0; k < n; ++k) {
    if (minor_to_major[k] != n - 1 - k) return false;
  }
  return true;
}

// Reorder `total` elements from the device physical order described by
// `minor_to_major` (minor_to_major[0] = fastest-varying logical dim) into
// logical row-major order. A no-op copy when the layout is already row-major,
// so the common path costs nothing extra. `src` and `dst` must not alias.
template <typename T>
void device_to_row_major(const T *src, T *dst, const std::vector<int64_t> &dims,
                         const std::vector<int64_t> &minor_to_major) {
  const int64_t n = dims.size();
  int64_t total = 1;
  for (int64_t d : dims) total *= d;
  if (n <= 1 || is_row_major(minor_to_major)) {
    std::copy(src, src + total, dst);
    return;
  }
  // Physical stride of each logical dimension, derived from minor_to_major.
  std::vector<int64_t> phys(n, 1);
  int64_t acc = 1;
  for (int64_t k = 0; k < n; ++k) {
    phys[minor_to_major[k]] = acc;
    acc *= dims[minor_to_major[k]];
  }
  const std::vector<int64_t> row = dims2strides(dims, /*row_major=*/true);
  for (int64_t r = 0; r < total; ++r) {
    int64_t tmp = r, p = 0;
    for (int64_t d = 0; d < n; ++d) {
      const int64_t coord = tmp / row[d];
      tmp %= row[d];
      p += coord * phys[d];
    }
    dst[r] = src[p];
  }
}

size_t sizeof_pjrt_buffer_type(PJRT_Buffer_Type type);

bool format_is_irrelevant(const std::vector<int64_t> &dims);

std::optional<std::vector<int64_t>> get_byte_strides(
    const std::vector<int64_t> &dims, bool row_major, size_t sizeof_type);

inline int64_t number_of_elements(const std::vector<int64_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

std::vector<int64_t> id2indices(int lid, const std::vector<int64_t> strides);

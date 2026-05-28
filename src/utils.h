#pragma once
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "gc.h"
#include "xla/pjrt/c/pjrt_c_api.h"

void check_err(const PJRT_Api *api, PJRT_Error *err);

PJRT_Error_Code get_error_code(const PJRT_Api *api, PJRT_Error *err);

void destroy_error(const PJRT_Api *api, PJRT_Error *err);

// Run a PJRT allocation call. If it returns RESOURCE_EXHAUSTED, call R's gc()
// (via rpjrt::call_r_gc) and retry the call once. Any other error, or a
// still-failing retry, is surfaced via check_err. `alloc_fn` must be callable
// twice and return a PJRT_Error*; the underlying *_Args struct is filled in
// each call so it is safe to reuse.
template <typename F>
void try_alloc(const PJRT_Api *api, F &&alloc_fn) {
  PJRT_Error *err = alloc_fn();
  if (err != nullptr &&
      get_error_code(api, err) == PJRT_Error_Code_RESOURCE_EXHAUSTED) {
    destroy_error(api, err);
    rpjrt::call_r_gc();
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

size_t sizeof_pjrt_buffer_type(PJRT_Buffer_Type type);

bool format_is_irrelevant(const std::vector<int64_t> &dims);

std::optional<std::vector<int64_t>> get_byte_strides(
    const std::vector<int64_t> &dims, bool row_major, size_t sizeof_type);

inline int64_t number_of_elements(const std::vector<int64_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

std::vector<int64_t> id2indices(int lid, const std::vector<int64_t> strides);

#pragma once
#include <optional>
#include <vector>

#include "pjrt.h"

void check_err(const PJRT_Api *api, PJRT_Error *err);

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
  for (int64_t i = n - 2; i >= 0; --i)
    row_strides[i] = row_strides[i + 1] * dims[i + 1];
  for (int64_t i = 1; i < n; ++i)
    col_strides[i] = col_strides[i - 1] * dims[i - 1];

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

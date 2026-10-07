// Shared utilities for XLA FFI handlers backed by LAPACK / cuSOLVER.
#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "xla/ffi/api/ffi.h"

#define PJRT_RETURN_IF_ERROR(expr) \
  do {                             \
    auto _e = (expr);              \
    if (!_e.success()) return _e;  \
  } while (0)

namespace rpjrt {

// Validate an int64_t dimension fits in int (LAPACK / cuSOLVER both use int).
// Mirrors jaxlib's MaybeCastNoOverflow<int>().
inline xla::ffi::Error dim_to_int(std::int64_t v, const char *name, int &out) {
  if (v < 0 || v > std::numeric_limits<int>::max()) {
    return xla::ffi::Error::InvalidArgument(std::string(name) +
                                            " dimension out of int range");
  }
  out = static_cast<int>(v);
  return xla::ffi::Error::Success();
}

// Split the dimensions of a batch of square matrices, [..., n, n], into the
// matrix size `n` and the number of matrices `batch` (the product of the
// leading dimensions; 1 when there are none).
template <typename Dims>
inline xla::ffi::Error batched_square_dims(const Dims &dims, const char *op,
                                           int &n, std::int64_t &batch) {
  std::size_t r = dims.size();
  if (r < 2 || dims[r - 2] != dims[r - 1]) {
    return xla::ffi::Error::InvalidArgument(
        std::string(op) + " requires a batch of square matrices [..., n, n]");
  }
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[r - 1], "matrix", n));
  batch = 1;
  for (std::size_t i = 0; i + 2 < r; i++) batch *= dims[i];
  return xla::ffi::Error::Success();
}

}  // namespace rpjrt

// Dispatch on a buffer's element_type for f32/f64 only (the only float
// precisions our LAPACK / cuSOLVER paths support). Use as:
//   PJRT_DISPATCH_FLOAT(input.element_type(), op_impl, args...)
// where op_impl<T> is a template returning xla::ffi::Error.
#define PJRT_DISPATCH_FLOAT(et, IMPL, ...)          \
  do {                                              \
    switch (et) {                                   \
      case xla::ffi::DataType::F32:                 \
        return IMPL<float>(__VA_ARGS__);            \
      case xla::ffi::DataType::F64:                 \
        return IMPL<double>(__VA_ARGS__);           \
      default:                                      \
        return xla::ffi::Error::InvalidArgument(    \
            "operation only supports f32 and f64"); \
    }                                               \
  } while (0)
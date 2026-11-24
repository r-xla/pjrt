// Shared utilities for XLA FFI handlers backed by LAPACK / cuSOLVER.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

// Validate the workspace size a LAPACK workspace query (lwork = -1) reports,
// as a floating-point number, and convert it to the int lwork LAPACK takes.
// A large matrix can need more than INT_MAX elements, which 32-bit LAPACK
// cannot address.
template <typename S>
inline xla::ffi::Error workspace_to_int(S size, const char *op, int &out) {
  const double v = static_cast<double>(size);
  if (!(v >= 0) || v > std::numeric_limits<int>::max()) {
    return xla::ffi::Error::InvalidArgument(std::string(op) +
                                            " workspace size out of int range");
  }
  out = static_cast<int>(v);
  return xla::ffi::Error::Success();
}

// A batch of m x n matrices, each passed as its column-major data: a row-major
// [..., n, m] buffer, the way pjrt stores an R array of shape (m, n, ...) with
// its shape reversed (`pjrt_buffer(row_major = TRUE)`). The batch dimensions
// lead, and LAPACK and cuSOLVER read the data of every matrix as the matrix
// itself, with leading dimension m.
struct MatrixBatch {
  int m = 0;
  int n = 0;
  // The number of matrices: the product of the batch dimensions, 1 when there
  // are none.
  std::int64_t size = 1;
  xla::ffi::Span<const std::int64_t> dims;

  // Fail unless `buffer`, the `what` of `op`, has the batch dimensions
  // followed by `tail`: the buffers of a call must agree on their batch.
  xla::ffi::Error check(xla::ffi::Span<const std::int64_t> buffer,
                        std::initializer_list<std::int64_t> tail,
                        const char *op, const char *what) const {
    bool ok =
        buffer.size() == dims.size() + tail.size() &&
        std::equal(dims.begin(), dims.end(), buffer.begin()) &&
        std::equal(tail.begin(), tail.end(), buffer.begin() + dims.size());
    if (ok) return xla::ffi::Error::Success();
    return xla::ffi::Error::InvalidArgument(
        std::string(op) + " requires " + what +
        " with the batch dimensions of its operand");
  }
};

inline xla::ffi::Error matrix_batch(xla::ffi::Span<const std::int64_t> dims,
                                    const char *op, MatrixBatch &out) {
  std::size_t r = dims.size();
  if (r < 2) {
    return xla::ffi::Error::InvalidArgument(
        std::string(op) + " requires a batch of matrices [..., n, m]");
  }
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[r - 1], "rows", out.m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[r - 2], "cols", out.n));
  out.dims = dims.first(r - 2);
  out.size = 1;
  for (std::int64_t d : out.dims) out.size *= d;
  return xla::ffi::Error::Success();
}

inline xla::ffi::Error square_matrix_batch(
    xla::ffi::Span<const std::int64_t> dims, const char *op, MatrixBatch &out) {
  PJRT_RETURN_IF_ERROR(matrix_batch(dims, op, out));
  if (out.m != out.n) {
    return xla::ffi::Error::InvalidArgument(
        std::string(op) + " requires a batch of square matrices [..., n, n]");
  }
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

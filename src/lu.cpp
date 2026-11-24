// LU decomposition with partial pivoting, P A = L U, over a batch of matrices
// that XLA's intra-op threads share.
//
// LAPACK getrf overwrites A with L (strictly below the diagonal, unit
// diagonal implicit) and U (on and above the diagonal). The k = min(m, n)
// pivot indices are returned in `ipiv`, 1-based row swaps applied during
// elimination.
//
// Every m x n matrix is passed as its column-major data, a row-major [n, m]
// block behind the batch dimensions (see `MatrixBatch`).
//
// Outputs:
//   LU      : [..., n, m], same dtype as input
//   pivots  : [..., k]     int32
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "ffi_common.h"
#include "ffi_handlers.h"
#include "ffi_lapack.h"
#include "ffi_thread_pool.h"

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error lu_impl(ThreadPool &pool, AnyBuffer input,
                     Result<AnyBuffer> lu_out, Result<AnyBuffer> piv_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(matrix_batch(input.dimensions(), "lu", batch));
  const int m = batch.m;
  const int n = batch.n;
  const int k = std::min(m, n);
  PJRT_RETURN_IF_ERROR(
      batch.check((*lu_out).dimensions(), {n, m}, "lu", "LU [..., n, m]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*piv_out).dimensions(), {k}, "lu", "pivots [..., k]"));

  const T *in = static_cast<const T *>(input.untyped_data());
  T *lu = static_cast<T *>((*lu_out).untyped_data());
  int *ipiv = static_cast<int *>((*piv_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const int lda = std::max(m, 1);

  return parallel_for(
      pool, batch.size, lapack_max_threads(),
      [&](std::int64_t b0, std::int64_t b1) {
        // getrf factors in place: seed the output with the input.
        copy_unless_aliased(lu + b0 * mat, in + b0 * mat, (b1 - b0) * mat);
        for (std::int64_t b = b0; b < b1; b++) {
          int info;
          Lapack<T>::getrf(&m, &n, lu + b * mat, &lda, ipiv + b * k, &info);
          // info > 0 means U has an exact zero pivot: the factorisation is
          // still complete, the matrix is just singular.
          if (info < 0) return lapack_check_info(info, "getrf");
        }
        return Error::Success();
      });
}

static Error do_lu(ThreadPool pool, AnyBuffer input, Result<AnyBuffer> lu_out,
                   Result<AnyBuffer> piv_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), lu_impl, pool, input, lu_out,
                      piv_out);
}

XLA_FFI_DEFINE_HANDLER(lu_handler, do_lu,
                       Ffi::Bind()
                           .Ctx<ThreadPool>()   // XLA's intra-op threads
                           .Arg<AnyBuffer>()    // matrices [..., n, m]
                           .Ret<AnyBuffer>()    // LU [..., n, m] (same dtype)
                           .Ret<AnyBuffer>());  // pivots [..., k] (int32)

XLA_FFI_Handler *get_lu_handler() { return lu_handler; }

}  // namespace rpjrt

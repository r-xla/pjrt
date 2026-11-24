// Cholesky factorisation via LAPACK potrf, over a batch of matrices that XLA's
// intra-op threads share.
//
// Input: A, shape [..., n, n], f32 or f64, every matrix stored contiguously
// in column-major order (see `MatrixBatch`). Only the triangle selected by the
// `lower` attribute is read.
//
// Outputs:
//   factor : [..., n, n], same dtype. The `lower` triangle holds L (A = L L^T)
//            or U (A = U^T U); the other triangle keeps the input's values.
//   info   : [...] int32, potrf's info per matrix: 0 on success, k > 0 if the
//            leading minor of order k is not positive definite (the factor is
//            then incomplete).
//
// A matrix that is not positive definite is not an error: the caller reads
// `info` and decides (anvl fills that factor with NaN). Returning info as an
// output rather than failing is what lets the CUDA handler stay asynchronous
// (devInfo lives on the device), and keeps both platforms on one contract.
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
static Error potrf_impl(ThreadPool &pool, bool lower, AnyBuffer input,
                        Result<AnyBuffer> out, Result<AnyBuffer> info_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(square_matrix_batch(input.dimensions(), "potrf", batch));
  const int n = batch.n;
  PJRT_RETURN_IF_ERROR(
      batch.check((*out).dimensions(), {n, n}, "potrf", "factor [..., n, n]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*info_out).dimensions(), {}, "potrf", "info [...]"));

  const T *in = static_cast<const T *>(input.untyped_data());
  T *a = static_cast<T *>((*out).untyped_data());
  int *info = static_cast<int *>((*info_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(n) * n;
  const char uplo = lower ? 'L' : 'U';
  const int lda = std::max(n, 1);

  return parallel_for(
      pool, batch.size, lapack_max_threads(),
      [&](std::int64_t b0, std::int64_t b1) {
        // potrf factors in place: seed the output with the input.
        copy_unless_aliased(a + b0 * mat, in + b0 * mat, (b1 - b0) * mat);
        for (std::int64_t b = b0; b < b1; b++) {
          Lapack<T>::potrf(&uplo, &n, a + b * mat, &lda, info + b);
          // info < 0 is an illegal argument, a bug on our side rather than
          // data.
          if (info[b] < 0) return lapack_check_info(info[b], "potrf");
        }
        return Error::Success();
      });
}

static Error do_potrf(ThreadPool pool, bool lower, AnyBuffer input,
                      Result<AnyBuffer> out, Result<AnyBuffer> info_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), potrf_impl, pool, lower, input, out,
                      info_out);
}

XLA_FFI_DEFINE_HANDLER(potrf_handler, do_potrf,
                       Ffi::Bind()
                           .Ctx<ThreadPool>()    // XLA's intra-op threads
                           .Attr<bool>("lower")  // which triangle
                           .Arg<AnyBuffer>()     // matrices [..., n, n]
                           .Ret<AnyBuffer>()     // factor [..., n, n]
                           .Ret<AnyBuffer>());   // info [...] (int32)

XLA_FFI_Handler *get_potrf_handler() { return potrf_handler; }

}  // namespace rpjrt

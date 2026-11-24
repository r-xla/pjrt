// Symmetric eigendecomposition via LAPACK syevd, over a batch of matrices that
// XLA's intra-op threads share.
//
// Input is interpreted as symmetric using only its lower triangle (we pass
// uplo = 'L'). The factorisation produces:
//   V : (n, n)  eigenvectors as columns (orthonormal)
//   W : (n,)    eigenvalues in ascending order
// such that A = V diag(W) V^T (using the lower triangle of A). Every matrix is
// stored contiguously in column-major order behind the batch dimensions (see
// `MatrixBatch`).
//
// jobz = 'V' (compute eigenvectors). For values-only the user can drop V.
//
// syevd needs *two* workspaces (real `work` of size lwork, integer `iwork` of
// size liwork) -- both queried with the lwork = -1 idiom.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ffi_common.h"
#include "ffi_handlers.h"
#include "ffi_lapack.h"
#include "ffi_thread_pool.h"

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error eigh_impl(ThreadPool &pool, AnyBuffer input,
                       Result<AnyBuffer> v_out, Result<AnyBuffer> w_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(square_matrix_batch(input.dimensions(), "eigh", batch));
  const int n = batch.n;
  PJRT_RETURN_IF_ERROR(
      batch.check((*v_out).dimensions(), {n, n}, "eigh", "V [..., n, n]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*w_out).dimensions(), {n}, "eigh", "W [..., n]"));
  if (batch.size == 0) return Error::Success();

  const T *in = static_cast<const T *>(input.untyped_data());
  T *v = static_cast<T *>((*v_out).untyped_data());
  T *w = static_cast<T *>((*w_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(n) * n;

  const char jobz = 'V';
  const char uplo = 'L';
  const int lda = std::max(n, 1);

  // The workspace query reads only the dimensions, which all matrices share.
  int lwork = -1, liwork = -1;
  T work_size;
  int iwork_size;
  int info;
  Lapack<T>::syevd(&jobz, &uplo, &n, v, &lda, w, &work_size, &lwork,
                   &iwork_size, &liwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "syevd workspace query"));
  PJRT_RETURN_IF_ERROR(workspace_to_int(work_size, "syevd", lwork));
  liwork = iwork_size;

  return parallel_for(
      pool, batch.size, lapack_max_threads(),
      [&](std::int64_t b0, std::int64_t b1) {
        // syevd overwrites A with the eigenvectors: seed V with the input.
        copy_unless_aliased(v + b0 * mat, in + b0 * mat, (b1 - b0) * mat);
        std::vector<T> work(lwork);
        std::vector<int> iwork(liwork);
        for (std::int64_t b = b0; b < b1; b++) {
          int info_b;
          Lapack<T>::syevd(&jobz, &uplo, &n, v + b * mat, &lda, w + b * n,
                           work.data(), &lwork, iwork.data(), &liwork, &info_b);
          PJRT_RETURN_IF_ERROR(lapack_check_info(info_b, "syevd"));
        }
        return Error::Success();
      });
}

static Error do_eigh(ThreadPool pool, AnyBuffer input, Result<AnyBuffer> v_out,
                     Result<AnyBuffer> w_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), eigh_impl, pool, input, v_out,
                      w_out);
}

XLA_FFI_DEFINE_HANDLER(eigh_handler, do_eigh,
                       Ffi::Bind()
                           .Ctx<ThreadPool>()   // XLA's intra-op threads
                           .Arg<AnyBuffer>()    // symmetric [..., n, n]
                           .Ret<AnyBuffer>()    // eigenvectors [..., n, n]
                           .Ret<AnyBuffer>());  // eigenvalues [..., n]

XLA_FFI_Handler *get_eigh_handler() { return eigh_handler; }

}  // namespace rpjrt

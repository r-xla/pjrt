// Singular value decomposition via LAPACK gesdd (divide-and-conquer), over a
// batch of matrices that XLA's intra-op threads share.
//
// Reduced ("economy") SVD: jobz = 'S'. For A of shape (m, n) with k = min(m,
// n):
//   U  : (m, k)
//   S  : (k,)     (always non-negative, real)
//   Vt : (k, n)
// such that A = U diag(S) Vt. Every matrix is passed as its column-major data,
// a row-major [n, m] block behind the batch dimensions (see `MatrixBatch`).
//
// gesdd needs an integer workspace `iwork` of size 8*k in addition to the
// real workspace queried via lwork = -1.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ffi_common.h"
#include "ffi_handlers.h"
#include "ffi_lapack.h"
#include "ffi_thread_pool.h"

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error svd_impl(ThreadPool &pool, AnyBuffer input,
                      Result<AnyBuffer> u_out, Result<AnyBuffer> s_out,
                      Result<AnyBuffer> vt_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(matrix_batch(input.dimensions(), "svd", batch));
  const int m = batch.m;
  const int n = batch.n;
  const int k = std::min(m, n);
  PJRT_RETURN_IF_ERROR(
      batch.check((*u_out).dimensions(), {k, m}, "svd", "U [..., k, m]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*s_out).dimensions(), {k}, "svd", "S [..., k]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*vt_out).dimensions(), {n, k}, "svd", "Vt [..., n, k]"));
  if (batch.size == 0) return Error::Success();

  const T *in = static_cast<const T *>(input.untyped_data());
  T *u = static_cast<T *>((*u_out).untyped_data());
  T *s = static_cast<T *>((*s_out).untyped_data());
  T *vt = static_cast<T *>((*vt_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const std::size_t u_mat = static_cast<std::size_t>(m) * k;
  const std::size_t vt_mat = static_cast<std::size_t>(k) * n;

  const char jobz = 'S';
  const int lda = std::max(m, 1);
  const int ldu = std::max(m, 1);
  const int ldvt = std::max(k, 1);

  // The workspace query reads only the dimensions, which all matrices share
  // (jaxlib passes no matrices to it either).
  int lwork = -1;
  T work_size;
  int info;
  Lapack<T>::gesdd(&jobz, &m, &n, nullptr, &lda, nullptr, nullptr, &ldu,
                   nullptr, &ldvt, &work_size, &lwork, nullptr, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "gesdd workspace query"));
  PJRT_RETURN_IF_ERROR(workspace_to_int(work_size, "gesdd", lwork));

  return parallel_for(
      pool, batch.size, lapack_max_threads(),
      [&](std::int64_t b0, std::int64_t b1) {
        // gesdd destroys A, and writes none of its outputs there: factor a
        // copy.
        std::vector<T> a(mat);
        std::vector<T> work(lwork);
        std::vector<int> iwork(8 * static_cast<std::size_t>(k));
        for (std::int64_t b = b0; b < b1; b++) {
          std::memcpy(a.data(), in + b * mat, mat * sizeof(T));
          int info_b;
          Lapack<T>::gesdd(&jobz, &m, &n, a.data(), &lda, s + b * k,
                           u + b * u_mat, &ldu, vt + b * vt_mat, &ldvt,
                           work.data(), &lwork, iwork.data(), &info_b);
          PJRT_RETURN_IF_ERROR(lapack_check_info(info_b, "gesdd"));
        }
        return Error::Success();
      });
}

static Error do_svd(ThreadPool pool, AnyBuffer input, Result<AnyBuffer> u_out,
                    Result<AnyBuffer> s_out, Result<AnyBuffer> vt_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), svd_impl, pool, input, u_out, s_out,
                      vt_out);
}

XLA_FFI_DEFINE_HANDLER(svd_handler, do_svd,
                       Ffi::Bind()
                           .Ctx<ThreadPool>()   // XLA's intra-op threads
                           .Arg<AnyBuffer>()    // matrices [..., n, m]
                           .Ret<AnyBuffer>()    // U  [..., k, m]
                           .Ret<AnyBuffer>()    // S  [..., k]
                           .Ret<AnyBuffer>());  // Vt [..., n, k]

XLA_FFI_Handler *get_svd_handler() { return svd_handler; }

}  // namespace rpjrt

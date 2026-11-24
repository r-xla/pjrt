// QR decomposition primitives, over a batch of matrices that XLA's intra-op
// threads share. Every m x n matrix is passed as its column-major data, a
// row-major [n, m] block behind the batch dimensions (see `MatrixBatch`); the
// shapes below are those of one matrix.
//
//   geqrf : A (m, n) -> packed (m, n), tau (k,)
//     Householder QR factorisation. The output `packed` holds R in the
//     upper triangle (including diagonal) and the Householder reflectors
//     encoding Q in the strict lower triangle. R as a separate `(k, n)`
//     tensor is recovered from `packed` via stablehlo `triu` (= iota +
//     compare + select).
//
//   orgqr : packed (m, n), tau (k,) -> Q (m, k)
//     Materialises the orthogonal factor Q from the reflectors+tau
//     produced by geqrf. cuSOLVER / LAPACK both run this in place on an
//     m x k matrix; we copy the first m*k entries of `packed` into the Q
//     output buffer and factorise there.
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

// ---- geqrf -----------------------------------------------------------------

template <typename T>
static Error geqrf_impl(ThreadPool &pool, AnyBuffer input,
                        Result<AnyBuffer> packed_out,
                        Result<AnyBuffer> tau_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(matrix_batch(input.dimensions(), "geqrf", batch));
  const int m = batch.m;
  const int n = batch.n;
  const int k = std::min(m, n);
  PJRT_RETURN_IF_ERROR(batch.check((*packed_out).dimensions(), {n, m}, "geqrf",
                                   "packed [..., n, m]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*tau_out).dimensions(), {k}, "geqrf", "tau [..., k]"));
  if (batch.size == 0) return Error::Success();

  const T *in = static_cast<const T *>(input.untyped_data());
  T *packed = static_cast<T *>((*packed_out).untyped_data());
  T *tau = static_cast<T *>((*tau_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const int lda = std::max(m, 1);

  // The workspace query reads only the dimensions, which all matrices share.
  int lwork = -1;
  T work_size;
  int info;
  Lapack<T>::geqrf(&m, &n, packed, &lda, tau, &work_size, &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "geqrf workspace query"));
  PJRT_RETURN_IF_ERROR(workspace_to_int(work_size, "geqrf", lwork));

  return parallel_for(
      pool, batch.size, lapack_max_threads(),
      [&](std::int64_t b0, std::int64_t b1) {
        // geqrf factors in place: seed the output with the input.
        copy_unless_aliased(packed + b0 * mat, in + b0 * mat, (b1 - b0) * mat);
        std::vector<T> work(lwork);
        for (std::int64_t b = b0; b < b1; b++) {
          int info_b;
          Lapack<T>::geqrf(&m, &n, packed + b * mat, &lda, tau + b * k,
                           work.data(), &lwork, &info_b);
          PJRT_RETURN_IF_ERROR(lapack_check_info(info_b, "geqrf"));
        }
        return Error::Success();
      });
}

static Error do_geqrf(ThreadPool pool, AnyBuffer input,
                      Result<AnyBuffer> packed_out, Result<AnyBuffer> tau_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), geqrf_impl, pool, input, packed_out,
                      tau_out);
}

XLA_FFI_DEFINE_HANDLER_EXPLICIT(
    geqrf_handler, do_geqrf,
    Ffi::Bind()
        .Ctx<ThreadPool>()   // XLA's intra-op threads
        .Arg<AnyBuffer>()    // matrices [..., n, m]
        .Ret<AnyBuffer>()    // packed [..., n, m]
        .Ret<AnyBuffer>());  // tau [..., k]

// ---- orgqr -----------------------------------------------------------------

template <typename T>
static Error orgqr_impl(ThreadPool &pool, AnyBuffer packed_in, AnyBuffer tau_in,
                        Result<AnyBuffer> q_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(matrix_batch(packed_in.dimensions(), "orgqr", batch));
  const int m = batch.m;
  const int n = batch.n;
  const int k = std::min(m, n);
  PJRT_RETURN_IF_ERROR(
      batch.check(tau_in.dimensions(), {k}, "orgqr", "tau [..., k]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*q_out).dimensions(), {k, m}, "orgqr", "Q [..., k, m]"));
  if (batch.size == 0) return Error::Success();

  const T *packed = static_cast<const T *>(packed_in.untyped_data());
  const T *tau = static_cast<const T *>(tau_in.untyped_data());
  T *q = static_cast<T *>((*q_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const std::size_t q_mat = static_cast<std::size_t>(m) * k;
  const int lda = std::max(m, 1);

  int lwork = -1;
  T work_size;
  int info;
  Lapack<T>::orgqr(&m, &k, &k, q, &lda, tau, &work_size, &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "orgqr workspace query"));
  PJRT_RETURN_IF_ERROR(workspace_to_int(work_size, "orgqr", lwork));

  return parallel_for(
      pool, batch.size, lapack_max_threads(),
      [&](std::int64_t b0, std::int64_t b1) {
        std::vector<T> work(lwork);
        for (std::int64_t b = b0; b < b1; b++) {
          // orgqr runs in place on an m x k matrix: the first k columns of the
          // packed matrix, its first m * k entries in column-major order.
          copy_unless_aliased(q + b * q_mat, packed + b * mat, q_mat);
          int info_b;
          Lapack<T>::orgqr(&m, &k, &k, q + b * q_mat, &lda, tau + b * k,
                           work.data(), &lwork, &info_b);
          PJRT_RETURN_IF_ERROR(lapack_check_info(info_b, "orgqr"));
        }
        return Error::Success();
      });
}

static Error do_orgqr(ThreadPool pool, AnyBuffer packed, AnyBuffer tau,
                      Result<AnyBuffer> q_out) {
  PJRT_DISPATCH_FLOAT(packed.element_type(), orgqr_impl, pool, packed, tau,
                      q_out);
}

XLA_FFI_DEFINE_HANDLER_EXPLICIT(
    orgqr_handler, do_orgqr,
    Ffi::Bind()
        .Ctx<ThreadPool>()   // XLA's intra-op threads
        .Arg<AnyBuffer>()    // packed [..., n, m]
        .Arg<AnyBuffer>()    // tau [..., k]
        .Ret<AnyBuffer>());  // Q [..., k, m]

XLA_FFI_Handler *get_geqrf_handler() { return geqrf_handler; }
XLA_FFI_Handler *get_orgqr_handler() { return orgqr_handler; }

}  // namespace rpjrt

// CUDA QR primitives via cuSOLVER, over a batch of matrices. Mirrors
// src/qr.cpp on the GPU, with every m x n matrix passed as its column-major
// data, a row-major [n, m] block behind the batch dimensions (see
// `MatrixBatch`).
//
// As on the host, QR is split into two custom calls (shapes per matrix):
//   geqrf : A (m, n) -> packed (m, n), tau (k,)
//   orgqr : packed (m, n), tau (k,) -> Q (m, k)
// R is recovered from `packed` via stablehlo `triu` outside the handler.
#include "ffi_common.h"
#include "ffi_handlers.h"

#ifndef _WIN32
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "ffi_cuda.h"
#endif

using namespace xla::ffi;

namespace rpjrt {

// ---- geqrf -----------------------------------------------------------------

#ifndef _WIN32
template <typename T>
static Error geqrf_cuda_impl(void *stream, ScratchAllocator &scratch,
                             AnyBuffer input, Result<AnyBuffer> packed_out,
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

  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  T *packed = static_cast<T *>((*packed_out).untyped_data());
  T *tau = static_cast<T *>((*tau_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const int lda = std::max(m, 1);

  // geqrf factors in place: seed the output with the input.
  if (reinterpret_cast<CUdeviceptr>(packed) != input_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(packed), input_ptr,
                      mat * batch.size * sizeof(T), stream),
        "cuMemcpyDtoDAsync (input -> packed)");
  }

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf_bs(g)(solver.handle.get(), m, n, packed, lda, &lwork),
      "cusolverDn?geqrf_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "geqrf workspace", d_work));

  // The stream runs the factorisations one after the other, so they share the
  // workspace.
  for (std::int64_t b = 0; b < batch.size; b++) {
    PJRT_RETURN_IF_GPU_ERROR(
        CuSolver<T>::geqrf(g)(solver.handle.get(), m, n, packed + b * mat, lda,
                              tau + b * k, d_work, lwork, solver.info),
        "cusolverDn?geqrf");
  }

  return Error::Success();
}
#endif  // _WIN32

static Error do_geqrf_cuda(void *stream, ScratchAllocator scratch,
                           AnyBuffer input, Result<AnyBuffer> packed_out,
                           Result<AnyBuffer> tau_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA geqrf is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), geqrf_cuda_impl, stream, scratch,
                      input, packed_out, tau_out);
#endif
}

XLA_FFI_DEFINE_HANDLER_EXPLICIT(geqrf_handler_cuda, do_geqrf_cuda,
                                Ffi::Bind()
                                    .Ctx<PlatformStream<void *>>()
                                    .Ctx<ScratchAllocator>()
                                    .Arg<AnyBuffer>()    // [..., n, m]
                                    .Ret<AnyBuffer>()    // packed [..., n, m]
                                    .Ret<AnyBuffer>());  // tau [..., k]

// ---- orgqr -----------------------------------------------------------------

#ifndef _WIN32
template <typename T>
static Error orgqr_cuda_impl(void *stream, ScratchAllocator &scratch,
                             AnyBuffer packed_in, AnyBuffer tau_in,
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

  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  const T *packed = static_cast<const T *>(packed_in.untyped_data());
  const T *tau = static_cast<const T *>(tau_in.untyped_data());
  T *q = static_cast<T *>((*q_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const std::size_t q_mat = static_cast<std::size_t>(m) * k;
  const int lda = std::max(m, 1);

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(CuSolver<T>::orgqr_bs(g)(solver.handle.get(), m, k,
                                                    k, q, lda, tau, &lwork),
                           "cusolverDn?orgqr_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "orgqr workspace", d_work));

  // The stream runs the factorisations one after the other, so they share the
  // workspace.
  for (std::int64_t b = 0; b < batch.size; b++) {
    // orgqr runs in place on an m x k matrix: the first k columns of the
    // packed matrix, its first m * k entries in column-major order. Skip the
    // copy if packed and Q already alias.
    if (q != packed) {
      PJRT_RETURN_IF_GPU_ERROR(
          g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(q + b * q_mat),
                        reinterpret_cast<CUdeviceptr>(packed + b * mat),
                        q_mat * sizeof(T), stream),
          "cuMemcpyDtoDAsync (packed -> Q)");
    }
    PJRT_RETURN_IF_GPU_ERROR(
        CuSolver<T>::orgqr(g)(solver.handle.get(), m, k, k, q + b * q_mat, lda,
                              tau + b * k, d_work, lwork, solver.info),
        "cusolverDn?orgqr");
  }

  return Error::Success();
}
#endif  // _WIN32

static Error do_orgqr_cuda(void *stream, ScratchAllocator scratch,
                           AnyBuffer packed, AnyBuffer tau_in,
                           Result<AnyBuffer> q_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA orgqr is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(packed.element_type(), orgqr_cuda_impl, stream, scratch,
                      packed, tau_in, q_out);
#endif
}

XLA_FFI_DEFINE_HANDLER_EXPLICIT(orgqr_handler_cuda, do_orgqr_cuda,
                                Ffi::Bind()
                                    .Ctx<PlatformStream<void *>>()
                                    .Ctx<ScratchAllocator>()
                                    .Arg<AnyBuffer>()    // packed [..., n, m]
                                    .Arg<AnyBuffer>()    // tau [..., k]
                                    .Ret<AnyBuffer>());  // Q [..., k, m]

XLA_FFI_Handler *get_geqrf_handler_cuda() { return geqrf_handler_cuda; }
XLA_FFI_Handler *get_orgqr_handler_cuda() { return orgqr_handler_cuda; }

}  // namespace rpjrt

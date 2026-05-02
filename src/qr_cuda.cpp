// CUDA QR decomposition via cuSOLVER. Mirrors src/qr.cpp on the GPU.
//
// The shared dlopen loader, per-stream HandleGuard, and the `Solver` prologue
// (handle + scratch-allocated devInfo) live in ffi_cusolver.h/.cpp; this file
// only contains the QR algorithm itself.
//
// On Windows the handler is still defined but always returns Unimplemented
// -- pjrt has no CUDA support on Windows, but we keep the symbol so the
// Rcpp::export wrapper resolves cleanly without `#ifdef`s.
#include <Rcpp.h>

#include "ffi_common.h"

#ifndef _WIN32
#include "ffi_cusolver.h"

#include <algorithm>
#include <cstddef>
#endif

using namespace xla::ffi;

namespace rpjrt {

#ifndef _WIN32
template <typename T>
static Error qr_cuda_impl(void *stream, ScratchAllocator &scratch,
                          AnyBuffer input, Result<AnyBuffer> q_out,
                          Result<AnyBuffer> r_out) {
  Solver solver(get_gpu_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  int k = std::min(m, n);

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto q_ptr = reinterpret_cast<CUdeviceptr>((*q_out).untyped_data());
  auto r_ptr = reinterpret_cast<CUdeviceptr>((*r_out).untyped_data());

  // Cast to size_t before multiplying to avoid int overflow for large
  // matrices (mirrors jaxlib's int64_t widening of stride math).
  std::size_t a_bytes = static_cast<std::size_t>(m) * n * sizeof(T);
  std::size_t r_bytes = static_cast<std::size_t>(k) * n * sizeof(T);
  std::size_t q_bytes = static_cast<std::size_t>(m) * k * sizeof(T);

  T *d_a;
  T *d_tau;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(m) * n, "A copy", d_a));
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(k), "tau", d_tau));

  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(d_a),
                                         input_ptr, a_bytes, stream),
                           "cuMemcpyDtoDAsync (input -> A)");

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf_bs(g)(solver.handle.get(), m, n, d_a, m, &lwork),
      "cusolverDn?geqrf_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "geqrf workspace", d_work));

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf(g)(solver.handle.get(), m, n, d_a, m, d_tau, d_work,
                            lwork, solver.info),
      "cusolverDn?geqrf");

  // Extract R: zero the output, then copy upper triangular column by column.
  PJRT_RETURN_IF_GPU_ERROR(g.memset_d8(r_ptr, 0, r_bytes, stream),
                           "cuMemsetD8Async (R)");
  CUdeviceptr d_a_ptr = reinterpret_cast<CUdeviceptr>(d_a);
  for (int j = 0; j < n; j++) {
    int elems = std::min(j + 1, k);
    std::size_t r_off = static_cast<std::size_t>(j) * k * sizeof(T);
    std::size_t a_off = static_cast<std::size_t>(j) * m * sizeof(T);
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(r_ptr + r_off, d_a_ptr + a_off,
                      static_cast<std::size_t>(elems) * sizeof(T), stream),
        "cuMemcpyDtoDAsync (R column)");
  }

  // Copy first k columns of factored A to Q output (column-major, so first
  // m*k elements), then run orgqr in-place on Q.
  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(q_ptr, d_a_ptr, q_bytes, stream),
                           "cuMemcpyDtoDAsync (A -> Q)");

  int lwork_orgqr = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::orgqr_bs(g)(solver.handle.get(), m, k, k,
                               reinterpret_cast<const T *>(q_ptr), m, d_tau,
                               &lwork_orgqr),
      "cusolverDn?orgqr_bufferSize");

  // Reuse the geqrf workspace if it's already big enough (saves an alloc
  // for the common case where geqrf needs more scratch than orgqr).
  T *work_ptr;
  if (lwork_orgqr <= lwork) {
    work_ptr = d_work;
  } else {
    PJRT_RETURN_IF_ERROR(allocate_workspace<T>(scratch,
                                               static_cast<std::size_t>(lwork_orgqr),
                                               "orgqr workspace", work_ptr));
  }

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::orgqr(g)(solver.handle.get(), m, k, k,
                            reinterpret_cast<T *>(q_ptr), m, d_tau, work_ptr,
                            lwork_orgqr, solver.info),
      "cusolverDn?orgqr");

  return Error::Success();
}
#endif // _WIN32

static Error do_qr_cuda(void *stream, ScratchAllocator scratch, AnyBuffer input,
                        Result<AnyBuffer> q_out, Result<AnyBuffer> r_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA QR is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), qr_cuda_impl, stream, scratch,
                      input, q_out, r_out);
#endif
}

XLA_FFI_DEFINE_HANDLER(qr_handler_cuda, do_qr_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Ctx<ScratchAllocator>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_qr_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::qr_handler_cuda, R_NilValue,
                           R_NilValue);
}

// CUDA QR decomposition via cuSOLVER. Mirrors src/qr.cpp on the GPU.
//
// The shared dlopen loader, DeviceMem RAII, per-stream HandleGuard, and the
// `Solver` prologue (handle + devInfo) live in ffi_cusolver.h/.cpp; this file
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
static Error qr_cuda_impl(void *stream, AnyBuffer input,
                          Result<AnyBuffer> q_out, Result<AnyBuffer> r_out) {
  Solver solver(get_gpu_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(stream));
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
  std::size_t tau_bytes = static_cast<std::size_t>(k) * sizeof(T);

  DeviceMem d_a(g), d_tau(g), d_work(g);
  PJRT_RETURN_IF_GPU_ERROR(d_a.alloc(a_bytes), "cuMemAlloc (A)");
  PJRT_RETURN_IF_GPU_ERROR(d_tau.alloc(tau_bytes), "cuMemAlloc (tau)");

  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(d_a.ptr, input_ptr, a_bytes, stream),
                           "cuMemcpyDtoDAsync (input -> A)");

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf_bs(g)(solver.handle.get(), m, n,
                               reinterpret_cast<T *>(d_a.ptr), m, &lwork),
      "cusolverDn?geqrf_bufferSize");
  PJRT_RETURN_IF_ERROR(
      allocate_workspace<T>(lwork, "cuMemAlloc (geqrf workspace)", d_work));

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf(g)(solver.handle.get(), m, n,
                            reinterpret_cast<T *>(d_a.ptr), m,
                            reinterpret_cast<T *>(d_tau.ptr),
                            reinterpret_cast<T *>(d_work.ptr), lwork,
                            reinterpret_cast<int *>(solver.info.ptr)),
      "cusolverDn?geqrf");

  // Extract R: zero the output, then copy upper triangular column by column.
  PJRT_RETURN_IF_GPU_ERROR(g.memset_d8(r_ptr, 0, r_bytes, stream),
                           "cuMemsetD8Async (R)");
  for (int j = 0; j < n; j++) {
    int elems = std::min(j + 1, k);
    std::size_t r_off = static_cast<std::size_t>(j) * k * sizeof(T);
    std::size_t a_off = static_cast<std::size_t>(j) * m * sizeof(T);
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(r_ptr + r_off, d_a.ptr + a_off,
                      static_cast<std::size_t>(elems) * sizeof(T), stream),
        "cuMemcpyDtoDAsync (R column)");
  }

  // Copy first k columns of factored A to Q output (column-major, so first
  // m*k elements), then run orgqr in-place on Q.
  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(q_ptr, d_a.ptr, q_bytes, stream),
                           "cuMemcpyDtoDAsync (A -> Q)");

  int lwork_orgqr = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::orgqr_bs(g)(solver.handle.get(), m, k, k,
                               reinterpret_cast<const T *>(q_ptr), m,
                               reinterpret_cast<const T *>(d_tau.ptr),
                               &lwork_orgqr),
      "cusolverDn?orgqr_bufferSize");

  // Reuse the geqrf workspace if it's already big enough (saves an alloc
  // for the common case where geqrf needs more scratch than orgqr).
  DeviceMem d_work2(g);
  T *work_ptr;
  if (lwork_orgqr <= lwork) {
    work_ptr = reinterpret_cast<T *>(d_work.ptr);
  } else {
    PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
        lwork_orgqr, "cuMemAlloc (orgqr workspace)", d_work2));
    work_ptr = reinterpret_cast<T *>(d_work2.ptr);
  }

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::orgqr(g)(solver.handle.get(), m, k, k,
                            reinterpret_cast<T *>(q_ptr), m,
                            reinterpret_cast<const T *>(d_tau.ptr), work_ptr,
                            lwork_orgqr,
                            reinterpret_cast<int *>(solver.info.ptr)),
      "cusolverDn?orgqr");

  return Error::Success();
}
#endif // _WIN32

static Error do_qr_cuda(void *stream, AnyBuffer input, Result<AnyBuffer> q_out,
                        Result<AnyBuffer> r_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA QR is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), qr_cuda_impl, stream, input, q_out,
                      r_out);
#endif
}

XLA_FFI_DEFINE_HANDLER(qr_handler_cuda, do_qr_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_qr_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::qr_handler_cuda, R_NilValue,
                           R_NilValue);
}

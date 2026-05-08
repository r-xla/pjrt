// CUDA QR primitives via cuSOLVER. Mirrors src/qr.cpp on the GPU.
//
// As on the host, QR is split into two custom calls:
//   geqrf : A (m, n) -> packed (m, n), tau (k,)
//   orgqr : packed (m, n), tau (k,) -> Q (m, k)
// R is recovered from `packed` via stablehlo `triu` outside the handler.
#include <Rcpp.h>

#include "ffi_common.h"

#ifndef _WIN32
#include <algorithm>
#include <cstddef>

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
  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto packed_ptr = reinterpret_cast<CUdeviceptr>((*packed_out).untyped_data());
  auto tau_ptr = reinterpret_cast<CUdeviceptr>((*tau_out).untyped_data());

  std::size_t a_bytes = static_cast<std::size_t>(m) * n * sizeof(T);

  if (packed_ptr != input_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(packed_ptr, input_ptr, a_bytes, stream),
        "cuMemcpyDtoDAsync (input -> packed)");
  }

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf_bs(g)(solver.handle.get(), m, n,
                               reinterpret_cast<T *>(packed_ptr), m, &lwork),
      "cusolverDn?geqrf_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "geqrf workspace", d_work));

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::geqrf(g)(
          solver.handle.get(), m, n, reinterpret_cast<T *>(packed_ptr), m,
          reinterpret_cast<T *>(tau_ptr), d_work, lwork, solver.info),
      "cusolverDn?geqrf");

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

XLA_FFI_DEFINE_HANDLER(geqrf_handler_cuda, do_geqrf_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Ctx<ScratchAllocator>()
                           .Arg<AnyBuffer>()    // input (m, n)
                           .Ret<AnyBuffer>()    // packed (m, n)
                           .Ret<AnyBuffer>());  // tau (k,)

// ---- orgqr -----------------------------------------------------------------

#ifndef _WIN32
template <typename T>
static Error orgqr_cuda_impl(void *stream, ScratchAllocator &scratch,
                             AnyBuffer packed, AnyBuffer tau_in,
                             Result<AnyBuffer> q_out) {
  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto dims = packed.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  int k = std::min(m, n);

  auto packed_ptr = reinterpret_cast<CUdeviceptr>(packed.untyped_data());
  auto tau_ptr = reinterpret_cast<CUdeviceptr>(tau_in.untyped_data());
  auto q_ptr = reinterpret_cast<CUdeviceptr>((*q_out).untyped_data());

  std::size_t q_bytes = static_cast<std::size_t>(m) * k * sizeof(T);

  // orgqr runs in place on an m x k matrix. Copy the first m*k entries of
  // packed (= first k columns, column-major) into the Q output, then
  // factorise there. Skip the copy if packed and Q already alias.
  if (q_ptr != packed_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(q_ptr, packed_ptr, q_bytes, stream),
                             "cuMemcpyDtoDAsync (packed -> Q)");
  }

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::orgqr_bs(g)(solver.handle.get(), m, k, k,
                               reinterpret_cast<const T *>(q_ptr), m,
                               reinterpret_cast<const T *>(tau_ptr), &lwork),
      "cusolverDn?orgqr_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "orgqr workspace", d_work));

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::orgqr(g)(
          solver.handle.get(), m, k, k, reinterpret_cast<T *>(q_ptr), m,
          reinterpret_cast<const T *>(tau_ptr), d_work, lwork, solver.info),
      "cusolverDn?orgqr");

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

XLA_FFI_DEFINE_HANDLER(orgqr_handler_cuda, do_orgqr_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Ctx<ScratchAllocator>()
                           .Arg<AnyBuffer>()    // packed reflectors (m, n)
                           .Arg<AnyBuffer>()    // tau (k,)
                           .Ret<AnyBuffer>());  // Q (m, k)

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_geqrf_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::geqrf_handler_cuda, R_NilValue,
                           R_NilValue);
}

// [[Rcpp::export]]
SEXP get_orgqr_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::orgqr_handler_cuda, R_NilValue,
                           R_NilValue);
}

// CUDA LU decomposition via cuSOLVER. Mirrors src/lu.cpp on the GPU.
#include <Rcpp.h>

#include "ffi_common.h"

#ifndef _WIN32
#include <cstddef>

#include "ffi_cuda.h"
#endif

using namespace xla::ffi;

namespace rpjrt {

#ifndef _WIN32
template <typename T>
static Error lu_cuda_impl(void *stream, ScratchAllocator &scratch,
                          AnyBuffer input, Result<AnyBuffer> lu_out,
                          Result<AnyBuffer> piv_out) {
  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto lu_ptr = reinterpret_cast<CUdeviceptr>((*lu_out).untyped_data());
  auto piv_ptr = reinterpret_cast<CUdeviceptr>((*piv_out).untyped_data());

  std::size_t a_bytes = static_cast<std::size_t>(m) * n * sizeof(T);

  if (lu_ptr != input_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(lu_ptr, input_ptr, a_bytes, stream),
                             "cuMemcpyDtoDAsync (input -> LU)");
  }

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::getrf_bs(g)(solver.handle.get(), m, n,
                               reinterpret_cast<T *>(lu_ptr), m, &lwork),
      "cusolverDn?getrf_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "getrf workspace", d_work));

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::getrf(g)(solver.handle.get(), m, n,
                            reinterpret_cast<T *>(lu_ptr), m, d_work,
                            reinterpret_cast<int *>(piv_ptr), solver.info),
      "cusolverDn?getrf");

  // devInfo is intentionally not read back: a singular matrix surfaces as
  // numerical garbage downstream rather than a launch-time error, matching
  // jaxlib's getrf path.

  return Error::Success();
}
#endif  // _WIN32

static Error do_lu_cuda(void *stream, ScratchAllocator scratch, AnyBuffer input,
                        Result<AnyBuffer> lu_out, Result<AnyBuffer> piv_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA LU is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), lu_cuda_impl, stream, scratch,
                      input, lu_out, piv_out);
#endif
}

XLA_FFI_DEFINE_HANDLER(lu_handler_cuda, do_lu_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Ctx<ScratchAllocator>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_lu_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::lu_handler_cuda, R_NilValue,
                           R_NilValue);
}

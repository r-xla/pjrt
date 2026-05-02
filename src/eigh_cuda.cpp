// CUDA symmetric eigendecomposition via cuSOLVER syevd.
//
// jobz / uplo are passed as int enum values:
//   jobz = CUSOLVER_EIG_MODE_VECTOR (1)
//   uplo = CUBLAS_FILL_MODE_LOWER   (0)
// matching the LAPACK 'V' / 'L' choice in src/eigh.cpp.
#include <Rcpp.h>

#include "ffi_common.h"

#ifndef _WIN32
#include "ffi_cusolver.h"

#include <cstddef>
#endif

using namespace xla::ffi;

namespace rpjrt {

#ifndef _WIN32
template <typename T>
static Error eigh_cuda_impl(void *stream, AnyBuffer input,
                            Result<AnyBuffer> v_out, Result<AnyBuffer> w_out) {
  Solver solver(get_gpu_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(stream));
  auto &g = solver.g;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  if (m != n)
    return Error::InvalidArgument("eigh requires a square matrix");

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto v_ptr = reinterpret_cast<CUdeviceptr>((*v_out).untyped_data());
  auto w_ptr = reinterpret_cast<CUdeviceptr>((*w_out).untyped_data());

  std::size_t a_bytes = static_cast<std::size_t>(n) * n * sizeof(T);

  // syevd overwrites its A argument with eigenvectors -- copy input into V
  // first, factorise in place there.
  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(v_ptr, input_ptr, a_bytes, stream),
                           "cuMemcpyDtoDAsync (input -> V)");

  const int jobz = 1; // CUSOLVER_EIG_MODE_VECTOR
  const int uplo = 0; // CUBLAS_FILL_MODE_LOWER

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::syevd_bs(g)(solver.handle.get(), jobz, uplo, n,
                               reinterpret_cast<const T *>(v_ptr), n,
                               reinterpret_cast<const T *>(w_ptr), &lwork),
      "cusolverDn?syevd_bufferSize");

  DeviceMem d_work(g);
  PJRT_RETURN_IF_ERROR(
      allocate_workspace<T>(lwork, "cuMemAlloc (syevd workspace)", d_work));

  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::syevd(g)(solver.handle.get(), jobz, uplo, n,
                            reinterpret_cast<T *>(v_ptr), n,
                            reinterpret_cast<T *>(w_ptr),
                            reinterpret_cast<T *>(d_work.ptr), lwork,
                            reinterpret_cast<int *>(solver.info.ptr)),
      "cusolverDn?syevd");

  return Error::Success();
}
#endif // _WIN32

static Error do_eigh_cuda(void *stream, AnyBuffer input,
                          Result<AnyBuffer> v_out, Result<AnyBuffer> w_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA eigh is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), eigh_cuda_impl, stream, input,
                      v_out, w_out);
#endif
}

XLA_FFI_DEFINE_HANDLER(eigh_handler_cuda, do_eigh_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_eigh_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::eigh_handler_cuda, R_NilValue,
                           R_NilValue);
}

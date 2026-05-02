// CUDA SVD via cuSOLVER gesvd.
//
// cuSOLVER's gesvd requires m >= n. For the m < n case the user can call
// nv_svd on the transpose and swap U <-> V; we surface a clear
// InvalidArgument error rather than silently doing this. JAX's older
// cuSOLVER path has the same restriction.
//
// jobu / jobvt are passed as 'S' (reduced); for m >= n that gives:
//   U  : (m, n)    -- ldu = m
//   S  : (n,)
//   Vt : (n, n)    -- ldvt = n
// matching the host gesdd output shapes when k = n.
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
static Error svd_cuda_impl(void *stream, ScratchAllocator &scratch,
                           AnyBuffer input, Result<AnyBuffer> u_out,
                           Result<AnyBuffer> s_out,
                           Result<AnyBuffer> vt_out) {
  Solver solver(get_gpu_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  if (m < n) {
    return Error::InvalidArgument(
        "CUDA SVD requires m >= n; transpose the input and swap U<->V "
        "for the wide case");
  }

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto u_ptr = reinterpret_cast<CUdeviceptr>((*u_out).untyped_data());
  auto s_ptr = reinterpret_cast<CUdeviceptr>((*s_out).untyped_data());
  auto vt_ptr = reinterpret_cast<CUdeviceptr>((*vt_out).untyped_data());

  std::size_t a_bytes = static_cast<std::size_t>(m) * n * sizeof(T);

  // gesvd overwrites A. Allocate a working copy so the input buffer is
  // preserved (XLA may have aliased it elsewhere).
  T *d_a;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(m) * n, "A copy", d_a));
  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(d_a),
                                         input_ptr, a_bytes, stream),
                           "cuMemcpyDtoDAsync (input -> A)");

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::gesvd_bs(g)(solver.handle.get(), m, n, &lwork),
      "cusolverDn?gesvd_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "gesvd workspace", d_work));

  // jobu / jobvt are 'S' (reduced). They're typed as signed char in the
  // cuSOLVER ABI; passing the literal char works because of integer
  // promotion at the call site.
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::gesvd(g)(solver.handle.get(), 'S', 'S', m, n, d_a, m,
                            reinterpret_cast<T *>(s_ptr),
                            reinterpret_cast<T *>(u_ptr), m,
                            reinterpret_cast<T *>(vt_ptr), n, d_work, lwork,
                            /*rwork=*/nullptr, solver.info),
      "cusolverDn?gesvd");

  return Error::Success();
}
#endif // _WIN32

static Error do_svd_cuda(void *stream, ScratchAllocator scratch,
                         AnyBuffer input, Result<AnyBuffer> u_out,
                         Result<AnyBuffer> s_out, Result<AnyBuffer> vt_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA SVD is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), svd_cuda_impl, stream, scratch,
                      input, u_out, s_out, vt_out);
#endif
}

XLA_FFI_DEFINE_HANDLER(svd_handler_cuda, do_svd_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Ctx<ScratchAllocator>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_svd_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::svd_handler_cuda, R_NilValue,
                           R_NilValue);
}

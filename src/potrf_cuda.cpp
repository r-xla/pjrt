// CUDA Cholesky factorisation via cuSOLVER potrf. Mirrors src/potrf.cpp on
// the GPU, with the same operand / result / attribute contract.
//
// uplo is passed as a cublasFillMode_t: CUBLAS_FILL_MODE_LOWER (0) or
// CUBLAS_FILL_MODE_UPPER (1). Each matrix of the batch gets its own potrf call
// whose devInfo is that matrix's slot of the `info` output, so the result
// stays on the device and the handler never synchronises the stream.
#include <Rcpp.h>

#include "ffi_common.h"

#ifndef _WIN32
#include <cstddef>
#include <cstdint>

#include "ffi_cuda.h"
#endif

using namespace xla::ffi;

namespace rpjrt {

#ifndef _WIN32
template <typename T>
static Error potrf_cuda_impl(void *stream, ScratchAllocator &scratch,
                             bool lower, AnyBuffer input, Result<AnyBuffer> out,
                             Result<AnyBuffer> info_out) {
  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  int n;
  std::int64_t batch;
  PJRT_RETURN_IF_ERROR(
      batched_square_dims(input.dimensions(), "potrf", n, batch));
  if (batch == 0) return Error::Success();

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto out_ptr = reinterpret_cast<CUdeviceptr>((*out).untyped_data());
  int *info = reinterpret_cast<int *>((*info_out).untyped_data());

  std::size_t mat = static_cast<std::size_t>(n) * n;

  if (n == 0) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memset_d8(reinterpret_cast<CUdeviceptr>(info), 0,
                    static_cast<std::size_t>(batch) * sizeof(int), stream),
        "cuMemsetD8Async (info)");
    return Error::Success();
  }

  // potrf factors in place: copy the input into the output first.
  if (out_ptr != input_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(out_ptr, input_ptr,
                      mat * static_cast<std::size_t>(batch) * sizeof(T),
                      stream),
        "cuMemcpyDtoDAsync (input -> factor)");
  }

  T *a = reinterpret_cast<T *>(out_ptr);
  const int uplo = lower ? 0 : 1;

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::potrf_bs(g)(solver.handle.get(), uplo, n, a, n, &lwork),
      "cusolverDn?potrf_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "potrf workspace", d_work));

  for (std::int64_t b = 0; b < batch; b++) {
    PJRT_RETURN_IF_GPU_ERROR(
        CuSolver<T>::potrf(g)(solver.handle.get(), uplo, n, a + b * mat, n,
                              d_work, lwork, info + b),
        "cusolverDn?potrf");
  }

  return Error::Success();
}
#endif  // _WIN32

static Error do_potrf_cuda(void *stream, ScratchAllocator scratch, bool lower,
                           AnyBuffer input, Result<AnyBuffer> out,
                           Result<AnyBuffer> info_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA potrf is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), potrf_cuda_impl, stream, scratch,
                      lower, input, out, info_out);
#endif
}

XLA_FFI_DEFINE_HANDLER(potrf_handler_cuda, do_potrf_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Ctx<ScratchAllocator>()
                           .Attr<bool>("lower")
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_potrf_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::potrf_handler_cuda, R_NilValue,
                           R_NilValue);
}

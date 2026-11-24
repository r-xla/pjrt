// CUDA LU decomposition via cuSOLVER, over a batch of matrices. Mirrors
// src/lu.cpp on the GPU, with every m x n matrix passed as its column-major
// data, a row-major [n, m] block behind the batch dimensions (see
// `MatrixBatch`).
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

#ifndef _WIN32
template <typename T>
static Error lu_cuda_impl(void *stream, ScratchAllocator &scratch,
                          AnyBuffer input, Result<AnyBuffer> lu_out,
                          Result<AnyBuffer> piv_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(matrix_batch(input.dimensions(), "lu", batch));
  const int m = batch.m;
  const int n = batch.n;
  const int k = std::min(m, n);
  PJRT_RETURN_IF_ERROR(
      batch.check((*lu_out).dimensions(), {n, m}, "lu", "LU [..., n, m]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*piv_out).dimensions(), {k}, "lu", "pivots [..., k]"));
  if (batch.size == 0) return Error::Success();

  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  T *lu = static_cast<T *>((*lu_out).untyped_data());
  int *ipiv = static_cast<int *>((*piv_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const int lda = std::max(m, 1);

  // getrf factors in place: seed the output with the input.
  if (reinterpret_cast<CUdeviceptr>(lu) != input_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(lu), input_ptr,
                      mat * batch.size * sizeof(T), stream),
        "cuMemcpyDtoDAsync (input -> LU)");
  }

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::getrf_bs(g)(solver.handle.get(), m, n, lu, lda, &lwork),
      "cusolverDn?getrf_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "getrf workspace", d_work));

  // The stream runs the factorisations one after the other, so they share the
  // workspace. devInfo is intentionally not read back: a singular matrix
  // surfaces as numerical garbage downstream rather than a launch-time error,
  // matching jaxlib's getrf path.
  for (std::int64_t b = 0; b < batch.size; b++) {
    PJRT_RETURN_IF_GPU_ERROR(
        CuSolver<T>::getrf(g)(solver.handle.get(), m, n, lu + b * mat, lda,
                              d_work, ipiv + b * k, solver.info),
        "cusolverDn?getrf");
  }

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

XLA_FFI_DEFINE_HANDLER_EXPLICIT(lu_handler_cuda, do_lu_cuda,
                                Ffi::Bind()
                                    .Ctx<PlatformStream<void *>>()
                                    .Ctx<ScratchAllocator>()
                                    .Arg<AnyBuffer>()    // [..., n, m]
                                    .Ret<AnyBuffer>()    // LU [..., n, m]
                                    .Ret<AnyBuffer>());  // pivots [..., k]

XLA_FFI_Handler *get_lu_handler_cuda() { return lu_handler_cuda; }

}  // namespace rpjrt

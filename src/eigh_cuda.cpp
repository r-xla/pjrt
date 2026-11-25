// CUDA symmetric eigendecomposition via cuSOLVER syevd, over a batch of
// matrices. Mirrors src/eigh.cpp on the GPU.
//
// jobz / uplo are passed as int enum values:
//   jobz = CUSOLVER_EIG_MODE_VECTOR (1)
//   uplo = CUBLAS_FILL_MODE_LOWER   (0)
// matching the LAPACK 'V' / 'L' choice in src/eigh.cpp.
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
static Error eigh_cuda_impl(void *stream, ScratchAllocator &scratch,
                            AnyBuffer input, Result<AnyBuffer> v_out,
                            Result<AnyBuffer> w_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(square_matrix_batch(input.dimensions(), "eigh", batch));
  const int n = batch.n;
  PJRT_RETURN_IF_ERROR(
      batch.check((*v_out).dimensions(), {n, n}, "eigh", "V [..., n, n]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*w_out).dimensions(), {n}, "eigh", "W [..., n]"));
  if (batch.size == 0) return Error::Success();

  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  T *v = static_cast<T *>((*v_out).untyped_data());
  T *w = static_cast<T *>((*w_out).untyped_data());
  const std::size_t mat = static_cast<std::size_t>(n) * n;
  const int lda = std::max(n, 1);

  // syevd overwrites its A argument with eigenvectors -- copy input into V
  // first, factorise in place there. We skip the copy when input and V
  // already point at the same buffer.
  if (reinterpret_cast<CUdeviceptr>(v) != input_ptr) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(v), input_ptr,
                      mat * batch.size * sizeof(T), stream),
        "cuMemcpyDtoDAsync (input -> V)");
  }

  const int jobz = 1;  // CUSOLVER_EIG_MODE_VECTOR
  const int uplo = 0;  // CUBLAS_FILL_MODE_LOWER

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(CuSolver<T>::syevd_bs(g)(solver.handle.get(), jobz,
                                                    uplo, n, v, lda, w, &lwork),
                           "cusolverDn?syevd_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "syevd workspace", d_work));

  // The stream runs the factorisations one after the other, so they share the
  // workspace.
  for (std::int64_t b = 0; b < batch.size; b++) {
    PJRT_RETURN_IF_GPU_ERROR(
        CuSolver<T>::syevd(g)(solver.handle.get(), jobz, uplo, n, v + b * mat,
                              lda, w + b * n, d_work, lwork, solver.info),
        "cusolverDn?syevd");
  }

  return Error::Success();
}
#endif  // _WIN32

static Error do_eigh_cuda(void *stream, ScratchAllocator scratch,
                          AnyBuffer input, Result<AnyBuffer> v_out,
                          Result<AnyBuffer> w_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA eigh is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), eigh_cuda_impl, stream, scratch,
                      input, v_out, w_out);
#endif
}

XLA_FFI_DEFINE_HANDLER_EXPLICIT(eigh_handler_cuda, do_eigh_cuda,
                                Ffi::Bind()
                                    .Ctx<PlatformStream<void *>>()
                                    .Ctx<ScratchAllocator>()
                                    .Arg<AnyBuffer>()    // [..., n, n]
                                    .Ret<AnyBuffer>()    // V [..., n, n]
                                    .Ret<AnyBuffer>());  // W [..., n]

XLA_FFI_Handler *get_eigh_handler_cuda() { return eigh_handler_cuda; }

}  // namespace rpjrt

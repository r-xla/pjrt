// CUDA SVD via cuSOLVER gesvd, over a batch of matrices. Mirrors src/svd.cpp
// on the GPU, with every m x n matrix passed as its column-major data, a
// row-major [n, m] block behind the batch dimensions (see `MatrixBatch`).
//
// gesvd requires m >= n. For m < n we use A = U D V^t  <=>  A^t = V D^t U^t:
// with `transposed`, the lowering declares the blocks of the operand / U / Vt
// column-major instead (the batch dimensions still leading), which hands over
// the column-major data of A^t, we run
// gesvd on A^t, and the same layouts on the outputs transpose back — which on
// this side just means swapping the U / Vt output slots.
//
// jobu / jobvt are passed as 'S' (reduced); for m >= n that gives:
//   U  : (m, n)    -- ldu = m
//   S  : (n,)
//   Vt : (n, n)    -- ldvt = n
// matching the host gesdd output shapes when k = n.
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
static Error svd_cuda_impl(void *stream, ScratchAllocator &scratch,
                           bool transposed, AnyBuffer input,
                           Result<AnyBuffer> u_out, Result<AnyBuffer> s_out,
                           Result<AnyBuffer> vt_out) {
  MatrixBatch batch;
  PJRT_RETURN_IF_ERROR(matrix_batch(input.dimensions(), "svd", batch));
  const int k = std::min(batch.m, batch.n);
  PJRT_RETURN_IF_ERROR(
      batch.check((*u_out).dimensions(), {k, batch.m}, "svd", "U [..., k, m]"));
  PJRT_RETURN_IF_ERROR(
      batch.check((*s_out).dimensions(), {k}, "svd", "S [..., k]"));
  PJRT_RETURN_IF_ERROR(batch.check((*vt_out).dimensions(), {batch.n, k}, "svd",
                                   "Vt [..., n, k]"));
  if (batch.size == 0) return Error::Success();

  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  // The matrix gesvd factors, of A or of A^t, with m >= n.
  const int m = transposed ? batch.n : batch.m;
  const int n = transposed ? batch.m : batch.n;

  const T *in = static_cast<const T *>(input.untyped_data());
  T *s = static_cast<T *>((*s_out).untyped_data());
  T *u_data = static_cast<T *>((*u_out).untyped_data());
  T *vt_data = static_cast<T *>((*vt_out).untyped_data());
  T *u = transposed ? vt_data : u_data;
  T *vt = transposed ? u_data : vt_data;

  // Cast m to size_t before multiplying: dim_to_int only guarantees each
  // dimension individually fits in int, not their product. e.g. a
  // 50000 x 50000 matrix has m, n in int range but m*n overflows int.
  // gesvd's U is m x n and its Vt n x n.
  const std::size_t mat = static_cast<std::size_t>(m) * n;
  const std::size_t vt_mat = static_cast<std::size_t>(n) * n;
  const int lda = std::max(m, 1);
  const int ldvt = std::max(n, 1);

  // gesvd destroys A in place AND requires A and U to be non-overlapping
  // when jobu='S'
  T *d_a;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(scratch, mat, "A copy", d_a));

  int lwork = 0;
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::gesvd_bs(g)(solver.handle.get(), m, n, &lwork),
      "cusolverDn?gesvd_bufferSize");

  T *d_work;
  PJRT_RETURN_IF_ERROR(allocate_workspace<T>(
      scratch, static_cast<std::size_t>(lwork), "gesvd workspace", d_work));

  // The stream runs the factorisations one after the other, so they share the
  // copy of A and the workspace.
  for (std::int64_t b = 0; b < batch.size; b++) {
    PJRT_RETURN_IF_GPU_ERROR(
        g.memcpy_dtod(reinterpret_cast<CUdeviceptr>(d_a),
                      reinterpret_cast<CUdeviceptr>(in + b * mat),
                      mat * sizeof(T), stream),
        "cuMemcpyDtoDAsync (input -> A)");
    // jobu / jobvt are 'S' (reduced). cuSOLVER's prototype types these as
    // signed char; the C++ char literal 'S' (byte 83) is implicitly
    // converted to signed char at the call site, no truncation.
    PJRT_RETURN_IF_GPU_ERROR(
        CuSolver<T>::gesvd(g)(solver.handle.get(), 'S', 'S', m, n, d_a, lda,
                              s + b * n, u + b * mat, lda, vt + b * vt_mat,
                              ldvt, d_work, lwork, /*rwork=*/nullptr,
                              solver.info),
        "cusolverDn?gesvd");
  }

  return Error::Success();
}
#endif  // _WIN32

static Error do_svd_cuda(void *stream, ScratchAllocator scratch,
                         bool transposed, AnyBuffer input,
                         Result<AnyBuffer> u_out, Result<AnyBuffer> s_out,
                         Result<AnyBuffer> vt_out) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA SVD is not supported on Windows");
#else
  PJRT_DISPATCH_FLOAT(input.element_type(), svd_cuda_impl, stream, scratch,
                      transposed, input, u_out, s_out, vt_out);
#endif
}

XLA_FFI_DEFINE_HANDLER_EXPLICIT(svd_handler_cuda, do_svd_cuda,
                                Ffi::Bind()
                                    .Ctx<PlatformStream<void *>>()
                                    .Ctx<ScratchAllocator>()
                                    .Attr<bool>("transposed")
                                    .Arg<AnyBuffer>()    // [..., n, m]
                                    .Ret<AnyBuffer>()    // U  [..., k, m]
                                    .Ret<AnyBuffer>()    // S  [..., k]
                                    .Ret<AnyBuffer>());  // Vt [..., n, k]

XLA_FFI_Handler *get_svd_handler_cuda() { return svd_handler_cuda; }

}  // namespace rpjrt

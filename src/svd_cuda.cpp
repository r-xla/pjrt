// CUDA SVD via cuSOLVER gesvd. Mirrors src/svd.cpp on the GPU, with every
// matrix passed as its column-major data: an m x n matrix is a row-major
// [n, m] buffer (see `matrix_dims()`).
//
// gesvd requires m >= n. For m < n we use A = U D V^t  <=>  A^t = V D^t U^t:
// with `transposed`, the lowering declares the operand / U / Vt buffers
// column-major instead, which hands over the column-major data of A^t, we run
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
#include <cstddef>

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
  Solver solver(get_cuda_libs());
  PJRT_RETURN_IF_ERROR(solver.begin(scratch, stream));
  auto &g = solver.g;

  int m_orig, n_orig;
  PJRT_RETURN_IF_ERROR(matrix_dims(input.dimensions(), "svd", m_orig, n_orig));
  int m = transposed ? n_orig : m_orig;
  int n = transposed ? m_orig : n_orig;

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  T *u_data = static_cast<T *>((*u_out).untyped_data());
  T *s_data = static_cast<T *>((*s_out).untyped_data());
  T *vt_data = static_cast<T *>((*vt_out).untyped_data());

  T *u_param = transposed ? vt_data : u_data;
  T *vt_param = transposed ? u_data : vt_data;

  // Cast m to size_t before multiplying: dim_to_int only guarantees each
  // dimension individually fits in int, not their product. e.g. a
  // 50000 x 50000 matrix has m, n in int range but m*n overflows int.
  std::size_t a_bytes = static_cast<std::size_t>(m) * n * sizeof(T);

  // gesvd destroys A in place AND requires A and U to be non-overlapping
  // when jobu='S'
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

  // jobu / jobvt are 'S' (reduced). cuSOLVER's prototype types these as
  // signed char; the C++ char literal 'S' (byte 83) is implicitly
  // converted to signed char at the call site, no truncation.
  PJRT_RETURN_IF_GPU_ERROR(
      CuSolver<T>::gesvd(g)(solver.handle.get(), 'S', 'S', m, n, d_a, m, s_data,
                            u_param, m, vt_param, n, d_work, lwork,
                            /*rwork=*/nullptr, solver.info),
      "cusolverDn?gesvd");

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
                                    .Arg<AnyBuffer>()
                                    .Ret<AnyBuffer>()
                                    .Ret<AnyBuffer>()
                                    .Ret<AnyBuffer>());

XLA_FFI_Handler *get_svd_handler_cuda() { return svd_handler_cuda; }

}  // namespace rpjrt

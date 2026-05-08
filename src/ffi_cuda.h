// CUDA-side FFI support: cuSOLVER function table, generic CUDA driver
// helpers (memcpy / memset / stream-sync), and the per-stream cuSOLVER
// handle pool. All entries are dynamically loaded via dlopen so the R
// package can be built without the CUDA SDK headers and run on machines
// without a CUDA install. Mirrors the role of
// jaxlib/gpu/solver_kernels_ffi.cc + jaxlib/gpu/solver_handle_pool.cc,
// adapted to a runtime-link-only model.
//
// The cuSOLVER bits are used by the linalg kernels (qr, lu, svd, eigh).
// The generic driver helpers are also used outside the linalg path -- e.g.
// print_tensor's CUDA handler in ffi.cpp uses `memcpy_dtoh` and
// `stream_synchronize` to pull a buffer to the host before formatting.
//
// Workspace, devInfo, and per-call working buffers are allocated through
// XLA's ffi::ScratchAllocator (BFC-pool-backed, integrated with XLA's
// memory accounting). The dlopen surface only covers cuSOLVER itself plus
// the memcpy/memset/sync helpers cuSOLVER doesn't provide; raw
// cuMemAlloc/cuMemFree are not needed.
//
// Only the non-Windows half is meaningful; on Windows there is no CUDA,
// and dlopen is POSIX-only.
#pragma once

#include "ffi_common.h"
#include "xla/ffi/api/ffi.h"

#ifndef _WIN32

#include <cstddef>
#include <cstdint>
#include <string>

namespace rpjrt {

// Opaque CUDA / cuSOLVER types (no SDK headers needed). uintptr_t matches
// the typedef used elsewhere in pjrt (see ffi.cpp).
using CUdeviceptr = std::uintptr_t;

// Status-check helper for CUDA driver / cuSOLVER calls. Every API call returns
// an int; we propagate non-zero values as Error::Internal annotated with the
// site name. Mirrors jaxlib's JAX_FFI_RETURN_IF_GPU_ERROR.
#define PJRT_RETURN_IF_GPU_ERROR(expr, what)                                   \
  do {                                                                         \
    int _status = (expr);                                                      \
    if (_status != 0) {                                                        \
      return xla::ffi::Error::Internal(std::string(what) +                     \
                                       " failed with status = " +              \
                                       std::to_string(_status));               \
    }                                                                          \
  } while (0)

// Function pointers for the cuSOLVER + CUDA driver entry points the package
// uses. New ops add their entries here and to the loader in ffi_cuda.cpp.
//
// Naming: `s_` / `d_` prefixes are float / double precision. The operation
// stems (`geqrf`, `orgqr`, `getrf`, `gesvd`, `syevd`) follow the same LAPACK
// naming convention documented at the top of `ffi_lapack.h` -- cuSOLVER
// mirrors LAPACK's interface, so `s_geqrf` corresponds to `cusolverDnSgeqrf`
// (= LAPACK's `sgeqrf` on the GPU). The `_bs` suffix stands for "bufferSize"
// -- the cuSOLVER workspace-query companion of each routine (e.g.
// `cusolverDnSgeqrf_bufferSize`). It returns the optimal `lwork` for the
// given dimensions via its trailing `int *` out-parameter; the actual
// computation routine then runs with a workspace of that size.
//
// IMPORTANT -- pointer arguments and address space:
//   The typed `float *` / `double *` / `int *` matrix and workspace pointers
//   in the cuSOLVER routine signatures are *device* addresses. The C ABI
//   doesn't have a distinct "device pointer" type, so cuSOLVER reuses
//   plain `T *` and documents the address space in its API tables -- every
//   parameter row in the cuSOLVER docs has a "Memory" column that says
//   `host` or `device`, and for matrix data / workspace / `devInfo` the
//   answer is always `device`. Internally pjrt stores these addresses as
//   `CUdeviceptr` (= `uintptr_t`) so they aren't accidentally treated as
//   host-dereferenceable, and casts back to `T *` only at the cuSOLVER
//   call site to match the ABI. The `*_bufferSize` `int *` out-parameter
//   is the one exception -- it's documented as `host` and is filled in
//   synchronously from the call.
struct CudaLibs {
  // cuSOLVER dense handle management. The `dn_` prefix mirrors cuSOLVER's
  // `cusolverDn` (Dense) namespace; every dense routine takes a handle as its
  // first argument.
  //   dn_create      -> cusolverDnCreate(handle*): allocates a cuSOLVER dense
  //                     context. Handles are pooled per stream via HandleGuard
  //                     (see borrow_solver_handle in ffi_cuda.cpp).
  //   dn_destroy     -> cusolverDnDestroy(handle): frees the context. Called
  //                     when the pool tears down.
  //   dn_set_stream  -> cusolverDnSetStream(handle, stream): binds the handle
  //                     to a CUDA stream so subsequent calls launch on it.
  //                     Must be called before each use because handles aren't
  //                     safe to share across streams concurrently.
  int (*dn_create)(void **);
  int (*dn_destroy)(void *);
  int (*dn_set_stream)(void *, void *);

  // QR.
  int (*s_geqrf_bs)(void *, int, int, float *, int, int *);
  int (*s_geqrf)(void *, int, int, float *, int, float *, float *, int, int *);
  int (*d_geqrf_bs)(void *, int, int, double *, int, int *);
  int (*d_geqrf)(void *, int, int, double *, int, double *, double *, int,
                 int *);
  int (*s_orgqr_bs)(void *, int, int, int, const float *, int, const float *,
                    int *);
  int (*s_orgqr)(void *, int, int, int, float *, int, const float *, float *,
                 int, int *);
  int (*d_orgqr_bs)(void *, int, int, int, const double *, int, const double *,
                    int *);
  int (*d_orgqr)(void *, int, int, int, double *, int, const double *,
                 double *, int, int *);

  // LU. ipiv and devInfo are device int32; ipiv is 1-based row indices.
  int (*s_getrf_bs)(void *, int, int, float *, int, int *);
  int (*s_getrf)(void *, int, int, float *, int, float *, int *, int *);
  int (*d_getrf_bs)(void *, int, int, double *, int, int *);
  int (*d_getrf)(void *, int, int, double *, int, double *, int *, int *);

  // SVD via cusolverDn?gesvd. The bufferSize variant takes only (handle, m, n)
  // and returns the worst-case workspace; jobu/jobvt are not part of the
  // workspace query. rwork is unused for real precisions (pass nullptr).
  int (*s_gesvd_bs)(void *, int, int, int *);
  int (*d_gesvd_bs)(void *, int, int, int *);
  int (*s_gesvd)(void *, signed char, signed char, int, int, float *, int,
                 float *, float *, int, float *, int, float *, int, float *,
                 int *);
  int (*d_gesvd)(void *, signed char, signed char, int, int, double *, int,
                 double *, double *, int, double *, int, double *, int,
                 double *, int *);

  // Symmetric/Hermitian eigendecomposition. jobz/uplo are cusolverEigMode_t /
  // cublasFillMode_t (both int enums). We always pass jobz = 1 (vectors) and
  // uplo = 0 (lower) -- see eigh_cuda.cpp.
  int (*s_syevd_bs)(void *, int, int, int, const float *, int, const float *,
                    int *);
  int (*d_syevd_bs)(void *, int, int, int, const double *, int, const double *,
                    int *);
  int (*s_syevd)(void *, int, int, int, float *, int, float *, float *, int,
                 int *);
  int (*d_syevd)(void *, int, int, int, double *, int, double *, double *, int,
                 int *);

  // CUDA driver helpers. Allocation goes through ffi::ScratchAllocator, but
  // memcpy / memset / stream-sync still need driver entry points.
  // memcpy_dtoh / stream_synchronize are used outside the linalg kernels too
  // (e.g. by print_tensor's CUDA handler in ffi.cpp).
  //
  //   memcpy_dtod         -> cuMemcpyDtoDAsync_v2(dst, src, n_bytes, stream):
  //                          async device-to-device copy of `n_bytes`. Used to
  //                          duplicate input buffers before factorisation
  //                          (LAPACK/cuSOLVER overwrite their input in place).
  //   memset_d8           -> cuMemsetD8Async(dst, value, n_bytes, stream):
  //                          async fill of `n_bytes` device bytes with the
  //                          single-byte `value`. Used e.g. to zero the
  //                          strict-upper triangle of R / strict-lower of L
  //                          when extracting factors from cuSOLVER output.
  //   memcpy_dtoh         -> cuMemcpyDtoHAsync_v2(host, dev, n_bytes, stream):
  //                          async device-to-host copy. Used by print_tensor
  //                          to pull the buffer back to the host before
  //                          formatting; the linalg kernels avoid it (output
  //                          stays on the device).
  //   stream_synchronize  -> cuStreamSynchronize(stream): block the host
  //                          until all work previously enqueued on `stream`
  //                          has completed. Required after a D2H copy when
  //                          the host needs to read the destination buffer
  //                          synchronously (print_tensor). Linalg kernels do
  //                          not call it -- they let XLA's stream ordering
  //                          handle the wait.
  int (*memcpy_dtod)(CUdeviceptr, CUdeviceptr, std::size_t, void *);
  int (*memset_d8)(CUdeviceptr, unsigned char, std::size_t, void *);
  int (*memcpy_dtoh)(void *, CUdeviceptr, std::size_t, void *);
  int (*stream_synchronize)(void *);

  bool loaded = false;
};

CudaLibs &get_cuda_libs();

// Borrowed cuSOLVER handle, returned to the per-stream pool on destruction.
class HandleGuard {
public:
  HandleGuard() = default;
  HandleGuard(void *stream, void *handle) : stream_(stream), handle_(handle) {}
  HandleGuard(HandleGuard &&o) noexcept;
  HandleGuard &operator=(HandleGuard &&o) noexcept;
  ~HandleGuard();
  HandleGuard(const HandleGuard &) = delete;
  HandleGuard &operator=(const HandleGuard &) = delete;
  void *get() const { return handle_; }

private:
  void release();
  void *stream_ = nullptr;
  void *handle_ = nullptr;
};

xla::ffi::Error borrow_solver_handle(CudaLibs &g, void *stream,
                                     HandleGuard &out);

// Bundled prologue for a CUDA linalg kernel: a borrowed cuSOLVER handle on
// `stream`, plus a scratch-allocated device `int` for `devInfo` (every
// cuSOLVER routine wants one). All four built-in linalg kernels open with the
// same three steps -- loaded-check, handle borrow, info alloc -- and `Solver`
// rolls them into one initialiser.
//
// `info` is owned by the caller's ScratchAllocator and freed when the FFI
// handler returns. Mirrors jaxlib's GeqrfImpl prologue in
// solver_kernels_ffi.cc (which also threads ScratchAllocator through every
// solver kernel).
struct Solver {
  CudaLibs &g;
  HandleGuard handle;

  // Device-side `int` written by every cuSOLVER routine as its `devInfo`
  // out-parameter.
  // We always ignore (just like jax) it because if forces host synchronization,
  // which comes at a performance cost.
  int *info = nullptr;

  explicit Solver(CudaLibs &g) : g(g) {}

  // Borrow a handle for `stream` and allocate devInfo from `scratch`. Call
  // once per kernel invocation, before any cuSOLVER calls.
  xla::ffi::Error begin(xla::ffi::ScratchAllocator &scratch, void *stream);
};

// Allocate `n_elements * sizeof(T)` bytes from `scratch`, with a site-name
// annotation. Centralises the size widening and the optional -> Error
// translation so each kernel doesn't open-code it per workspace.
template <typename T>
xla::ffi::Error allocate_workspace(xla::ffi::ScratchAllocator &scratch,
                                   std::size_t n_elements, const char *name,
                                   T *&out) {
  auto p = scratch.Allocate(n_elements * sizeof(T));
  if (!p.has_value()) {
    return xla::ffi::Error::Internal(std::string(name) +
                                     " scratch allocation failed");
  }
  out = static_cast<T *>(*p);
  return xla::ffi::Error::Success();
}

// Per-precision dispatch trait for cuSOLVER routines. Modelled on jaxlib's
// `solver::Geqrf<T>` / `solver::Getrf<T>` ... wrappers.
template <typename T> struct CuSolver;

template <> struct CuSolver<float> {
  static auto geqrf_bs(CudaLibs &g) { return g.s_geqrf_bs; }
  static auto geqrf(CudaLibs &g) { return g.s_geqrf; }
  static auto orgqr_bs(CudaLibs &g) { return g.s_orgqr_bs; }
  static auto orgqr(CudaLibs &g) { return g.s_orgqr; }
  static auto getrf_bs(CudaLibs &g) { return g.s_getrf_bs; }
  static auto getrf(CudaLibs &g) { return g.s_getrf; }
  static auto gesvd_bs(CudaLibs &g) { return g.s_gesvd_bs; }
  static auto gesvd(CudaLibs &g) { return g.s_gesvd; }
  static auto syevd_bs(CudaLibs &g) { return g.s_syevd_bs; }
  static auto syevd(CudaLibs &g) { return g.s_syevd; }
};

template <> struct CuSolver<double> {
  static auto geqrf_bs(CudaLibs &g) { return g.d_geqrf_bs; }
  static auto geqrf(CudaLibs &g) { return g.d_geqrf; }
  static auto orgqr_bs(CudaLibs &g) { return g.d_orgqr_bs; }
  static auto orgqr(CudaLibs &g) { return g.d_orgqr; }
  static auto getrf_bs(CudaLibs &g) { return g.d_getrf_bs; }
  static auto getrf(CudaLibs &g) { return g.d_getrf; }
  static auto gesvd_bs(CudaLibs &g) { return g.d_gesvd_bs; }
  static auto gesvd(CudaLibs &g) { return g.d_gesvd; }
  static auto syevd_bs(CudaLibs &g) { return g.d_syevd_bs; }
  static auto syevd(CudaLibs &g) { return g.d_syevd; }
};

} // namespace rpjrt

#endif // _WIN32

// cuSOLVER + CUDA-driver function table, dynamically loaded via dlopen so the
// R package can be built without the CUDA SDK headers and run on machines
// without a CUDA install. Mirrors the role of jaxlib/gpu/solver_kernels_ffi.cc
// + jaxlib/gpu/solver_handle_pool.cc, adapted to a runtime-link-only model.
//
// Workspace, devInfo, and per-call working buffers are allocated through XLA's
// ffi::ScratchAllocator (BFC-pool-backed, integrated with XLA's memory
// accounting). The dlopen surface only covers cuSOLVER itself plus the
// memcpy/memset/sync helpers cuSOLVER doesn't provide; raw cuMemAlloc/cuMemFree
// are no longer needed.
//
// Only the non-Windows half is meaningful; on Windows there is no CUDA, and
// dlopen is POSIX-only.
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
// uses. New ops add their entries here and to the loader in ffi_cusolver.cpp.
struct GpuLibs {
  // cuSOLVER handle management.
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
  int (*memcpy_dtod)(CUdeviceptr, CUdeviceptr, std::size_t, void *);
  int (*memset_d8)(CUdeviceptr, unsigned char, std::size_t, void *);

  bool loaded = false;
};

GpuLibs &get_gpu_libs();

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

xla::ffi::Error borrow_solver_handle(GpuLibs &g, void *stream,
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
  GpuLibs &g;
  HandleGuard handle;
  int *info = nullptr;
  explicit Solver(GpuLibs &g) : g(g) {}

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
  static auto geqrf_bs(GpuLibs &g) { return g.s_geqrf_bs; }
  static auto geqrf(GpuLibs &g) { return g.s_geqrf; }
  static auto orgqr_bs(GpuLibs &g) { return g.s_orgqr_bs; }
  static auto orgqr(GpuLibs &g) { return g.s_orgqr; }
  static auto getrf_bs(GpuLibs &g) { return g.s_getrf_bs; }
  static auto getrf(GpuLibs &g) { return g.s_getrf; }
  static auto gesvd_bs(GpuLibs &g) { return g.s_gesvd_bs; }
  static auto gesvd(GpuLibs &g) { return g.s_gesvd; }
  static auto syevd_bs(GpuLibs &g) { return g.s_syevd_bs; }
  static auto syevd(GpuLibs &g) { return g.s_syevd; }
};

template <> struct CuSolver<double> {
  static auto geqrf_bs(GpuLibs &g) { return g.d_geqrf_bs; }
  static auto geqrf(GpuLibs &g) { return g.d_geqrf; }
  static auto orgqr_bs(GpuLibs &g) { return g.d_orgqr_bs; }
  static auto orgqr(GpuLibs &g) { return g.d_orgqr; }
  static auto getrf_bs(GpuLibs &g) { return g.d_getrf_bs; }
  static auto getrf(GpuLibs &g) { return g.d_getrf; }
  static auto gesvd_bs(GpuLibs &g) { return g.d_gesvd_bs; }
  static auto gesvd(GpuLibs &g) { return g.d_gesvd; }
  static auto syevd_bs(GpuLibs &g) { return g.d_syevd_bs; }
  static auto syevd(GpuLibs &g) { return g.d_syevd; }
};

} // namespace rpjrt

#endif // _WIN32

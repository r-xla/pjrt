// Implementation of the shared cuSOLVER infrastructure: dlopen-based loader
// and a per-stream handle pool. All cuSOLVER-backed kernels (qr, lu, svd,
// eigh) use these singletons so they share one set of loaded function
// pointers and one handle pool. Device-memory allocation goes through XLA's
// ffi::ScratchAllocator, not via this loader.
#include "ffi_cusolver.h"

#ifndef _WIN32

#include <dlfcn.h>

#include <map>
#include <mutex>
#include <vector>

using namespace xla::ffi;

namespace rpjrt {

template <typename T> static T load_sym(void *lib, const char *name) {
  return reinterpret_cast<T>(dlsym(lib, name));
}

GpuLibs &get_gpu_libs() {
  static GpuLibs g;
  if (g.loaded)
    return g;

  // Probe a couple of candidates: SDK installs ship the unversioned symlink,
  // runtime-only installs (typical in containers) only ship the SONAME.
  void *cusolver = dlopen("libcusolver.so", RTLD_LAZY);
  if (!cusolver)
    cusolver = dlopen("libcusolver.so.11", RTLD_LAZY);
  if (!cusolver)
    return g;

  void *cuda = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!cuda)
    return g;

  g.dn_create = load_sym<decltype(g.dn_create)>(cusolver, "cusolverDnCreate");
  g.dn_destroy =
      load_sym<decltype(g.dn_destroy)>(cusolver, "cusolverDnDestroy");
  g.dn_set_stream =
      load_sym<decltype(g.dn_set_stream)>(cusolver, "cusolverDnSetStream");

  g.s_geqrf_bs =
      load_sym<decltype(g.s_geqrf_bs)>(cusolver, "cusolverDnSgeqrf_bufferSize");
  g.s_geqrf = load_sym<decltype(g.s_geqrf)>(cusolver, "cusolverDnSgeqrf");
  g.d_geqrf_bs =
      load_sym<decltype(g.d_geqrf_bs)>(cusolver, "cusolverDnDgeqrf_bufferSize");
  g.d_geqrf = load_sym<decltype(g.d_geqrf)>(cusolver, "cusolverDnDgeqrf");
  g.s_orgqr_bs =
      load_sym<decltype(g.s_orgqr_bs)>(cusolver, "cusolverDnSorgqr_bufferSize");
  g.s_orgqr = load_sym<decltype(g.s_orgqr)>(cusolver, "cusolverDnSorgqr");
  g.d_orgqr_bs =
      load_sym<decltype(g.d_orgqr_bs)>(cusolver, "cusolverDnDorgqr_bufferSize");
  g.d_orgqr = load_sym<decltype(g.d_orgqr)>(cusolver, "cusolverDnDorgqr");

  g.s_getrf_bs =
      load_sym<decltype(g.s_getrf_bs)>(cusolver, "cusolverDnSgetrf_bufferSize");
  g.s_getrf = load_sym<decltype(g.s_getrf)>(cusolver, "cusolverDnSgetrf");
  g.d_getrf_bs =
      load_sym<decltype(g.d_getrf_bs)>(cusolver, "cusolverDnDgetrf_bufferSize");
  g.d_getrf = load_sym<decltype(g.d_getrf)>(cusolver, "cusolverDnDgetrf");

  g.s_gesvd_bs =
      load_sym<decltype(g.s_gesvd_bs)>(cusolver, "cusolverDnSgesvd_bufferSize");
  g.d_gesvd_bs =
      load_sym<decltype(g.d_gesvd_bs)>(cusolver, "cusolverDnDgesvd_bufferSize");
  g.s_gesvd = load_sym<decltype(g.s_gesvd)>(cusolver, "cusolverDnSgesvd");
  g.d_gesvd = load_sym<decltype(g.d_gesvd)>(cusolver, "cusolverDnDgesvd");

  g.s_syevd_bs =
      load_sym<decltype(g.s_syevd_bs)>(cusolver, "cusolverDnSsyevd_bufferSize");
  g.d_syevd_bs =
      load_sym<decltype(g.d_syevd_bs)>(cusolver, "cusolverDnDsyevd_bufferSize");
  g.s_syevd = load_sym<decltype(g.s_syevd)>(cusolver, "cusolverDnSsyevd");
  g.d_syevd = load_sym<decltype(g.d_syevd)>(cusolver, "cusolverDnDsyevd");

  g.memcpy_dtod =
      load_sym<decltype(g.memcpy_dtod)>(cuda, "cuMemcpyDtoDAsync_v2");
  g.memset_d8 = load_sym<decltype(g.memset_d8)>(cuda, "cuMemsetD8Async");

  g.loaded = true;
  return g;
}

// Per-stream cuSOLVER handle pool.
//
// cuSOLVER handles are not safe to share across streams: cusolverDnSetStream
// rebinds the handle, racing with concurrent launches issued from another
// stream. We mirror jaxlib's SolverHandlePool (jaxlib/gpu/solver_handle_pool.cc):
// a mutex-guarded free-list of handles per stream, RAII-returned to the pool
// when the borrow goes out of scope. Handles are pooled forever, never
// destroyed (acceptable for a process-wide resource).
namespace {
struct SolverHandlePool {
  std::mutex mu;
  std::map<void *, std::vector<void *>> free_handles;

  static SolverHandlePool &instance() {
    static SolverHandlePool p;
    return p;
  }
};
} // namespace

HandleGuard::HandleGuard(HandleGuard &&o) noexcept
    : stream_(o.stream_), handle_(o.handle_) {
  o.handle_ = nullptr;
}

HandleGuard &HandleGuard::operator=(HandleGuard &&o) noexcept {
  if (this != &o) {
    release();
    stream_ = o.stream_;
    handle_ = o.handle_;
    o.handle_ = nullptr;
  }
  return *this;
}

HandleGuard::~HandleGuard() { release(); }

void HandleGuard::release() {
  if (!handle_)
    return;
  auto &pool = SolverHandlePool::instance();
  std::lock_guard<std::mutex> lock(pool.mu);
  pool.free_handles[stream_].push_back(handle_);
  handle_ = nullptr;
}

Error borrow_solver_handle(GpuLibs &g, void *stream, HandleGuard &out) {
  auto &pool = SolverHandlePool::instance();
  void *handle = nullptr;
  {
    std::lock_guard<std::mutex> lock(pool.mu);
    auto &vec = pool.free_handles[stream];
    if (!vec.empty()) {
      handle = vec.back();
      vec.pop_back();
    }
  }
  if (!handle) {
    PJRT_RETURN_IF_GPU_ERROR(g.dn_create(&handle), "cusolverDnCreate");
  }
  if (stream) {
    int s = g.dn_set_stream(handle, stream);
    if (s != 0) {
      // Return the handle to the pool so it isn't lost on this error path.
      std::lock_guard<std::mutex> lock(pool.mu);
      pool.free_handles[stream].push_back(handle);
      return Error::Internal("cusolverDnSetStream failed with status = " +
                             std::to_string(s));
    }
  }
  out = HandleGuard(stream, handle);
  return Error::Success();
}

Error Solver::begin(ScratchAllocator &scratch, void *stream) {
  if (!g.loaded)
    return Error::Internal("CUDA/cuSOLVER libraries not available");
  PJRT_RETURN_IF_ERROR(borrow_solver_handle(g, stream, handle));
  auto p = scratch.Allocate(sizeof(int));
  if (!p.has_value())
    return Error::Internal("scratch allocation failed (devInfo)");
  info = static_cast<int *>(*p);
  return Error::Success();
}

} // namespace rpjrt

#endif // _WIN32

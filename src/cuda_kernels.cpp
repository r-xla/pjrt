// Driver-API loader for the kernels pjrt precompiles into its fatbins.
// See cuda_kernels.h for why this uses libcuda.so.1 rather than libcudart.
#include <Rcpp.h>

#include "cuda_kernels.h"

namespace rpjrt {

bool cuda_kernels_available() { return kCudaFatbinCount > 0; }

}  // namespace rpjrt

// [[Rcpp::export]]
bool impl_cuda_kernels_available() { return rpjrt::cuda_kernels_available(); }

#ifndef _WIN32

#include <dlfcn.h>

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace xla::ffi;

namespace rpjrt {
namespace {

// CUresult values we need to distinguish by name rather than just report.
constexpr int kCudaSuccess = 0;
constexpr int kCudaErrorNotFound = 500;  // CUDA_ERROR_NOT_FOUND

// The subset of the CUDA driver API needed to load a fatbin and launch out of
// it. Deliberately separate from CudaLibs in ffi_cuda.h: that table gives up
// when libcusolver is missing, and a custom kernel has no business depending
// on cuSOLVER being installed.
struct Driver {
  int (*ctx_get_current)(void **);
  int (*module_load_data)(void **, const void *);
  int (*module_get_function)(void **, void *, const char *);
  int (*launch_kernel)(void *, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       void *, void **, void **);
  int (*get_error_name)(int, const char **);
  bool loaded = false;
};

Driver &driver() {
  static Driver d;
  static std::once_flag once;
  std::call_once(once, [&] {
    // The PJRT plugin has already pulled libcuda into the process by the time
    // an FFI handler runs; this dlopen just gets us a handle on it.
    void *lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lib) return;

    auto sym = [lib](const char *name) { return dlsym(lib, name); };
    d.ctx_get_current = (decltype(d.ctx_get_current))sym("cuCtxGetCurrent");
    d.module_load_data = (decltype(d.module_load_data))sym("cuModuleLoadData");
    d.module_get_function =
        (decltype(d.module_get_function))sym("cuModuleGetFunction");
    d.launch_kernel = (decltype(d.launch_kernel))sym("cuLaunchKernel");
    d.get_error_name = (decltype(d.get_error_name))sym("cuGetErrorName");

    d.loaded = d.ctx_get_current && d.module_load_data &&
               d.module_get_function && d.launch_kernel;
  });
  return d;
}

// Driver statuses are CUresult enum values, and the bare numbers say nothing.
// CUDA_ERROR_NO_BINARY_FOR_GPU in particular is what a user sees if the arch
// list in `configure` ever falls behind their hardware.
Error driver_error(int status, const std::string &what) {
  const char *name = nullptr;
  if (driver().get_error_name) driver().get_error_name(status, &name);
  std::string detail =
      name ? std::string(name) : "CUDA error " + std::to_string(status);
  return Error::Internal(what + " failed: " + detail);
}

// Guards both caches below.
std::mutex &cache_mutex() {
  static std::mutex m;
  return m;
}

// Modules are per-context: a handle loaded under one context is meaningless
// under another, so a client spreading work over several GPUs needs one set
// each. Caller must hold cache_mutex().
Error modules_for_context(void *ctx, const std::vector<void *> **modules_out) {
  static std::map<void *, std::vector<void *>> modules;
  auto it = modules.find(ctx);
  if (it != modules.end()) {
    *modules_out = &it->second;
    return Error::Success();
  }

  std::vector<void *> loaded;
  loaded.reserve(kCudaFatbinCount);
  for (unsigned int i = 0; i < kCudaFatbinCount; ++i) {
    void *module = nullptr;
    // cuModuleLoadData takes a cubin, PTX, or fatbin image; handing it the
    // fatbin lets the driver select the architecture (and JIT the PTX, for a
    // device newer than every cubin we compiled).
    int status = driver().module_load_data(&module, kCudaFatbins[i].data);
    if (status != kCudaSuccess) {
      return driver_error(status, std::string("cuModuleLoadData for ") +
                                      kCudaFatbins[i].source);
    }
    loaded.push_back(module);
  }

  auto inserted = modules.emplace(ctx, std::move(loaded));
  *modules_out = &inserted.first->second;
  return Error::Success();
}

}  // namespace

Error cuda_kernel(const char *name, void **kernel_out) {
  if (!cuda_kernels_available()) {
    return Error::Internal(
        std::string("pjrt was built without CUDA kernels, so '") + name +
        "' is unavailable. Rebuild pjrt with nvcc on the PATH, or install the "
        "cuda12.8 R package, which ships one.");
  }
  Driver &d = driver();
  if (!d.loaded) {
    return Error::Internal("libcuda.so.1 could not be loaded");
  }

  // XLA activates the device context before invoking an FFI handler, so the
  // calling thread already has the right one; we only need its identity as a
  // cache key.
  void *ctx = nullptr;
  int status = d.ctx_get_current(&ctx);
  if (status != kCudaSuccess) return driver_error(status, "cuCtxGetCurrent");
  if (ctx == nullptr) {
    return Error::Internal("no CUDA context is current on this thread");
  }

  std::lock_guard<std::mutex> lock(cache_mutex());

  static std::map<std::pair<void *, std::string>, void *> functions;
  auto key = std::make_pair(ctx, std::string(name));
  auto cached = functions.find(key);
  if (cached != functions.end()) {
    *kernel_out = cached->second;
    return Error::Success();
  }

  const std::vector<void *> *modules = nullptr;
  PJRT_RETURN_IF_ERROR(modules_for_context(ctx, &modules));

  // Each .cu is its own module, so finding a kernel means asking each in turn.
  // Only "not found" is worth stepping over; anything else is a real fault.
  for (void *module : *modules) {
    void *kernel = nullptr;
    status = d.module_get_function(&kernel, module, name);
    if (status == kCudaSuccess) {
      functions.emplace(std::move(key), kernel);
      *kernel_out = kernel;
      return Error::Success();
    }
    if (status != kCudaErrorNotFound) {
      return driver_error(status,
                          std::string("cuModuleGetFunction for ") + name);
    }
  }

  std::string sources;
  for (unsigned int i = 0; i < kCudaFatbinCount; ++i) {
    if (i > 0) sources += ", ";
    sources += kCudaFatbins[i].source;
  }
  return Error::Internal(std::string("no CUDA kernel named '") + name +
                         "' in any of: " + sources);
}

Error cuda_launch(void *kernel, unsigned int grid_dim, unsigned int block_dim,
                  void *stream, void **args) {
  int status = driver().launch_kernel(kernel, grid_dim, 1, 1, block_dim, 1, 1,
                                      /*shared_mem=*/0, stream, args, nullptr);
  if (status != kCudaSuccess) return driver_error(status, "cuLaunchKernel");
  return Error::Success();
}

}  // namespace rpjrt

#endif  // _WIN32

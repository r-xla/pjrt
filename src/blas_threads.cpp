// How many threads the host LAPACK handlers may call LAPACK from at once.
#include <cstdint>
#include <limits>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "ffi_lapack.h"

namespace rpjrt {

namespace {

using IntGetter = int (*)();
using Int64Getter = std::int64_t (*)();

// What a BLAS that may not be called from several threads at once reports.
struct BlasThreads {
  // The number of threads it runs a call on.
  IntGetter int_threads = nullptr;
  Int64Getter int64_threads = nullptr;
  // OpenBLAS's threading model: 0 for a serial build, which is safe to call
  // from several threads at once only if it was built with USE_LOCKING, which
  // it does not report.
  IntGetter openblas_parallel = nullptr;

  bool found() const { return int_threads || int64_threads; }

  bool concurrent_calls_ok() const {
    if (openblas_parallel && openblas_parallel() == 0) return false;
    std::int64_t n = 1;
    if (int_threads) n = int_threads();
    if (int64_threads) n = int64_threads();
    return n <= 1;
  }
};

#ifndef _WIN32
// Look for a BLAS that runs threads of its own in the library that provides
// `routine`, and in the libraries it loads.
void find_blas_threads(void *routine, BlasThreads &out) {
  Dl_info info;
  if (!dladdr(routine, &info) || !info.dli_fname) return;
  void *lib = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
  if (!lib) return;
  for (const char *name : {"openblas_get_num_threads", "MKL_Get_Max_Threads",
                           "flexiblas_get_num_threads"}) {
    if (void *f = dlsym(lib, name)) {
      out.int_threads = reinterpret_cast<IntGetter>(f);
      break;
    }
  }
  if (void *f = dlsym(lib, "openblas_get_parallel")) {
    out.openblas_parallel = reinterpret_cast<IntGetter>(f);
  }
  if (!out.found()) {
    if (void *f = dlsym(lib, "bli_thread_get_num_threads")) {
      out.int64_threads = reinterpret_cast<Int64Getter>(f);
    }
  }
  // The getters stay valid: pjrt itself links the library.
  dlclose(lib);
}
#endif

BlasThreads blas_threads() {
  BlasThreads threads;
#ifndef _WIN32
  // The BLAS that pjrt calls, then the one its LAPACK calls, which LAPACK's
  // own library may provide.
  find_blas_threads(reinterpret_cast<void *>(&dtrsm_), threads);
  if (!threads.found()) {
    find_blas_threads(reinterpret_cast<void *>(&dgetrf_), threads);
  }
#endif
  return threads;
}

}  // namespace

std::int64_t lapack_max_threads() {
  // Windows links Rtools' static reference BLAS, which runs no threads.
  static const BlasThreads threads = blas_threads();
  return threads.concurrent_calls_ok()
             ? std::numeric_limits<std::int64_t>::max()
             : 1;
}

}  // namespace rpjrt

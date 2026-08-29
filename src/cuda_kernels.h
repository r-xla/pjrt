// Access to the CUDA kernels pjrt ships in `src/cuda/`.
//
// Those .cu files are compiled ahead of time by `configure` into fatbins --
// one per source file, each covering every GPU architecture the PJRT CUDA
// plugin supports -- and embedded in the package's shared object as byte
// arrays (see the generated `src/cuda_fatbin.cpp`). At run time a fatbin is
// handed to the CUDA driver, which picks the cubin matching the device, or
// JITs the PTX for a device newer than any cubin in it.
//
// Everything here goes through the *driver* API (libcuda.so.1), dlopen'd on
// first use exactly like the cuSOLVER entry points in ffi_cuda.h. That is what
// keeps a CUDA kernel from becoming a link-time dependency: the package still
// builds on a machine with no CUDA at all, and still loads on one with no GPU.
// Compiling the .cu against the CUDA *runtime* API instead, with its `<<<>>>`
// launch syntax, would put libcudart in NEEDED and break both.
//
// To add a kernel: drop a .cu in src/cuda/ with an `extern "C" __global__`
// entry point, and call cuda_kernel() with that symbol name from a handler.
// The build picks the file up on its own.
#pragma once

#include "ffi_common.h"
#include "xla/ffi/api/ffi.h"

namespace rpjrt {

// One precompiled src/cuda/*.cu. `source` is the file it came from and is only
// used to make lookup failures legible.
struct CudaFatbin {
  const char *source;
  const unsigned char *data;
  unsigned int size;
};

// Defined by the generated src/cuda_fatbin.cpp. The count is 0 when pjrt was
// configured without a usable nvcc, in which case cuda_kernel() fails with an
// explanation rather than silently doing nothing.
extern const CudaFatbin kCudaFatbins[];
extern const unsigned int kCudaFatbinCount;

// Whether this build has device code at all. Exposed to R as
// `pjrt_cuda_kernels_available()` so callers can choose a fallback lowering
// instead of emitting a custom call that would fail at run time.
bool cuda_kernels_available();

#ifndef _WIN32

// Resolve `name` -- the `extern "C"` symbol of a `__global__` function in one
// of the src/cuda/ sources -- to a launchable kernel.
//
// Modules are loaded once per CUDA context and cached along with the functions
// resolved out of them, so a repeat call costs a map lookup under a mutex.
xla::ffi::Error cuda_kernel(const char *name, void **kernel_out);

// Launch `kernel` on `stream` with a 1-D grid. `args` is the array of
// pointers-to-arguments cuLaunchKernel expects: one entry per kernel
// parameter, each pointing at the value to pass (so an `int *` parameter needs
// the address of the pointer variable, not the pointer itself).
xla::ffi::Error cuda_launch(void *kernel, unsigned int grid_dim,
                            unsigned int block_dim, void *stream, void **args);

#endif  // _WIN32

}  // namespace rpjrt

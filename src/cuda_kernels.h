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
// To add a kernel: drop a .cu in src/cuda/ plus a header declaring its entry
// point (see cuda/lu_pivots_to_permutation.h), and launch it from a handler
// through a `Kernel` built from that declaration. The build picks the .cu up
// on its own.
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
//
// Prefer `Kernel` below to calling this directly: building that array by hand
// erases the argument types, and nothing downstream can notice a mismatch.
xla::ffi::Error cuda_launch(void *kernel, unsigned int grid_dim,
                            unsigned int block_dim, void *stream, void **args);

// A launchable kernel, typed by its signature.
//
// The device code is compiled separately by nvcc into a fatbin, and an
// `extern "C"` symbol carries no argument types, so nothing in the object file
// can catch a handler that disagrees with the kernel it is launching -- and
// the consequence is not a link error but a `cuLaunchKernel` reading whatever
// happens to sit at those addresses. Instantiating this from the kernel's own
// declaration puts the signature back in the type system on the host side:
//
//   #include "cuda/lu_pivots_to_permutation.h"
//   constexpr Kernel<decltype(pjrt_lu_pivots_to_permutation)> kLuPivots{
//       "pjrt_lu_pivots_to_permutation"};
//   ...
//   return kLuPivots(grid, block, stream, pivots_ptr, perm_ptr, k, n);
//
// `Args` is fixed by the declaration rather than deduced from the call, so an
// argument of the wrong type is a compile error instead of a silent
// conversion. The .cu is held to the same declaration by including the header
// that provides it, which is what makes the two ends agree.
template <typename Signature>
struct Kernel;

template <typename... Args>
struct Kernel<void(Args...)> {
  // The `extern "C"` symbol name, as it appears in the .cu.
  const char *name;

  xla::ffi::Error operator()(unsigned int grid_dim, unsigned int block_dim,
                             void *stream, Args... args) const {
    // The parameters live until operator() returns, and cuLaunchKernel
    // marshals the values before it does, so their addresses are good here.
    void *arg_ptrs[] = {
        const_cast<void *>(static_cast<const void *>(&args))...};
    void *kernel = nullptr;
    PJRT_RETURN_IF_ERROR(cuda_kernel(name, &kernel));
    return cuda_launch(kernel, grid_dim, block_dim, stream, arg_ptrs);
  }
};

#endif  // _WIN32

}  // namespace rpjrt

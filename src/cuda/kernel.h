// Shared by every kernel declaration header in this directory.
//
// Those headers are read twice: by nvcc, compiling the .cu that defines the
// kernel, and by the host compiler, building the FFI handler that launches it.
// Only the first knows what `__global__` means, so the declarations spell it
// through this macro. (Defining `__global__` itself would work, but it is a
// reserved identifier and the host compiler is within its rights to object.)
#pragma once

#ifdef __CUDACC__
#define PJRT_CUDA_KERNEL __global__
#else
#define PJRT_CUDA_KERNEL
#endif

// The one declaration of this kernel. `lu_pivots_to_permutation.cu` includes
// it and defines the function, so nvcc checks the definition against it;
// `../lu_pivots_cuda.cpp` includes it and derives its launch signature from it
// with `decltype`, so the host side is checked against it too. A change to the
// parameter list that is not mirrored on both sides fails to compile rather
// than corrupting device memory at run time -- see cuda_kernels.h.
#pragma once

#include "kernel.h"

extern "C" PJRT_CUDA_KERNEL void pjrt_lu_pivots_to_permutation(
    const int *pivots, int *permutation, int k, int n);

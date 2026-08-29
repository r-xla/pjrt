// Turn LAPACK-style pivots into a permutation vector.
//
// getrf reports its row exchanges as `pivots`: a length-k sequence where step
// i swapped row i with row pivots[i] (1-based, as LAPACK and cuSOLVER both
// number rows). Replaying those swaps on the identity yields the permutation
// vector P with P A = L U. See src/lu_pivots.cpp for the host implementation
// and the buffer contract.
//
// The swap chain is sequential by construction -- swap i acts on the result of
// swap i-1 -- so there is no parallelism to exploit across it. This kernel
// exists to collapse *launches*, not to add threads: the XLA while-loop it
// replaces runs k iterations of dynamic-slice / dynamic-update-slice, each one
// its own launch round-tripping the whole vector through global memory. Here
// the identity fill is spread over the block and the chain runs once, in
// registers, in a single launch. This is the same trade jaxlib makes in
// jaxlib/gpu/lu_pivots_to_permutation.cu.cc.
//
// Launch contract: exactly one block. The __syncthreads() below only orders
// threads within a block, so a multi-block launch would let the swap chain
// start before the identity fill finished.

extern "C" __global__ void pjrt_lu_pivots_to_permutation(
    const int *pivots, int *permutation, int k, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    permutation[i] = i + 1;
  }
  __syncthreads();

  if (threadIdx.x != 0) return;
  for (int i = 0; i < k; ++i) {
    // `pivots` is device memory written by getrf. Out-of-range entries can
    // only come from a failed factorisation (cuSOLVER leaves the tail of ipiv
    // untouched when it bails early); skipping them keeps the output a valid
    // permutation instead of corrupting memory outside the buffer.
    const int j = pivots[i] - 1;
    if (j < 0 || j >= n) continue;
    const int tmp = permutation[i];
    permutation[i] = permutation[j];
    permutation[j] = tmp;
  }
}

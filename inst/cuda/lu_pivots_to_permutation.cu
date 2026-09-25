// Converts LAPACK / cuSOLVER getrf pivots into the permutation they apply.
//
// `pivots` holds, for each of `batch` matrices, `k` 1-based row indices:
// getrf swapped row i with row pivots[i], in order. `perm` receives, for each
// matrix, the 1-based permutation of its `m` rows that those swaps amount to.
//
// The swaps of one matrix are sequential, so each thread handles one matrix
// -- the same approach as jaxlib's LuPivotsToPermutationKernel. Doing this in
// one launch avoids the `while` loop XLA would otherwise run on the host,
// with a kernel launch and a device-to-host copy per iteration.
extern "C" __global__ void lu_pivots_to_permutation(const int *pivots,
                                                    int *perm, int batch,
                                                    int k, int m) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  const int *piv = pivots + static_cast<long long>(b) * k;
  int *p = perm + static_cast<long long>(b) * m;
  for (int i = 0; i < m; ++i) p[i] = i + 1;
  for (int i = 0; i < k && i < m; ++i) {
    int j = piv[i] - 1;
    if (j >= 0 && j < m) {
      int tmp = p[i];
      p[i] = p[j];
      p[j] = tmp;
    }
  }
}

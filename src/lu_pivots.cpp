// Turn LAPACK-style pivots into a permutation vector.
//
// `lu` (src/lu.cpp) returns getrf's raw `ipiv`: a length-k sequence of 1-based
// row swaps, where step i exchanged row i with row pivots[i]. Callers almost
// always want the permutation P instead, and deriving it in the IR costs a
// k-iteration while loop -- cheap on CPU, but on GPU k separate kernel
// launches that dominate the factorisation itself. Exposing it as a custom
// call lets both platforms do it in one shot.
//
// Inputs:
//   pivots       : (k,) int32, 1-based, as returned by `lu`
// Outputs:
//   permutation  : (n,) int32, 1-based, with n >= k
#include <Rcpp.h>

#include <algorithm>
#include <cstdint>

#include "ffi_common.h"

using namespace xla::ffi;

namespace rpjrt {

// Shared by both platforms: n comes from the result buffer rather than the
// input, because the permutation is over the matrix's rows (m) while there are
// only min(m, n) pivots.
static Error check_pivot_sizes(std::int64_t k, std::int64_t n) {
  if (k > n) {
    return Error::InvalidArgument(
        "lu_pivots_to_permutation: got more pivots (" + std::to_string(k) +
        ") than permutation entries (" + std::to_string(n) + ")");
  }
  return Error::Success();
}

static Error do_lu_pivots_to_permutation(
    BufferR1<DataType::S32> pivots, Result<BufferR1<DataType::S32>> perm) {
  const auto k = static_cast<std::int64_t>(pivots.element_count());
  const auto n = static_cast<std::int64_t>(perm->element_count());
  PJRT_RETURN_IF_ERROR(check_pivot_sizes(k, n));

  const std::int32_t *piv = pivots.typed_data();
  std::int32_t *out = perm->typed_data();

  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = static_cast<std::int32_t>(i + 1);
  }
  for (std::int64_t i = 0; i < k; ++i) {
    // Out-of-range entries only occur when the factorisation bailed early and
    // left the tail of ipiv unwritten; skipping them keeps the result a
    // permutation. src/cuda/lu_pivots_to_permutation.cu does the same.
    const std::int64_t j = static_cast<std::int64_t>(piv[i]) - 1;
    if (j < 0 || j >= n) continue;
    std::swap(out[i], out[j]);
  }

  return Error::Success();
}

XLA_FFI_DEFINE_HANDLER(lu_pivots_to_permutation_handler,
                       do_lu_pivots_to_permutation,
                       Ffi::Bind()
                           .Arg<BufferR1<DataType::S32>>()   // pivots
                           .Ret<BufferR1<DataType::S32>>());  // permutation

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_lu_pivots_to_permutation_handler() {
  return R_MakeExternalPtr((void *)rpjrt::lu_pivots_to_permutation_handler,
                           R_NilValue, R_NilValue);
}

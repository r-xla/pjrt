// LU decomposition with partial pivoting: P A = L U.
//
// LAPACK getrf overwrites A with L (strictly below the diagonal, unit
// diagonal implicit) and U (on and above the diagonal). The k = min(m, n)
// pivot indices are returned in `ipiv`, 1-based row swaps applied during
// elimination.
//
// Outputs:
//   LU      : (m, n), same dtype as input
//   pivots  : (k,)   int32
#include <Rcpp.h>

#include "ffi_common.h"
#include "ffi_lapack.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error lu_impl(AnyBuffer input, Result<AnyBuffer> lu_out,
                     Result<AnyBuffer> piv_out) {
  using S = typename Lapack<T>::S;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));

  std::vector<S> a(static_cast<std::size_t>(m) * n);
  const T *in = static_cast<const T *>(input.untyped_data());
  for (std::size_t i = 0; i < a.size(); i++)
    a[i] = static_cast<S>(in[i]);

  // getrf has no workspace argument -- pivoting is in place.
  int *ipiv = static_cast<int *>((*piv_out).untyped_data());
  int info;
  Lapack<T>::getrf(&m, &n, a.data(), &m, ipiv, &info);
  // info > 0 means U(info, info) = 0 (singular). Surface that as an error;
  // the user has no way to act on a "factorised but singular" return.
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "getrf"));

  T *lu_data = static_cast<T *>((*lu_out).untyped_data());
  for (std::size_t i = 0; i < a.size(); i++)
    lu_data[i] = static_cast<T>(a[i]);

  return Error::Success();
}

static Error do_lu(AnyBuffer input, Result<AnyBuffer> lu_out,
                   Result<AnyBuffer> piv_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), lu_impl, input, lu_out, piv_out);
}

XLA_FFI_DEFINE_HANDLER(lu_handler, do_lu,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()    // input matrix
                           .Ret<AnyBuffer>()    // LU (same dtype)
                           .Ret<AnyBuffer>());  // pivots (int32)

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_lu_handler() {
  return R_MakeExternalPtr((void *)rpjrt::lu_handler, R_NilValue, R_NilValue);
}

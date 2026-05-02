// Symmetric eigendecomposition via LAPACK syevd.
//
// Input is interpreted as symmetric using only its lower triangle (we pass
// uplo = 'L'). The factorisation produces:
//   V : (n, n)  eigenvectors as columns (orthonormal)
//   W : (n,)    eigenvalues in ascending order
// such that A = V diag(W) V^T (using the lower triangle of A).
//
// jobz = 'V' (compute eigenvectors). For values-only the user can drop V.
//
// syevd needs *two* workspaces (real `work` of size lwork, integer `iwork` of
// size liwork) -- both queried with the lwork = -1 idiom.
#include <Rcpp.h>

#include "ffi_common.h"
#include "ffi_lapack.h"

#include <cstddef>
#include <vector>

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error eigh_impl(AnyBuffer input, Result<AnyBuffer> v_out,
                       Result<AnyBuffer> w_out) {
  using S = typename Lapack<T>::S;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  if (m != n)
    return Error::InvalidArgument("eigh requires a square matrix");

  // syevd overwrites A with eigenvectors. Copy input -> V output buffer and
  // factorise in place there.
  std::vector<S> a(static_cast<std::size_t>(n) * n);
  const T *in = static_cast<const T *>(input.untyped_data());
  for (std::size_t i = 0; i < a.size(); i++)
    a[i] = static_cast<S>(in[i]);

  std::vector<S> w(n);

  const char jobz = 'V';
  const char uplo = 'L';
  int info;

  int lwork = -1, liwork = -1;
  S work_size;
  int iwork_size;
  Lapack<T>::syevd(&jobz, &uplo, &n, a.data(), &n, w.data(), &work_size, &lwork,
                   &iwork_size, &liwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "syevd workspace query"));

  lwork = static_cast<int>(work_size);
  liwork = iwork_size;
  std::vector<S> work(lwork);
  std::vector<int> iwork(liwork);
  Lapack<T>::syevd(&jobz, &uplo, &n, a.data(), &n, w.data(), work.data(),
                   &lwork, iwork.data(), &liwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "syevd"));

  T *v_data = static_cast<T *>((*v_out).untyped_data());
  for (std::size_t i = 0; i < a.size(); i++)
    v_data[i] = static_cast<T>(a[i]);

  T *w_data = static_cast<T *>((*w_out).untyped_data());
  for (int i = 0; i < n; i++)
    w_data[i] = static_cast<T>(w[i]);

  return Error::Success();
}

static Error do_eigh(AnyBuffer input, Result<AnyBuffer> v_out,
                     Result<AnyBuffer> w_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), eigh_impl, input, v_out, w_out);
}

XLA_FFI_DEFINE_HANDLER(eigh_handler, do_eigh,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()    // symmetric matrix
                           .Ret<AnyBuffer>()    // eigenvectors (n, n)
                           .Ret<AnyBuffer>());  // eigenvalues (n,)

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_eigh_handler() {
  return R_MakeExternalPtr((void *)rpjrt::eigh_handler, R_NilValue, R_NilValue);
}

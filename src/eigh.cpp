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

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ffi_common.h"
#include "ffi_lapack.h"

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
  if (m != n) return Error::InvalidArgument("eigh requires a square matrix");

  const T *in = static_cast<const T *>(input.untyped_data());
  T *v_data = static_cast<T *>((*v_out).untyped_data());
  T *w_data = static_cast<T *>((*w_out).untyped_data());
  std::size_t total = static_cast<std::size_t>(n) * n;

  // syevd overwrites its A argument with the eigenvectors -- factor in place
  // in the V output buffer. The pointer-equality guard inside promote_inplace
  // covers the input-output aliasing case (see eigh_cuda.cpp:43-56 for the
  // rationale -- mirrors jaxlib's `CopyIfDiffBuffer`).
  std::vector<S> a_storage, w_storage;
  S *a = promote_inplace<T>(a_storage, v_data, total, in);
  S *w = promote_output<T>(w_storage, w_data, n);

  const char jobz = 'V';
  const char uplo = 'L';
  int info;

  int lwork = -1, liwork = -1;
  S work_size;
  int iwork_size;
  Lapack<T>::syevd(&jobz, &uplo, &n, a, &n, w, &work_size, &lwork, &iwork_size,
                   &liwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "syevd workspace query"));

  lwork = static_cast<int>(work_size);
  liwork = iwork_size;
  std::vector<S> work(lwork);
  std::vector<int> iwork(liwork);
  Lapack<T>::syevd(&jobz, &uplo, &n, a, &n, w, work.data(), &lwork,
                   iwork.data(), &liwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "syevd"));

  demote_output<T>(a_storage, v_data, total);
  demote_output<T>(w_storage, w_data, n);

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

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_eigh_handler() {
  return R_MakeExternalPtr((void *)rpjrt::eigh_handler, R_NilValue, R_NilValue);
}
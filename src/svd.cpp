// Singular value decomposition via LAPACK gesdd (divide-and-conquer).
//
// Reduced ("economy") SVD: jobz = 'S'. For A of shape (m, n) with k = min(m,
// n):
//   U  : (m, k)
//   S  : (k,)     (always non-negative, real)
//   Vt : (k, n)
// such that A = U diag(S) Vt.
//
// gesdd needs an integer workspace `iwork` of size 8*k in addition to the
// real workspace queried via lwork = -1.
#include <Rcpp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ffi_common.h"
#include "ffi_lapack.h"

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error svd_impl(AnyBuffer input, Result<AnyBuffer> u_out,
                      Result<AnyBuffer> s_out, Result<AnyBuffer> vt_out) {
  using S = typename Lapack<T>::S;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  int k = std::min(m, n);

  const T *in = static_cast<const T *>(input.untyped_data());
  T *u_data = static_cast<T *>((*u_out).untyped_data());
  T *s_data = static_cast<T *>((*s_out).untyped_data());
  T *vt_data = static_cast<T *>((*vt_out).untyped_data());
  std::size_t total_a = static_cast<std::size_t>(m) * n;
  std::size_t total_u = static_cast<std::size_t>(m) * k;
  std::size_t total_vt = static_cast<std::size_t>(k) * n;

  // gesdd overwrites A (m x n). We always need a separate working buffer
  // because A's shape doesn't match any output, so promote_workspace
  // unconditionally allocates and copies input -> a_storage. The U / S /
  // Vt outputs go through promote_output / demote_output.
  std::vector<S> a_storage, sv_storage, u_storage, vt_storage;
  S *a = promote_workspace<T>(a_storage, total_a, in);
  S *sv = promote_output<T>(sv_storage, s_data, k);
  S *u = promote_output<T>(u_storage, u_data, total_u);
  S *vt = promote_output<T>(vt_storage, vt_data, total_vt);

  const char jobz = 'S';
  int ldu = m;
  int ldvt = k;
  int info;

  int lwork = -1;
  S work_size;
  std::vector<int> iwork(8 * k);
  Lapack<T>::gesdd(&jobz, &m, &n, a, &m, sv, u, &ldu, vt, &ldvt, &work_size,
                   &lwork, iwork.data(), &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "gesdd workspace query"));

  lwork = static_cast<int>(work_size);
  std::vector<S> work(lwork);
  Lapack<T>::gesdd(&jobz, &m, &n, a, &m, sv, u, &ldu, vt, &ldvt, work.data(),
                   &lwork, iwork.data(), &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "gesdd"));

  demote_output<T>(u_storage, u_data, total_u);
  demote_output<T>(sv_storage, s_data, k);
  demote_output<T>(vt_storage, vt_data, total_vt);

  return Error::Success();
}

static Error do_svd(AnyBuffer input, Result<AnyBuffer> u_out,
                    Result<AnyBuffer> s_out, Result<AnyBuffer> vt_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), svd_impl, input, u_out, s_out,
                      vt_out);
}

XLA_FFI_DEFINE_HANDLER(svd_handler, do_svd,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()    // input matrix
                           .Ret<AnyBuffer>()    // U  (m, k)
                           .Ret<AnyBuffer>()    // S  (k,)
                           .Ret<AnyBuffer>());  // Vt (k, n)

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_svd_handler() {
  return R_MakeExternalPtr((void *)rpjrt::svd_handler, R_NilValue, R_NilValue);
}

// Singular value decomposition via LAPACK gesdd (divide-and-conquer).
//
// Reduced ("economy") SVD: jobz = 'S'. For A of shape (m, n) with k = min(m, n):
//   U  : (m, k)
//   S  : (k,)     (always non-negative, real)
//   Vt : (k, n)
// such that A = U diag(S) Vt.
//
// gesdd needs an integer workspace `iwork` of size 8*k in addition to the
// real workspace queried via lwork = -1.
#include <Rcpp.h>

#include "ffi_common.h"
#include "ffi_lapack.h"

#include <algorithm>
#include <cstddef>
#include <vector>

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

  std::vector<S> a(static_cast<std::size_t>(m) * n);
  const T *in = static_cast<const T *>(input.untyped_data());
  for (std::size_t i = 0; i < a.size(); i++)
    a[i] = static_cast<S>(in[i]);

  std::vector<S> sv(k);
  std::vector<S> u(static_cast<std::size_t>(m) * k);
  std::vector<S> vt(static_cast<std::size_t>(k) * n);

  const char jobz = 'S';
  int ldu = m;
  int ldvt = k;
  int info;

  int lwork = -1;
  S work_size;
  std::vector<int> iwork(8 * k);
  Lapack<T>::gesdd(&jobz, &m, &n, a.data(), &m, sv.data(), u.data(), &ldu,
                   vt.data(), &ldvt, &work_size, &lwork, iwork.data(), &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "gesdd workspace query"));

  lwork = static_cast<int>(work_size);
  std::vector<S> work(lwork);
  Lapack<T>::gesdd(&jobz, &m, &n, a.data(), &m, sv.data(), u.data(), &ldu,
                   vt.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "gesdd"));

  T *u_data = static_cast<T *>((*u_out).untyped_data());
  T *s_data = static_cast<T *>((*s_out).untyped_data());
  T *vt_data = static_cast<T *>((*vt_out).untyped_data());
  for (std::size_t i = 0; i < u.size(); i++)
    u_data[i] = static_cast<T>(u[i]);
  for (int i = 0; i < k; i++)
    s_data[i] = static_cast<T>(sv[i]);
  for (std::size_t i = 0; i < vt.size(); i++)
    vt_data[i] = static_cast<T>(vt[i]);

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

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_svd_handler() {
  return R_MakeExternalPtr((void *)rpjrt::svd_handler, R_NilValue, R_NilValue);
}

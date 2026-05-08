// QR decomposition primitives.
//
//   geqrf : A (m, n) -> packed (m, n), tau (k,)
//     Householder QR factorisation. The output `packed` holds R in the
//     upper triangle (including diagonal) and the Householder reflectors
//     encoding Q in the strict lower triangle. R as a separate `(k, n)`
//     tensor is recovered from `packed` via stablehlo `triu` (= iota +
//     compare + select).
//
//   orgqr : packed (m, n), tau (k,) -> Q (m, k)
//     Materialises the orthogonal factor Q from the reflectors+tau
//     produced by geqrf. cuSOLVER / LAPACK both run this in place on an
//     m x k matrix; we copy the first m*k entries of `packed` into the Q
//     output buffer and factorise there.
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

// ---- geqrf -----------------------------------------------------------------

template <typename T>
static Error geqrf_impl(AnyBuffer input, Result<AnyBuffer> packed_out,
                        Result<AnyBuffer> tau_out) {
  using S = typename Lapack<T>::S;

  auto dims = input.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  int k = std::min(m, n);

  const T *in = static_cast<const T *>(input.untyped_data());
  T *packed_data = static_cast<T *>((*packed_out).untyped_data());
  T *tau_data = static_cast<T *>((*tau_out).untyped_data());
  std::size_t total = static_cast<std::size_t>(m) * n;

  std::vector<S> a_storage, tau_storage;
  S *a = promote_inplace<T>(a_storage, packed_data, total, in);
  S *tau = promote_output<T>(tau_storage, tau_data, k);

  int lwork = -1;
  S work_size;
  int info;
  Lapack<T>::geqrf(&m, &n, a, &m, tau, &work_size, &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "geqrf workspace query"));

  lwork = static_cast<int>(work_size);
  std::vector<S> work(lwork);
  Lapack<T>::geqrf(&m, &n, a, &m, tau, work.data(), &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "geqrf"));

  demote_output<T>(a_storage, packed_data, total);
  demote_output<T>(tau_storage, tau_data, k);

  return Error::Success();
}

static Error do_geqrf(AnyBuffer input, Result<AnyBuffer> packed_out,
                      Result<AnyBuffer> tau_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), geqrf_impl, input, packed_out,
                      tau_out);
}

XLA_FFI_DEFINE_HANDLER(geqrf_handler, do_geqrf,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()    // input matrix (m, n)
                           .Ret<AnyBuffer>()    // packed (m, n)
                           .Ret<AnyBuffer>());  // tau (k,)

// ---- orgqr -----------------------------------------------------------------

template <typename T>
static Error orgqr_impl(AnyBuffer packed, AnyBuffer tau_in,
                        Result<AnyBuffer> q_out) {
  using S = typename Lapack<T>::S;

  auto dims = packed.dimensions();
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[0], "rows", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(dims[1], "cols", n));
  int k = std::min(m, n);

  const T *p_in = static_cast<const T *>(packed.untyped_data());
  const T *tau_in_ptr = static_cast<const T *>(tau_in.untyped_data());
  T *q_data = static_cast<T *>((*q_out).untyped_data());
  std::size_t q_total = static_cast<std::size_t>(m) * k;

  // orgqr operates on an m x k matrix in place. Copy the first m*k entries
  // of packed (= the first k columns, since the layout is column-major)
  // into Q and factor there. tau is read-only.
  std::vector<S> a_storage, tau_storage;
  S *a = promote_inplace<T>(a_storage, q_data, q_total, p_in);
  const S *tau = promote_input<T>(tau_storage, k, tau_in_ptr);

  int lwork = -1;
  S work_size;
  int info;
  Lapack<T>::orgqr(&m, &k, &k, a, &m, tau, &work_size, &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "orgqr workspace query"));

  lwork = static_cast<int>(work_size);
  std::vector<S> work(lwork);
  Lapack<T>::orgqr(&m, &k, &k, a, &m, tau, work.data(), &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "orgqr"));

  demote_output<T>(a_storage, q_data, q_total);

  return Error::Success();
}

static Error do_orgqr(AnyBuffer packed, AnyBuffer tau_in,
                      Result<AnyBuffer> q_out) {
  PJRT_DISPATCH_FLOAT(packed.element_type(), orgqr_impl, packed, tau_in, q_out);
}

XLA_FFI_DEFINE_HANDLER(orgqr_handler, do_orgqr,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()    // packed reflectors (m, n)
                           .Arg<AnyBuffer>()    // tau (k,)
                           .Ret<AnyBuffer>());  // Q (m, k)

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_geqrf_handler() {
  return R_MakeExternalPtr((void *)rpjrt::geqrf_handler, R_NilValue,
                           R_NilValue);
}

// [[Rcpp::export]]
SEXP get_orgqr_handler() {
  return R_MakeExternalPtr((void *)rpjrt::orgqr_handler, R_NilValue,
                           R_NilValue);
}

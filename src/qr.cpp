// QR decomposition: A (m x n) -> Q (m x k), R (k x n), where k = min(m, n).
//
// All buffers use column-major layout (specified by operand_layouts /
// result_layouts on the custom_call). XLA handles the row-major <->
// column-major conversion transparently -- the handler works directly in
// LAPACK-native layout.
//
// The two-phase pattern is geqrf (compute Householder reflectors + R) then
// orgqr (materialise Q from the reflectors). LAPACK's `lwork = -1` workspace
// query is run before each call -- the optimal size depends on
// implementation-specific blocking parameters that we can't compute ahead
// of time.
#include <Rcpp.h>

#include "ffi_common.h"
#include "ffi_lapack.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error qr_impl(AnyBuffer input, Result<AnyBuffer> q_out,
                     Result<AnyBuffer> r_out) {
  // S is the LAPACK storage type: float/double on macOS+Linux, double on
  // Windows for both (R's bundled Rlapack.dll has no s* routines, so f32
  // input is promoted to double across the call).
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

  std::vector<S> tau(k);
  int lwork = -1;
  S work_size;
  int info;
  Lapack<T>::geqrf(&m, &n, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "geqrf workspace query"));

  lwork = static_cast<int>(work_size);
  std::vector<S> work(lwork);
  Lapack<T>::geqrf(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork,
                   &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "geqrf"));

  T *r_data = static_cast<T *>((*r_out).untyped_data());
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      r_data[j * k + i] =
          (i <= j) ? static_cast<T>(a[j * m + i]) : static_cast<T>(0);
    }
  }

  lwork = -1;
  Lapack<T>::orgqr(&m, &k, &k, a.data(), &m, tau.data(), &work_size, &lwork,
                   &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "orgqr workspace query"));

  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  Lapack<T>::orgqr(&m, &k, &k, a.data(), &m, tau.data(), work.data(), &lwork,
                   &info);
  PJRT_RETURN_IF_ERROR(lapack_check_info(info, "orgqr"));

  T *q_data = static_cast<T *>((*q_out).untyped_data());
  for (std::size_t i = 0; i < static_cast<std::size_t>(m) * k; i++)
    q_data[i] = static_cast<T>(a[i]);

  return Error::Success();
}

static Error do_qr(AnyBuffer input, Result<AnyBuffer> q_out,
                   Result<AnyBuffer> r_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), qr_impl, input, q_out, r_out);
}

XLA_FFI_DEFINE_HANDLER(qr_handler, do_qr,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()   // input matrix (column-major)
                           .Ret<AnyBuffer>()   // Q output (column-major)
                           .Ret<AnyBuffer>()); // R output (column-major)

} // namespace rpjrt

// [[Rcpp::export]]
SEXP get_qr_handler() {
  return R_MakeExternalPtr((void *)rpjrt::qr_handler, R_NilValue, R_NilValue);
}

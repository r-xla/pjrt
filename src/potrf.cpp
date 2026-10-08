// Cholesky factorisation via LAPACK potrf, over a batch of matrices.
//
// Input: A, shape [..., n, n], f32 or f64, every matrix stored contiguously
// in column-major order (the caller declares the layout). Only the triangle
// selected by the `lower` attribute is read.
//
// Outputs:
//   factor : [..., n, n], same dtype. The `lower` triangle holds L (A = L L^T)
//            or U (A = U^T U); the other triangle keeps the input's values.
//   info   : [...] int32, potrf's info per matrix: 0 on success, k > 0 if the
//            leading minor of order k is not positive definite (the factor is
//            then incomplete).
//
// A matrix that is not positive definite is not an error: the caller reads
// `info` and decides (anvl fills that factor with NaN). Returning info as an
// output rather than failing is what lets the CUDA handler stay asynchronous
// (devInfo lives on the device), and keeps both platforms on one contract.
#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ffi_common.h"
#include "ffi_lapack.h"

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error potrf_impl(bool lower, AnyBuffer input, Result<AnyBuffer> out,
                        Result<AnyBuffer> info_out) {
  using S = typename Lapack<T>::S;

  int n;
  std::int64_t batch;
  PJRT_RETURN_IF_ERROR(
      batched_square_dims(input.dimensions(), "potrf", n, batch));

  const T *in = static_cast<const T *>(input.untyped_data());
  T *out_data = static_cast<T *>((*out).untyped_data());
  int *info = static_cast<int *>((*info_out).untyped_data());
  std::size_t mat = static_cast<std::size_t>(n) * n;
  std::size_t total = mat * static_cast<std::size_t>(batch);

  // potrf factors in place: seed the output with the input and factor there.
  std::vector<S> a_storage;
  S *a = promote_inplace<T>(a_storage, out_data, total, in);

  const char uplo = lower ? 'L' : 'U';
  const int lda = n > 0 ? n : 1;
  for (std::int64_t b = 0; b < batch; b++) {
    int info_b;
    Lapack<T>::potrf(&uplo, &n, a + b * mat, &lda, &info_b);
    // info < 0 is an illegal argument, a bug on our side rather than data.
    if (info_b < 0) return lapack_check_info(info_b, "potrf");
    info[b] = info_b;
  }

  demote_output<T>(a_storage, out_data, total);

  return Error::Success();
}

static Error do_potrf(bool lower, AnyBuffer input, Result<AnyBuffer> out,
                      Result<AnyBuffer> info_out) {
  PJRT_DISPATCH_FLOAT(input.element_type(), potrf_impl, lower, input, out,
                      info_out);
}

XLA_FFI_DEFINE_HANDLER(potrf_handler, do_potrf,
                       Ffi::Bind()
                           .Attr<bool>("lower")  // which triangle
                           .Arg<AnyBuffer>()     // matrices [..., n, n]
                           .Ret<AnyBuffer>()     // factor [..., n, n]
                           .Ret<AnyBuffer>());   // info [...] (int32)

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_potrf_handler() {
  return R_MakeExternalPtr((void *)rpjrt::potrf_handler, R_NilValue,
                           R_NilValue);
}

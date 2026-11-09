// Triangular solve via BLAS trsm, over a batch of matrices: a stand-in for
// `stablehlo.triangular_solve`, with its attributes and semantics. Host only:
// on CUDA, XLA itself lowers that op to cuBLAS trsm, while on the CPU it
// expands it into far slower HLO.
//
// Unlike the factorisations, trsm takes its matrices in XLA's default
// row-major layout, so that standing in for the op costs no transposes.
//
// Inputs, every matrix stored contiguously in row-major order:
//   a : [..., k, k], f32 or f64, triangular. Only the triangle selected by
//       `lower` is read, and with `unit_diagonal` not its diagonal either.
//   b : [..., m, n], same dtype and batch axes, with k = m when `left_side`
//       and k = n otherwise.
//
// Output:
//   x : [..., m, n], same dtype, row-major, solving op(a) x = b when
//       `left_side` and x op(a) = b otherwise, where op(a) is t(a) when
//       `transpose_a` and a otherwise. It may alias b, but not a.
//
// A singular `a` is not detected, as with trsm itself: x then holds Inf / NaN.
#include <Rcpp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ffi_common.h"
#include "ffi_lapack.h"

using namespace xla::ffi;

namespace rpjrt {

template <typename T>
static Error trsm_impl(bool left_side, bool lower, bool transpose_a,
                       bool unit_diagonal, AnyBuffer a_in, AnyBuffer b_in,
                       Result<AnyBuffer> out) {
  using S = typename Lapack<T>::S;

  int k;
  std::int64_t batch;
  PJRT_RETURN_IF_ERROR(
      batched_square_dims(a_in.dimensions(), "trsm", k, batch));

  auto a_dims = a_in.dimensions();
  auto b_dims = b_in.dimensions();
  auto x_dims = (*out).dimensions();
  std::size_t r = b_dims.size();
  if (r != a_dims.size() ||
      !std::equal(a_dims.begin(), a_dims.end() - 2, b_dims.begin()) ||
      b_dims[left_side ? r - 2 : r - 1] != k) {
    return Error::InvalidArgument(
        "trsm requires b [..., m, n] with the batch axes of a [..., k, k], "
        "and k = m (left side) or k = n (right side)");
  }
  if (!std::equal(x_dims.begin(), x_dims.end(), b_dims.begin(), b_dims.end())) {
    return Error::InvalidArgument("trsm requires x of the shape of b");
  }
  int m, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(b_dims[r - 2], "matrix", m));
  PJRT_RETURN_IF_ERROR(dim_to_int(b_dims[r - 1], "matrix", n));

  std::size_t a_mat = static_cast<std::size_t>(k) * k;
  std::size_t b_mat = static_cast<std::size_t>(m) * n;
  std::size_t b_total = b_mat * static_cast<std::size_t>(batch);
  T *out_data = static_cast<T *>((*out).untyped_data());

  std::vector<S> a_storage;
  const S *a = promote_input<T>(a_storage, a_mat * batch,
                                static_cast<const T *>(a_in.untyped_data()));
  // trsm solves in place: seed the output with b and solve there.
  std::vector<S> x_storage;
  S *x = promote_inplace<T>(x_storage, out_data, b_total,
                            static_cast<const T *>(b_in.untyped_data()));

  // Column-major BLAS reads a row-major matrix as its transpose, so it solves
  // the transposed system, t(x) op(t(a)) = t(b) when `left_side` and
  // op(t(a)) t(x) = t(b) otherwise: from the other side, against the other
  // triangle of t(a), for the n x m matrix t(x).
  const char side = left_side ? 'R' : 'L';
  const char uplo = lower ? 'U' : 'L';
  const char transa = transpose_a ? 'T' : 'N';
  const char diag = unit_diagonal ? 'U' : 'N';
  const S alpha = 1;
  const int lda = k > 0 ? k : 1;
  const int ldb = n > 0 ? n : 1;
  for (std::int64_t i = 0; i < batch; i++) {
    Lapack<T>::trsm(&side, &uplo, &transa, &diag, &n, &m, &alpha, a + i * a_mat,
                    &lda, x + i * b_mat, &ldb);
  }

  demote_output<T>(x_storage, out_data, b_total);

  return Error::Success();
}

static Error do_trsm(bool left_side, bool lower, bool transpose_a,
                     bool unit_diagonal, AnyBuffer a, AnyBuffer b,
                     Result<AnyBuffer> out) {
  if (b.element_type() != a.element_type() ||
      (*out).element_type() != a.element_type()) {
    return Error::InvalidArgument("trsm requires a, b and x of one dtype");
  }
  PJRT_DISPATCH_FLOAT(a.element_type(), trsm_impl, left_side, lower,
                      transpose_a, unit_diagonal, a, b, out);
}

XLA_FFI_DEFINE_HANDLER(trsm_handler, do_trsm,
                       Ffi::Bind()
                           .Attr<bool>("left_side")      // op(a) x or x op(a)
                           .Attr<bool>("lower")          // which triangle
                           .Attr<bool>("transpose_a")    // op(a) = t(a)
                           .Attr<bool>("unit_diagonal")  // diag(a) taken as 1
                           .Arg<AnyBuffer>()             // a [..., k, k]
                           .Arg<AnyBuffer>()             // b [..., m, n]
                           .Ret<AnyBuffer>());           // x [..., m, n]

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_trsm_handler() {
  return R_MakeExternalPtr((void *)rpjrt::trsm_handler, R_NilValue, R_NilValue);
}

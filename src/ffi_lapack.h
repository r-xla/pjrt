// LAPACK extern declarations and per-precision dispatch traits.
//
// LAPACK naming convention (XYYZZZ):
//   X    = precision: s=single (f32), d=double (f64), c=complex single,
//          z=complex double. We only use s/d.
//   YY   = matrix type: ge=general, sy=symmetric (real), or=orthogonal (real;
//          un=unitary in the complex world).
//   ZZZ  = operation. The routines we use:
//     geqrf -- GEneral QR Factorisation: A = Q*R, returns R in upper triangle
//              of A and the Householder reflectors + tau encoding Q.
//     orgqr -- ORthogonal Generate Q from QR: materialises Q from the
//              reflectors+tau produced by geqrf.
//     getrf -- GEneral TRiangular Factorisation: LU with partial pivoting,
//              A = P*L*U.
//     gesdd -- GEneral SVD via Divide-and-conquer (faster than gesvd at
//              medium/large sizes; jaxlib uses it on CPU too).
//     gesvd -- GEneral SVD via QR iteration (used on the cuSOLVER side).
//     syevd -- SYmmetric EigenValue Decomposition via divide-and-conquer.
//     potrf -- POsitive-definite TRiangular Factorisation: Cholesky,
//              A = L L^T (uplo = 'L') or A = U^T U (uplo = 'U').
//     trsm  -- TRiangular Solve with Multiple right-hand sides: op(A) X = B
//              or X op(A) = B for a triangular A, in place on B. A BLAS
//              (level 3) routine rather than a LAPACK one.
//   The trailing underscore (`dgeqrf_`) is the Fortran ABI symbol mangling.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "ffi_common.h"

extern "C" {
// QR: factorisation + Q materialisation.
void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau,
             double *work, const int *lwork, int *info);
void dorgqr_(const int *m, const int *n, const int *k, double *a,
             const int *lda, const double *tau, double *work, const int *lwork,
             int *info);

// LU: partial-pivoting LU. ipiv is 1-based row indices; info > 0 means a
// pivot was zero (matrix singular).
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv,
             int *info);

// SVD: divide-and-conquer (gesdd) is generally faster than the QR-based
// gesvd for medium/large matrices and is what jaxlib uses on CPU. jobz
// selects which singular vectors to compute: 'A' = full U, V; 'S' = reduced;
// 'O' = overwrite A; 'N' = singular values only.
void dgesdd_(const char *jobz, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt,
             const int *ldvt, double *work, const int *lwork, int *iwork,
             int *info);

// Symmetric/Hermitian eigendecomposition (real). jobz: 'N' eigenvalues only,
// 'V' eigenvectors too. uplo: 'L' / 'U' selects which triangle of A holds
// the input.
void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a,
             const int *lda, double *w, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info);

// Cholesky. Only the `uplo` triangle of A is read and overwritten with the
// factor; the other triangle is left untouched. info > 0 means the leading
// minor of order info is not positive definite (the factorisation stopped).
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda,
             int *info);

// Triangular solve. side 'L' solves op(A) X = alpha B, 'R' X op(A) = alpha B,
// where A is m x m or n x n and B is m x n; X overwrites B. Only the `uplo`
// triangle of A is read, and with diag = 'U' not its diagonal either (taken
// to be 1). transa 'N' / 'T' selects op(A) = A / A^T.
void dtrsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, double *b, const int *ldb);
void sgeqrf_(const int *m, const int *n, float *a, const int *lda, float *tau,
             float *work, const int *lwork, int *info);
void sorgqr_(const int *m, const int *n, const int *k, float *a, const int *lda,
             const float *tau, float *work, const int *lwork, int *info);
void sgetrf_(const int *m, const int *n, float *a, const int *lda, int *ipiv,
             int *info);
void sgesdd_(const char *jobz, const int *m, const int *n, float *a,
             const int *lda, float *s, float *u, const int *ldu, float *vt,
             const int *ldvt, float *work, const int *lwork, int *iwork,
             int *info);
void ssyevd_(const char *jobz, const char *uplo, const int *n, float *a,
             const int *lda, float *w, float *work, const int *lwork,
             int *iwork, const int *liwork, int *info);
void spotrf_(const char *uplo, const int *n, float *a, const int *lda,
             int *info);
void strsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const float *alpha,
            const float *a, const int *lda, float *b, const int *ldb);
}

namespace rpjrt {

inline xla::ffi::Error lapack_check_info(int info, const char *routine) {
  if (info == 0) return xla::ffi::Error::Success();
  return xla::ffi::Error::Internal(
      std::string(routine) + " failed with info = " + std::to_string(info));
}

// Per-precision dispatch trait: Lapack<T>::potrf() etc. call the routine of
// the precision T.
template <typename T>
struct Lapack;

template <>
struct Lapack<double> {
  static void geqrf(const int *m, const int *n, double *a, const int *lda,
                    double *tau, double *work, const int *lwork, int *info) {
    dgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void orgqr(const int *m, const int *n, const int *k, double *a,
                    const int *lda, const double *tau, double *work,
                    const int *lwork, int *info) {
    dorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
  static void getrf(const int *m, const int *n, double *a, const int *lda,
                    int *ipiv, int *info) {
    dgetrf_(m, n, a, lda, ipiv, info);
  }
  static void gesdd(const char *jobz, const int *m, const int *n, double *a,
                    const int *lda, double *s, double *u, const int *ldu,
                    double *vt, const int *ldvt, double *work, const int *lwork,
                    int *iwork, int *info) {
    dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static void syevd(const char *jobz, const char *uplo, const int *n, double *a,
                    const int *lda, double *w, double *work, const int *lwork,
                    int *iwork, const int *liwork, int *info) {
    dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
  }
  static void potrf(const char *uplo, const int *n, double *a, const int *lda,
                    int *info) {
    dpotrf_(uplo, n, a, lda, info);
  }
  static void trsm(const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const double *alpha, const double *a, const int *lda,
                   double *b, const int *ldb) {
    dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
  }
};

template <>
struct Lapack<float> {
  static void geqrf(const int *m, const int *n, float *a, const int *lda,
                    float *tau, float *work, const int *lwork, int *info) {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void orgqr(const int *m, const int *n, const int *k, float *a,
                    const int *lda, const float *tau, float *work,
                    const int *lwork, int *info) {
    sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
  static void getrf(const int *m, const int *n, float *a, const int *lda,
                    int *ipiv, int *info) {
    sgetrf_(m, n, a, lda, ipiv, info);
  }
  static void gesdd(const char *jobz, const int *m, const int *n, float *a,
                    const int *lda, float *s, float *u, const int *ldu,
                    float *vt, const int *ldvt, float *work, const int *lwork,
                    int *iwork, int *info) {
    sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static void syevd(const char *jobz, const char *uplo, const int *n, float *a,
                    const int *lda, float *w, float *work, const int *lwork,
                    int *iwork, const int *liwork, int *info) {
    ssyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
  }
  static void potrf(const char *uplo, const int *n, float *a, const int *lda,
                    int *info) {
    spotrf_(uplo, n, a, lda, info);
  }
  static void trsm(const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n,
                   const float *alpha, const float *a, const int *lda, float *b,
                   const int *ldb) {
    strsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
  }
};

// The most threads a handler may call LAPACK from at once, for
// parallel_for(): one if the BLAS runs threads of its own (a multi-threaded
// OpenBLAS, MKL, BLIS or FlexiBLAS) or is a serial OpenBLAS, and no limit
// otherwise. Calls from XLA's threads at once make the threads of such a BLAS
// compete with one another and with XLA's, so that they run far slower than
// one call after the other, and a serial OpenBLAS is safe to call from several
// threads only if it was built to be. The limit holds within one handler call;
// XLA can still run two handlers at once. The BLAS's thread count is read on
// every call, so a batch is spread over XLA's threads once a multi-threaded
// BLAS is set to one thread.
std::int64_t lapack_max_threads();

// Copy `n` elements from `src` to `dst`, the buffer a routine that works in
// place factors in, unless XLA aliased the two.
template <typename T>
inline void copy_unless_aliased(T *dst, const T *src, std::size_t n) {
  if (dst != src) std::memcpy(dst, src, n * sizeof(T));
}

}  // namespace rpjrt

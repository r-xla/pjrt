// LAPACK extern declarations and per-precision dispatch traits.
//
// On macOS/Linux we link against system LAPACK (Accelerate, OpenBLAS, MKL),
// which provides single- and double-precision routines. On Windows R bundles
// its own Rlapack.dll, which only ships the double-precision variants. To
// keep the per-op kernels precision-agnostic, we expose a Lapack<T> trait
// whose ::S typedef is the precision actually used for the LAPACK call:
//   - non-Windows: Lapack<float>::S = float, Lapack<double>::S = double
//   - Windows:     Lapack<float>::S = double (promote), Lapack<double>::S = double
//
// Each kernel writes:
//
//   using S = typename Lapack<T>::S;
//   std::vector<S> a(...);     // promoted copy of input
//   Lapack<T>::geqrf(...);
//   ... back-cast to T on output ...
//
// and gets the right behaviour on both platforms with no #ifdefs in the
// kernel body. Mirrors the dispatch trait pattern in jaxlib's
// lapack_kernels.cc.
#pragma once

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

#ifndef _WIN32
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
#endif
}

namespace rpjrt {

inline xla::ffi::Error lapack_check_info(int info, const char *routine) {
  if (info == 0)
    return xla::ffi::Error::Success();
  return xla::ffi::Error::Internal(std::string(routine) +
                                   " failed with info = " +
                                   std::to_string(info));
}

// Per-precision dispatch trait. ::S is the storage type that the LAPACK
// routines actually see; on Windows this is always double.
template <typename T> struct Lapack;

template <> struct Lapack<double> {
  using S = double;
  static void geqrf(const int *m, const int *n, S *a, const int *lda, S *tau,
                    S *work, const int *lwork, int *info) {
    dgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void orgqr(const int *m, const int *n, const int *k, S *a,
                    const int *lda, const S *tau, S *work, const int *lwork,
                    int *info) {
    dorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
  static void getrf(const int *m, const int *n, S *a, const int *lda,
                    int *ipiv, int *info) {
    dgetrf_(m, n, a, lda, ipiv, info);
  }
  static void gesdd(const char *jobz, const int *m, const int *n, S *a,
                    const int *lda, S *s, S *u, const int *ldu, S *vt,
                    const int *ldvt, S *work, const int *lwork, int *iwork,
                    int *info) {
    dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static void syevd(const char *jobz, const char *uplo, const int *n, S *a,
                    const int *lda, S *w, S *work, const int *lwork, int *iwork,
                    const int *liwork, int *info) {
    dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
  }
};

#ifndef _WIN32
template <> struct Lapack<float> {
  using S = float;
  static void geqrf(const int *m, const int *n, S *a, const int *lda, S *tau,
                    S *work, const int *lwork, int *info) {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void orgqr(const int *m, const int *n, const int *k, S *a,
                    const int *lda, const S *tau, S *work, const int *lwork,
                    int *info) {
    sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
  static void getrf(const int *m, const int *n, S *a, const int *lda,
                    int *ipiv, int *info) {
    sgetrf_(m, n, a, lda, ipiv, info);
  }
  static void gesdd(const char *jobz, const int *m, const int *n, S *a,
                    const int *lda, S *s, S *u, const int *ldu, S *vt,
                    const int *ldvt, S *work, const int *lwork, int *iwork,
                    int *info) {
    sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static void syevd(const char *jobz, const char *uplo, const int *n, S *a,
                    const int *lda, S *w, S *work, const int *lwork, int *iwork,
                    const int *liwork, int *info) {
    ssyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
  }
};
#else
// Windows: promote f32 -> f64 -> f32 around the LAPACK call.
template <> struct Lapack<float> {
  using S = double;
  static void geqrf(const int *m, const int *n, S *a, const int *lda, S *tau,
                    S *work, const int *lwork, int *info) {
    dgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void orgqr(const int *m, const int *n, const int *k, S *a,
                    const int *lda, const S *tau, S *work, const int *lwork,
                    int *info) {
    dorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
  static void getrf(const int *m, const int *n, S *a, const int *lda,
                    int *ipiv, int *info) {
    dgetrf_(m, n, a, lda, ipiv, info);
  }
  static void gesdd(const char *jobz, const int *m, const int *n, S *a,
                    const int *lda, S *s, S *u, const int *ldu, S *vt,
                    const int *ldvt, S *work, const int *lwork, int *iwork,
                    int *info) {
    dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static void syevd(const char *jobz, const char *uplo, const int *n, S *a,
                    const int *lda, S *w, S *work, const int *lwork, int *iwork,
                    const int *liwork, int *info) {
    dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
  }
};
#endif

} // namespace rpjrt

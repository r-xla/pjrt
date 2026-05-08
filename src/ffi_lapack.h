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
//   The trailing underscore (`dgeqrf_`) is the Fortran ABI symbol mangling.
//
#pragma once

#include "ffi_common.h"

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

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
// On windows we use R's bundeld LAPACK for now, which only has double precision support
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

// ----------------------------------------------------------------------------
// Precision-promotion helpers
//
// LAPACK kernels work in `S = Lapack<T>::S`, which equals `T` except on
// Windows where T = float promotes to S = double (R's bundled LAPACK has
// no s* routines). When S == T LAPACK writes directly into the FFI buffers
// (no heap allocation); when S != T we stage through a `std::vector<S>`
// and cast at the boundary. These helpers factor out the
// `if constexpr (std::is_same_v<S, T>)` branch so each call site is one line.
//
// Pick the helper by the buffer's role in the LAPACK call:
//
//   promote_input     - LAPACK reads it (const S*).
//   promote_output    - LAPACK writes it; no input to seed.
//                       Pair with demote_output after the call.
//   promote_inplace   - LAPACK reads AND writes it; seeded from `src`.
//                       Use when an FFI output buffer can serve as the
//                       in-place factor target (e.g. eigh V <- A,
//                       LU lu <- A, QR packed <- A). Tolerates src == target
//                       for XLA input-output aliasing. Pair with demote_output.
//   promote_workspace - LAPACK reads AND writes it, but no FFI buffer fits
//                       the role -- always allocates. Use when output shapes
//                       don't match input (e.g. SVD's working A). No demote
//                       (the buffer is scratch, not an output).
//   demote_output     - cast staged storage back into the T-typed FFI buffer.
//                       No-op when S == T. Pair with the two writeable
//                       promote_* helpers above.
// ----------------------------------------------------------------------------

template <typename T, typename S = typename Lapack<T>::S>
inline S *promote_inplace(std::vector<S> &storage, T *target, std::size_t n,
                          const T *src) {
  if constexpr (std::is_same_v<S, T>) {
    if (src != target)
      std::memcpy(target, src, n * sizeof(T));
    return target;
  } else {
    storage.resize(n);
    for (std::size_t i = 0; i < n; i++)
      storage[i] = static_cast<S>(src[i]);
    return storage.data();
  }
}

template <typename T, typename S = typename Lapack<T>::S>
inline S *promote_output(std::vector<S> &storage, T *target, std::size_t n) {
  if constexpr (std::is_same_v<S, T>) {
    return target;
  } else {
    storage.resize(n);
    return storage.data();
  }
}

template <typename T, typename S = typename Lapack<T>::S>
inline const S *promote_input(std::vector<S> &storage, std::size_t n,
                              const T *src) {
  if constexpr (std::is_same_v<S, T>) {
    return src;
  } else {
    storage.resize(n);
    for (std::size_t i = 0; i < n; i++)
      storage[i] = static_cast<S>(src[i]);
    return storage.data();
  }
}

template <typename T, typename S = typename Lapack<T>::S>
inline S *promote_workspace(std::vector<S> &storage, std::size_t n,
                            const T *src) {
  storage.resize(n);
  if constexpr (std::is_same_v<S, T>) {
    std::memcpy(storage.data(), src, n * sizeof(T));
  } else {
    for (std::size_t i = 0; i < n; i++)
      storage[i] = static_cast<S>(src[i]);
  }
  return storage.data();
}

template <typename T, typename S = typename Lapack<T>::S>
inline void demote_output(const std::vector<S> &storage, T *target,
                          std::size_t n) {
  if constexpr (!std::is_same_v<S, T>) {
    for (std::size_t i = 0; i < n; i++)
      target[i] = static_cast<T>(storage[i]);
  }
}

} // namespace rpjrt

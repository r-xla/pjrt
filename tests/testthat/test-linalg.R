skip_if_metal("linalg custom_calls are CPU/CUDA only")

describe("qr (geqrf + orgqr)", {
  # Exercises the `geqrf` + `orgqr` LAPACK / cuSOLVER custom calls: QR
  # factorisation of tall, wide, and square matrices in f32 / f64, plus
  # input-buffer donation into the packed-reflectors output.
  #
  # Correctness: geqrf returns a packed matrix (Householder reflectors below
  # the diagonal, R in the upper triangle of the first k rows) plus the tau
  # scalars; we recover R by zeroing the strict lower triangle of those rows
  # and Q by passing the packed output + tau through orgqr. We then check
  # two defining properties: Q R = A (reconstruction) and Q^T Q = I_k (Q has
  # orthonormal columns). Sign / orientation of Q is implementation-defined
  # (e.g. depends on R's diagonal signs), so we don't compare Q or R against
  # any reference factorisation.

  run_geqrf <- function(a, dtype, donate = FALSE) {
    m <- nrow(a)
    n <- ncol(a)
    k <- min(m, n)
    in_spec <- list(dims = c(m, n), dtype = dtype)
    if (donate) {
      in_spec$aliases <- 1L
    }
    run_linalg(
      "geqrf",
      inputs = list(a),
      in_specs = list(in_spec),
      out_specs = list(
        list(dims = c(m, n), dtype = dtype),
        list(dims = k, dtype = dtype)
      )
    )
  }

  run_orgqr <- function(packed, tau, dtype) {
    m <- nrow(packed)
    n <- ncol(packed)
    k <- min(m, n)
    res <- run_linalg(
      "orgqr",
      inputs = list(packed, tau),
      in_specs = list(
        list(dims = c(m, n), dtype = dtype),
        list(dims = k, dtype = dtype)
      ),
      out_specs = list(list(dims = c(m, k), dtype = dtype))
    )
    res[[1L]]
  }

  # Recover Q and R from geqrf's packed output + tau. R is the upper triangle
  # of the first k rows; Q comes from orgqr.
  qr_factors <- function(packed, tau, dtype) {
    k <- min(nrow(packed), ncol(packed))
    R <- packed[seq_len(k), , drop = FALSE]
    R[lower.tri(R)] <- 0
    Q <- run_orgqr(packed, tau, dtype = dtype)
    list(Q = Q, R = R)
  }

  expect_qr_correct <- function(a, dtype) {
    res <- run_geqrf(a, dtype)
    qr <- qr_factors(res[[1L]], res[[2L]], dtype)
    tol <- if (dtype == "f64") 1e-10 else 1e-4
    expect_equal(qr$Q %*% qr$R, a, tolerance = tol)
    expect_equal(crossprod(qr$Q), diag(ncol(qr$Q)), tolerance = tol)
  }

  # ---- Tests ----

  it("factorises tall (m > n) in f64 and f32", {
    withr::local_seed(1)
    a <- matrix(rnorm(20), 5, 4)
    expect_qr_correct(a, "f64")
    expect_qr_correct(a, "f32")
  })

  it("factorises wide (m < n) in f64 and f32", {
    withr::local_seed(2)
    a <- matrix(rnorm(20), 4, 5)
    expect_qr_correct(a, "f64")
    expect_qr_correct(a, "f32")
  })

  it("factorises square in f64 and f32", {
    withr::local_seed(3)
    a <- matrix(rnorm(16), 4, 4)
    expect_qr_correct(a, "f64")
    expect_qr_correct(a, "f32")
  })

  it("works with a donated input buffer (geqrf)", {
    withr::local_seed(4)
    a <- matrix(rnorm(20), 5, 4)
    res <- run_geqrf(a, "f64", donate = TRUE)
    qr <- qr_factors(res[[1L]], res[[2L]], "f64")
    expect_equal(qr$Q %*% qr$R, a, tolerance = 1e-10)
  })

  it("does not overwrite the input buffer (geqrf)", {
    withr::local_seed(5)
    a <- matrix(rnorm(20), 5, 4)
    expect_inputs_preserved(
      "geqrf",
      inputs = list(a),
      in_specs = list(list(dims = c(5, 4), dtype = "f64")),
      out_specs = list(
        list(dims = c(5, 4), dtype = "f64"),
        list(dims = 4, dtype = "f64")
      )
    )
  })
})

# ---------------------------------------------------------------------------
# LU
# ---------------------------------------------------------------------------

describe("lu", {
  # Exercises the `lu` (getrf) LAPACK / cuSOLVER custom call: LU factorisation
  # with partial pivoting on tall, wide, and square matrices in f32 / f64,
  # int32 pivot dtype, input donation, and (CUDA) the copy-before-factor path
  # that prevents the in-place kernel from clobbering the input buffer.
  #
  # Correctness: getrf returns one packed matrix LU plus a 1-based pivot
  # vector. By the LAPACK contract, the strict lower triangle of LU holds L
  # (with an implicit unit diagonal), the upper triangle holds U, and the
  # pivots define a permutation P such that P A = L U. We extract L / U with
  # `upper.tri` / `lower.tri`, undo the row swaps in reverse order to get A
  # back, and compare against the original input.

  # ---- Helpers ----

  # Reconstruct A from packed LU + 1-based pivot indices: L is the strict
  # lower triangle of LU with a unit diagonal, U is the upper triangle, and
  # the pivots define row swaps such that P A = L U. Undo the swaps in
  # reverse order to recover A.
  lu_reconstruct <- function(LU, pivots) {
    k <- min(dim(LU))
    L <- LU[, seq_len(k), drop = FALSE]
    L[upper.tri(L)] <- 0
    diag(L) <- 1
    U <- LU[seq_len(k), , drop = FALSE]
    U[lower.tri(U)] <- 0
    a <- L %*% U
    for (i in rev(seq_along(pivots))) {
      a[c(i, pivots[i]), ] <- a[c(pivots[i], i), ]
    }
    a
  }

  run_lu <- function(a, dtype, donate = FALSE) {
    m <- nrow(a)
    n <- ncol(a)
    k <- min(m, n)
    in_spec <- list(dims = c(m, n), dtype = dtype)
    if (donate) {
      in_spec$aliases <- 1L
    }
    run_linalg(
      "lu",
      inputs = list(a),
      in_specs = list(in_spec),
      out_specs = list(
        list(dims = c(m, n), dtype = dtype),
        list(dims = k, dtype = "i32")
      )
    )
  }

  expect_lu_correct <- function(a, dtype) {
    res <- run_lu(a, dtype)
    tol <- if (dtype == "f64") 1e-10 else 1e-4
    expect_equal(lu_reconstruct(res[[1L]], as.integer(res[[2L]])), a, tolerance = tol)
  }

  # ---- Tests ----

  it("factorises tall (m > n) in f64 and f32", {
    withr::local_seed(11)
    a <- matrix(rnorm(20), 5, 4)
    expect_lu_correct(a, "f64")
    expect_lu_correct(a, "f32")
  })

  it("factorises wide (m < n) in f64 and f32", {
    withr::local_seed(12)
    a <- matrix(rnorm(20), 4, 5)
    expect_lu_correct(a, "f64")
    expect_lu_correct(a, "f32")
  })

  it("factorises square in f64 and f32", {
    withr::local_seed(13)
    a <- matrix(rnorm(16), 4, 4)
    expect_lu_correct(a, "f64")
    expect_lu_correct(a, "f32")
  })

  it("returns int32 pivots", {
    res <- run_lu(matrix(c(0, 1, 1, 1), nrow = 2), "f64")
    expect_true(is.integer(as.vector(res[[2L]])))
  })

  it("works with a donated input buffer", {
    withr::local_seed(14)
    a <- matrix(rnorm(16), 4, 4)
    res <- run_lu(a, "f64", donate = TRUE)
    expect_equal(lu_reconstruct(res[[1L]], as.integer(res[[2L]])), a, tolerance = 1e-10)
  })

  it("does not overwrite the input buffer", {
    withr::local_seed(15)
    a <- matrix(rnorm(16), 4, 4)
    expect_inputs_preserved(
      "lu",
      inputs = list(a),
      in_specs = list(list(dims = c(4, 4), dtype = "f64")),
      out_specs = list(
        list(dims = c(4, 4), dtype = "f64"),
        list(dims = 4, dtype = "i32")
      )
    )
  })
})

# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

describe("svd", {
  # Exercises the `svd` (gesdd / cusolverDnXgesvd) custom call: thin SVD on
  # tall and wide matrices in f32 / f64, input donation, and the CUDA
  # m >= n shape constraint enforced by cuSOLVER.
  #
  # Correctness: thin SVD returns U (m x k), S (k), Vt (k x n) with U and V
  # having orthonormal columns and S non-negative. The factorisation is not
  # unique (per-column sign flips of U / V; arbitrary orthogonal rotations
  # in any repeated-singular-value subspace), so we don't compare U or Vt
  # against a reference. Instead we assert four sign-invariant structural
  # properties: U diag(S) Vt = A, U^T U = I_k, Vt Vt^T = I_k, S >= 0; plus
  # sorted singular values matching R's `svd(a)$d` (those *are* unique).

  # ---- Helpers ----

  reconstruct <- function(res) {
    U <- res[[1L]]
    S <- as.numeric(res[[2L]])
    Vt <- res[[3L]]
    Sd <- if (length(S) == 1L) matrix(S) else diag(S)
    U %*% Sd %*% Vt
  }

  run_svd <- function(a, dtype, donate = FALSE) {
    m <- nrow(a)
    n <- ncol(a)
    k <- min(m, n)
    if (donate) {
      stopifnot(m >= n)
    }
    in_spec <- list(dims = c(m, n), dtype = dtype)
    if (donate) {
      in_spec$aliases <- 1L
    }
    run_linalg(
      "svd",
      inputs = list(a),
      in_specs = list(in_spec),
      out_specs = list(
        list(dims = c(m, k), dtype = dtype),
        list(dims = k, dtype = dtype),
        list(dims = c(k, n), dtype = dtype)
      )
    )
  }

  # Verify the SVD-defining structural properties: U^T U = I_k,
  # Vt Vt^T = I_k, S >= 0, and U diag(S) Vt = A. These together uniquely
  # characterise a valid SVD (up to per-column sign / rotations within
  # repeated-singular-value subspaces — both intentional non-uniqueness),
  # so we don't compare U or Vt directly. Also cross-check sorted singular
  # values against R's `svd()` since those *are* unique.
  expect_matches_r_svd <- function(a, dtype) {
    res <- run_svd(a, dtype)
    U <- res[[1L]]
    S <- as.numeric(res[[2L]])
    Vt <- res[[3L]]
    k <- length(S)
    tol <- if (dtype == "f64") 1e-10 else 1e-4
    expect_equal(reconstruct(res), a, tolerance = tol)
    expect_equal(crossprod(U), diag(k), tolerance = tol)
    expect_equal(tcrossprod(Vt), diag(k), tolerance = tol)
    expect_true(all(S >= 0))
    expect_equal(
      sort(S, decreasing = TRUE),
      sort(svd(a)$d, decreasing = TRUE),
      tolerance = tol
    )
  }

  # ---- Tests ----

  it("factorises a tall matrix (m >= n) in f64 and f32", {
    withr::local_seed(21)
    a <- matrix(rnorm(20), 5, 4)
    expect_matches_r_svd(a, "f64")
    expect_matches_r_svd(a, "f32")
  })

  it("factorises a wide matrix (m < n) in f64 and f32", {
    # CUDA's cusolverDnXgesvd requires m >= n; tested separately below.
    skip_if(is_cuda())
    withr::local_seed(22)
    a <- matrix(rnorm(20), 4, 5)
    expect_matches_r_svd(a, "f64")
    expect_matches_r_svd(a, "f32")
  })

  it("rejects m < n on CUDA", {
    skip_if(!is_cuda())
    expect_error(run_svd(matrix(rnorm(6), 2, 3), "f64"), "m >= n")
  })

  it("works with a donated input buffer", {
    withr::local_seed(24)
    a <- matrix(rnorm(20), 5, 4)
    res <- run_svd(a, "f64", donate = TRUE)
    expect_equal(reconstruct(res), a, tolerance = 1e-10)
  })

  # The thin-SVD output U has shape m x k = m x n when m >= n, matching the
  # input — so it's possible to clobber A in place. (For m < n, U is m x m
  # and the test doesn't apply.)
  it("does not overwrite the input buffer (m >= n)", {
    withr::local_seed(27)
    a <- matrix(rnorm(20), 5, 4)
    expect_inputs_preserved(
      "svd",
      inputs = list(a),
      in_specs = list(list(dims = c(5, 4), dtype = "f64")),
      out_specs = list(
        list(dims = c(5, 4), dtype = "f64"),
        list(dims = 4, dtype = "f64"),
        list(dims = c(4, 4), dtype = "f64")
      )
    )
  })
})

# ---------------------------------------------------------------------------
# eigh
# ---------------------------------------------------------------------------

describe("eigh", {
  # Exercises the `eigh` (syevd / cusolverDnXsyevd) custom call: symmetric
  # eigendecomposition in f32 / f64, the non-square input rejection, input
  # donation, and (CUDA) the copy-before-factor path that preserves the input
  # buffer across the in-place kernel.
  #
  # Correctness: syevd returns V (n x n) and W (n) with V orthogonal and the
  # eigenvalues W real. Per-column signs of V are not unique (flipping a
  # column leaves V diag(W) V^T unchanged), so we don't compare V against a
  # reference. Two sign-invariant checks instead: reconstruction
  # V diag(W) V^T = A, and sorted eigenvalues matching R's `eigen(a)$values`.

  # ---- Helpers ----

  random_spd <- function(n) {
    m <- matrix(rnorm(n * n), n, n)
    m %*% t(m) + diag(n) * 0.5
  }

  # Reconstruct A from V, W: A = V diag(W) V^T.
  reconstruct <- function(res) {
    V <- res[[1L]]
    W <- as.numeric(res[[2L]])
    Wd <- if (length(W) == 1L) matrix(W) else diag(W)
    V %*% Wd %*% t(V)
  }

  run_eigh <- function(a, dtype, donate = FALSE) {
    n <- nrow(a)
    in_spec <- list(dims = dim(a), dtype = dtype)
    if (donate) {
      in_spec$aliases <- 1L
    }
    run_linalg(
      "eigh",
      inputs = list(a),
      in_specs = list(in_spec),
      out_specs = list(
        list(dims = c(n, n), dtype = dtype),
        list(dims = n, dtype = dtype)
      )
    )
  }

  # Verify against R's eigen(): reconstruction (sign-invariant; eigenvectors
  # are only unique up to sign) plus eigenvalues (unique up to ordering).
  expect_matches_r_eigen <- function(a, dtype) {
    res <- run_eigh(a, dtype)
    tol <- if (dtype == "f64") 1e-10 else 1e-4
    expect_equal(reconstruct(res), a, tolerance = tol)
    expect_equal(
      sort(as.numeric(res[[2L]])),
      sort(eigen(a, only.values = TRUE)$values),
      tolerance = tol
    )
  }

  # ---- Tests ----

  it("factorises symmetric / SPD matrices in f64 and f32", {
    withr::local_seed(31)
    m <- matrix(rnorm(25), 5, 5)
    sym <- (m + t(m)) / 2
    spd <- random_spd(6)
    expect_matches_r_eigen(sym, "f64")
    expect_matches_r_eigen(sym, "f32")
    expect_matches_r_eigen(spd, "f64")
  })

  it("rejects non-square input", {
    expect_error(run_eigh(matrix(rnorm(6), nrow = 2), "f64"), "square")
  })

  it("works with a donated input buffer", {
    withr::local_seed(32)
    a <- random_spd(5)
    res <- run_eigh(a, "f64", donate = TRUE)
    expect_equal(reconstruct(res), a, tolerance = 1e-10)
  })

  it("does not overwrite the input buffer", {
    withr::local_seed(33)
    M <- matrix(rnorm(16), 4, 4)
    a <- (M + t(M)) / 2
    expect_inputs_preserved(
      "eigh",
      inputs = list(a),
      in_specs = list(list(dims = c(4, 4), dtype = "f64")),
      out_specs = list(
        list(dims = c(4, 4), dtype = "f64"),
        list(dims = 4, dtype = "f64")
      )
    )
  })
})

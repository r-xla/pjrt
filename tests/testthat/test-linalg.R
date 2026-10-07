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
  # tall and wide matrices in f32 / f64, plus input donation. cuSOLVER's
  # `gesvd` requires m >= n, so for m < n the CUDA handler reads a
  # `transposed` attribute and runs gesvd on A^t via a layout trick; these
  # tests exercise both branches of the FFI directly.
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

  # CUDA-only: when m < n, pass transposed = true and switch operand / U / Vt
  # to row-major layouts. The CPU handler ignores both.
  svd_attrs <- function(m, n) {
    if (!is_cuda()) {
      return(list())
    }
    transposed <- if (m < n) "true" else "false"
    list(backend_config = sprintf("{transposed = %s}", transposed))
  }

  svd_layouts <- function(m, n) {
    if (!is_cuda() || m >= n) {
      return(list(in_layouts = NULL, out_layouts = NULL))
    }
    list(
      in_layouts = row_major_layout(2L),
      out_layouts = c(row_major_layout(2L), col_major_layout(1L), row_major_layout(2L))
    )
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
    layouts <- svd_layouts(m, n)
    run_linalg(
      "svd",
      inputs = list(a),
      in_specs = list(in_spec),
      out_specs = list(
        list(dims = c(m, k), dtype = dtype),
        list(dims = k, dtype = dtype),
        list(dims = c(k, n), dtype = dtype)
      ),
      attrs = svd_attrs(m, n),
      in_layouts = layouts$in_layouts,
      out_layouts = layouts$out_layouts
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
    withr::local_seed(22)
    a <- matrix(rnorm(20), 4, 5)
    expect_matches_r_svd(a, "f64")
    expect_matches_r_svd(a, "f32")
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
      ),
      attrs = svd_attrs(5, 4)
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

# ---------------------------------------------------------------------------
# potrf (Cholesky)
# ---------------------------------------------------------------------------

describe("potrf", {
  # Exercises the `potrf` LAPACK / cuSOLVER custom call: Cholesky
  # factorisation of a batch of matrices [..., n, n] in f32 / f64, either
  # triangle, plus the per-matrix `info` output that reports a matrix that is
  # not positive definite instead of failing.
  #
  # Correctness: the triangle selected by `lower` must match R's `chol()` (U)
  # or its transpose (L); the other triangle is unspecified and not checked.

  # ---- Helpers ----

  random_spd <- function(n) {
    m <- matrix(rnorm(n * n), n, n)
    crossprod(m) + diag(n)
  }

  run_potrf <- function(a, dtype, lower, donate = FALSE) {
    d <- dim(a)
    nd <- length(d)
    in_spec <- list(dims = d, dtype = dtype)
    if (donate) {
      in_spec$aliases <- 1L
    }
    run_linalg(
      "potrf",
      inputs = list(a),
      in_specs = list(in_spec),
      out_specs = list(
        list(dims = d, dtype = dtype),
        list(dims = d[seq_len(nd - 2L)], dtype = "i32")
      ),
      attrs = list(backend_config = sprintf("{lower = %s}", tolower(lower))),
      in_layouts = batched_matrix_layout(nd),
      out_layouts = c(batched_matrix_layout(nd), row_major_layout(nd - 2L))
    )
  }

  # The triangle of `factor` that potrf wrote, the other one zeroed.
  triangle <- function(factor, lower) {
    factor[if (lower) upper.tri(factor) else lower.tri(factor)] <- 0
    factor
  }

  expect_matches_r_chol <- function(a, dtype, lower) {
    res <- run_potrf(a, dtype, lower)
    tol <- if (dtype == "f64") 1e-10 else 1e-4
    expected <- if (lower) t(chol(a)) else chol(a)
    expect_equal(triangle(res[[1L]], lower), expected, tolerance = tol)
    expect_equal(as.integer(res[[2L]]), 0L)
  }

  # ---- Tests ----

  it("factorises into the upper and the lower triangle in f64 and f32", {
    withr::local_seed(41)
    a <- random_spd(6)
    for (dtype in c("f64", "f32")) {
      expect_matches_r_chol(a, dtype, lower = FALSE)
      expect_matches_r_chol(a, dtype, lower = TRUE)
    }
  })

  it("reports the order of the first minor that is not positive definite", {
    a <- diag(c(1, 2, -1, 3))
    for (dtype in c("f64", "f32")) {
      for (lower in c(TRUE, FALSE)) {
        res <- run_potrf(a, dtype, lower)
        expect_equal(as.integer(res[[2L]]), 3L)
      }
    }
  })

  it("factorises every matrix of a batch and reports info per matrix", {
    withr::local_seed(42)
    n <- 4L
    a <- array(0, c(2L, 3L, n, n))
    for (i in 1:2) {
      for (j in 1:3) {
        a[i, j, , ] <- random_spd(n)
      }
    }
    a[2L, 1L, , ] <- -diag(n)
    res <- run_potrf(a, "f64", lower = TRUE)
    expect_equal(dim(res[[1L]]), dim(a))
    expect_equal(as.integer(res[[2L]]), c(0L, 1L, 0L, 0L, 0L, 0L))
    for (i in 1:2) {
      for (j in 1:3) {
        if (i == 2L && j == 1L) {
          next
        }
        expect_equal(
          triangle(res[[1L]][i, j, , ], lower = TRUE),
          t(chol(a[i, j, , ])),
          tolerance = 1e-10
        )
      }
    }
  })

  it("works with a donated input buffer", {
    withr::local_seed(43)
    a <- random_spd(5)
    res <- run_potrf(a, "f64", lower = FALSE, donate = TRUE)
    expect_equal(triangle(res[[1L]], lower = FALSE), chol(a), tolerance = 1e-10)
  })

  it("does not overwrite the input buffer", {
    withr::local_seed(44)
    a <- random_spd(4)
    expect_inputs_preserved(
      "potrf",
      inputs = list(a),
      in_specs = list(list(dims = c(4, 4), dtype = "f64")),
      out_specs = list(
        list(dims = c(4, 4), dtype = "f64"),
        list(dims = integer(), dtype = "i32")
      ),
      attrs = list(backend_config = "{lower = true}"),
      out_layouts = c(col_major_layout(2L), col_major_layout(0L))
    )
  })
})

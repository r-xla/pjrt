# End-to-end property tests for the built-in LAPACK / cuSOLVER custom_call
# handlers (`qr`, `lu`, `svd`, `eigh`).
#
# Each handler is exercised through a small JIT-compiled stablehlo.custom_call
# program (see helper-linalg.R for the harness). We don't depend on anvl --
# pjrt is the layer that owns these handlers, so the tests live here.
#
# Coverage strategy: a small set of "shape suites" generated for each op
# (square, tall, wide where the op supports it; 1x1; an identity matrix)
# crossed with f32 and f64. For each generated input we verify the
# defining property of the factorisation (reconstruction, orthogonality,
# triangularity, etc.) and -- where possible -- compare against base R's
# `qr()` / `solve()` / `svd()` / `eigen()`.

skip_if_metal("linalg custom_calls are CPU/CUDA only")

# Always test on the host (LAPACK) backend; the CUDA path is exercised when
# PJRT_PLATFORM=cuda is set in the environment. The CUDA SVD has an m >= n
# requirement (cuSOLVER's gesvd); we skip the wide-SVD tests on CUDA below.

# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------

qr_check <- function(a, label) {
  res <- run_qr(a)
  q <- res[[1L]]; r <- res[[2L]]
  m <- nrow(a); n <- ncol(a); k <- min(m, n)

  expect_equal(dim(q), c(m, k), info = label)
  expect_equal(dim(r), c(k, n), info = label)

  tol <- linalg_tol(a)
  # Reconstruction
  expect_equal(q %*% r, a, tolerance = tol, info = paste(label, "QR=A"))
  # Q has orthonormal columns
  expect_equal(t(q) %*% q, diag(k), tolerance = tol,
               info = paste(label, "Q^T Q = I"))
  # R is upper triangular: the leading k x k block has zeros below the diagonal.
  if (k > 1L) {
    R_block <- r[seq_len(k), seq_len(k), drop = FALSE]
    expect_lt(max(abs(R_block[lower.tri(R_block)])), tol * 10,
              label = paste(label, "R upper-triangular"))
  }
}

test_that("qr: square / tall / wide / 1x1 / identity (f64)", {
  set.seed(1)
  cases <- list(
    list(label = "1x1",          a = matrix(2.5, 1, 1)),
    list(label = "2x2",          a = matrix(c(1, 2, 3, 4), nrow = 2)),
    list(label = "3x3 random",   a = matrix(rnorm(9), nrow = 3)),
    list(label = "5x5 random",   a = matrix(rnorm(25), nrow = 5)),
    list(label = "tall 7x3",     a = matrix(rnorm(21), nrow = 7)),
    list(label = "wide 3x7",     a = matrix(rnorm(21), nrow = 3)),
    list(label = "identity 4x4", a = diag(4)),
    list(label = "20x20 random", a = matrix(rnorm(400), nrow = 20))
  )
  for (cs in cases) qr_check(cs$a, cs$label)
})

test_that("qr: f32 path", {
  set.seed(2)
  a <- matrix(rnorm(12), nrow = 4)
  res <- run_linalg(
    "qr", a,
    list(list(dims = c(4, 3), dtype = "f32"),
         list(dims = c(3, 3), dtype = "f32")),
    dtype = "f32"
  )
  q <- res[[1L]]; r <- res[[2L]]
  expect_equal(q %*% r, a, tolerance = 1e-4)
  expect_equal(t(q) %*% q, diag(3), tolerance = 1e-4)
})

test_that("qr: agrees with base::qr() up to column signs", {
  # base::qr returns a packed factorization but we can extract Q, R via
  # qr.Q / qr.R for direct comparison. Both LAPACK paths use the same
  # algorithm, so the result is unique up to the signs of R's diagonal.
  set.seed(3)
  for (dims in list(c(5, 3), c(4, 4), c(3, 5))) {
    a <- matrix(rnorm(prod(dims)), nrow = dims[1L])
    base_qr <- qr(a)
    Q_base <- qr.Q(base_qr); R_base <- qr.R(base_qr)
    res <- run_qr(a)
    Q <- res[[1L]]; R <- res[[2L]]
    # Resolve sign ambiguity: align signs of diagonals of R.
    k <- min(dims)
    sgn <- sign(diag(R)[seq_len(k)])
    sgn_base <- sign(diag(R_base)[seq_len(k)])
    flip <- sgn * sgn_base
    R <- diag(flip, k) %*% R
    Q <- Q %*% diag(flip, k)
    expect_equal(R, R_base, tolerance = 1e-10)
    expect_equal(Q, Q_base, tolerance = 1e-10)
  }
})

test_that("qr: input buffer is not modified", {
  a <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
  a0 <- a
  run_qr(a)
  expect_identical(a, a0)
})

# ---------------------------------------------------------------------------
# LU
# ---------------------------------------------------------------------------

apply_lu_pivots <- function(perm_init, pivots) {
  perm <- perm_init
  for (i in seq_along(pivots)) {
    j <- pivots[[i]]
    if (j != i) perm[c(i, j)] <- perm[c(j, i)]
  }
  perm
}

lu_check <- function(a, label) {
  res <- run_lu(a)
  LU <- res[[1L]]
  pivots <- as.integer(res[[2L]])
  m <- nrow(a); n <- ncol(a); k <- min(m, n)

  expect_equal(dim(LU), c(m, n), info = label)
  expect_equal(length(pivots), k, info = label)
  expect_true(all(pivots >= 1L & pivots <= m),
              info = paste(label, "pivots in [1, m]"))

  # Reconstruct L (m x k, unit lower-triangular) and U (k x n, upper).
  L <- matrix(0, nrow = m, ncol = k)
  diag(L) <- 1
  for (j in seq_len(k)) {
    if (j + 1L <= m) {
      L[(j + 1L):m, j] <- LU[(j + 1L):m, j]
    }
  }
  U <- matrix(0, nrow = k, ncol = n)
  for (i in seq_len(k)) {
    U[i, i:n] <- LU[i, i:n]
  }
  PA <- L %*% U
  perm <- apply_lu_pivots(seq_len(m), pivots)
  expect_equal(PA[order(perm), , drop = FALSE], a,
               tolerance = linalg_tol(a),
               info = paste(label, "P^-1 L U = A"))
}

test_that("lu: square / tall / wide / 1x1 / identity / forced pivot", {
  set.seed(11)
  lu_check(matrix(2.5, 1, 1), "1x1")
  lu_check(matrix(c(4, 3, 6, 3), nrow = 2), "2x2 no pivot")
  # First row has zero in column 1, so getrf must swap.
  lu_check(matrix(c(0, 1, 1, 1), nrow = 2), "2x2 forced pivot")
  lu_check(matrix(rnorm(9), nrow = 3), "3x3 random")
  lu_check(matrix(rnorm(25), nrow = 5), "5x5 random")
  lu_check(matrix(rnorm(21), nrow = 7), "tall 7x3")
  lu_check(matrix(rnorm(21), nrow = 3), "wide 3x7")
  lu_check(diag(4), "identity 4x4")
  lu_check(matrix(rnorm(900), nrow = 30), "30x30 random")
})

test_that("lu: f32 reconstruction", {
  set.seed(12)
  a <- matrix(rnorm(16), nrow = 4)
  res <- run_linalg(
    "lu", a,
    list(list(dims = c(4, 4), dtype = "f32"),
         list(dims = 4, dtype = "i32")),
    dtype = "f32"
  )
  LU <- res[[1L]]
  pivots <- as.integer(res[[2L]])
  L <- diag(4)
  L[lower.tri(L)] <- LU[lower.tri(LU)]
  U <- LU
  U[lower.tri(U)] <- 0
  perm <- apply_lu_pivots(seq_len(4), pivots)
  expect_equal((L %*% U)[order(perm), ], a, tolerance = 1e-4)
})

test_that("lu: pivots are int32 (validates result_layouts on a non-float output)", {
  res <- run_lu(matrix(c(0, 1, 1, 1), nrow = 2))
  expect_true(is.integer(as.vector(res[[2L]])))
})

# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

svd_check <- function(a, label) {
  res <- run_svd(a)
  U <- res[[1L]]; S <- as.numeric(res[[2L]]); Vt <- res[[3L]]
  m <- nrow(a); n <- ncol(a); k <- min(m, n)

  expect_equal(dim(U), c(m, k), info = label)
  expect_equal(length(S), k, info = label)
  expect_equal(dim(Vt), c(k, n), info = label)

  tol <- linalg_tol(a)
  # Singular values: non-negative, descending
  expect_true(all(S >= -tol),
              info = paste(label, "S >= 0"))
  if (length(S) > 1L) {
    expect_true(all(diff(S) <= tol),
                info = paste(label, "S descending"))
  }

  # U has orthonormal columns
  expect_equal(t(U) %*% U, diag(k), tolerance = tol,
               info = paste(label, "U^T U = I"))
  # V (= t(Vt)) has orthonormal columns
  expect_equal(Vt %*% t(Vt), diag(k), tolerance = tol,
               info = paste(label, "V^T V = I"))
  # Reconstruction
  Sd <- if (length(S) == 1L) matrix(S) else diag(S)
  expect_equal(U %*% Sd %*% Vt, a, tolerance = tol,
               info = paste(label, "U S Vt = A"))

  # Singular values match base::svd
  base_S <- svd(a)$d
  expect_equal(S, base_S, tolerance = tol,
               info = paste(label, "S matches base::svd"))
}

test_that("svd: square / tall / wide / 1x1 / identity (host)", {
  skip_if(is_cuda(), "wide-SVD on CUDA is unsupported (cuSOLVER gesvd: m >= n)")
  set.seed(21)
  svd_check(matrix(2.5, 1, 1), "1x1")
  svd_check(matrix(c(1, 0, 0, 1), nrow = 2), "identity 2x2")
  svd_check(matrix(rnorm(9), nrow = 3), "3x3 random")
  svd_check(matrix(rnorm(25), nrow = 5), "5x5 random")
  svd_check(matrix(rnorm(21), nrow = 7), "tall 7x3")
  svd_check(matrix(rnorm(21), nrow = 3), "wide 3x7 (host)")
  svd_check(matrix(rnorm(400), nrow = 20), "20x20 random")
})

test_that("svd: square / tall / 1x1 (CUDA - no wide)", {
  skip_if(!is_cuda())
  set.seed(22)
  svd_check(matrix(2.5, 1, 1), "1x1")
  svd_check(matrix(rnorm(9), nrow = 3), "3x3 random")
  svd_check(matrix(rnorm(21), nrow = 7), "tall 7x3")
})

test_that("svd: CUDA rejects m < n", {
  skip_if(!is_cuda())
  a <- matrix(rnorm(6), nrow = 2)  # 2x3, m < n
  expect_error(run_svd(a), "m >= n")
})

test_that("svd: f32 reconstruction", {
  set.seed(23)
  a <- matrix(rnorm(20), nrow = 5)
  res <- run_linalg(
    "svd", a,
    list(list(dims = c(5, 4), dtype = "f32"),
         list(dims = 4, dtype = "f32"),
         list(dims = c(4, 4), dtype = "f32")),
    dtype = "f32"
  )
  U <- res[[1L]]; S <- as.numeric(res[[2L]]); Vt <- res[[3L]]
  expect_equal(U %*% diag(S) %*% Vt, a, tolerance = 1e-4)
})

test_that("svd: ill-conditioned input still factorises correctly", {
  # Diagonal matrix with widely varying singular values.
  set.seed(24)
  s_true <- 10 ^ seq(0, -10, length.out = 5)  # 1, 1e-2.5, ..., 1e-10
  a <- diag(s_true)
  res <- run_svd(a)
  S <- as.numeric(res[[2L]])
  # Check the singular values come back in the right order, with high
  # relative accuracy on the dominant ones.
  expect_equal(S, sort(s_true, decreasing = TRUE), tolerance = 1e-9)
})

# ---------------------------------------------------------------------------
# eigh
# ---------------------------------------------------------------------------

random_symmetric <- function(n) {
  m <- matrix(rnorm(n * n), n, n)
  (m + t(m)) / 2
}
random_spd <- function(n) {
  m <- matrix(rnorm(n * n), n, n)
  m %*% t(m) + diag(n) * 0.5  # add small ridge
}

eigh_check <- function(a, label, ascending = TRUE) {
  res <- run_eigh(a)
  V <- res[[1L]]; W <- as.numeric(res[[2L]])
  n <- nrow(a)

  expect_equal(dim(V), c(n, n), info = label)
  expect_equal(length(W), n, info = label)

  tol <- linalg_tol(a, scale = 5)
  # Eigenvalues sorted ascending.
  if (n > 1L && ascending) {
    expect_true(all(diff(W) >= -tol),
                info = paste(label, "W ascending"))
  }
  # Orthonormal eigenvectors.
  expect_equal(t(V) %*% V, diag(n), tolerance = tol,
               info = paste(label, "V^T V = I"))
  # Reconstruction (symmetric A; we only fed the lower triangle but the
  # input was symmetric so we can compare against the full matrix).
  Wd <- if (n == 1L) matrix(W) else diag(W)
  expect_equal(V %*% Wd %*% t(V), a, tolerance = tol,
               info = paste(label, "V W V^T = A"))
}

test_that("eigh: 1x1 / 2x2 / random symmetric of varying sizes", {
  set.seed(31)
  eigh_check(matrix(2.5, 1, 1), "1x1")
  eigh_check(matrix(c(2, 1, 1, 2), nrow = 2), "2x2 known")
  eigh_check(random_symmetric(5), "5x5 symmetric")
  eigh_check(random_spd(5), "5x5 SPD")
  eigh_check(random_symmetric(10), "10x10 symmetric")
  eigh_check(random_spd(20), "20x20 SPD")
  # Identity: all eigenvalues 1, eigenvectors are any orthonormal basis.
  res <- run_eigh(diag(4))
  expect_equal(as.numeric(res[[2L]]), c(1, 1, 1, 1), tolerance = 1e-12)
})

test_that("eigh: 2x2 with known eigenvalues", {
  # [[2, 1], [1, 2]] has eigenvalues {1, 3} (ascending).
  res <- run_eigh(matrix(c(2, 1, 1, 2), nrow = 2))
  expect_equal(as.numeric(res[[2L]]), c(1, 3), tolerance = 1e-12)
})

test_that("eigh: matches base::eigen on symmetric input", {
  set.seed(32)
  for (n in c(3L, 6L, 10L)) {
    a <- random_symmetric(n)
    base_e <- eigen(a, symmetric = TRUE)  # base::eigen returns descending
    res <- run_eigh(a)
    W <- as.numeric(res[[2L]])
    # Reverse W to match base::eigen ordering.
    expect_equal(rev(W), base_e$values, tolerance = 1e-10,
                 info = paste0("n=", n))
  }
})

test_that("eigh: f32 reconstruction", {
  set.seed(33)
  a <- random_spd(6)
  res <- run_linalg(
    "eigh", a,
    list(list(dims = c(6, 6), dtype = "f32"),
         list(dims = 6, dtype = "f32")),
    dtype = "f32"
  )
  V <- res[[1L]]; W <- as.numeric(res[[2L]])
  expect_equal(V %*% diag(W) %*% t(V), a, tolerance = 1e-4)
})

test_that("eigh: rejects non-square input", {
  expect_error(run_eigh(matrix(rnorm(6), nrow = 2)), "square")
})

# ---------------------------------------------------------------------------
# Cross-cutting concerns
# ---------------------------------------------------------------------------

test_that("all four built-in linalg handlers are registered", {
  registered <- names(the[["custom_calls"]])
  expect_setequal(intersect(c("qr", "lu", "svd", "eigh"), registered),
                  c("qr", "lu", "svd", "eigh"))
})

test_that("repeated calls reuse cuSOLVER handles (no leak / no recreate)", {
  # On the host this is essentially a smoke test; on CUDA it exercises the
  # SolverHandlePool's borrow-and-return path many times in a row. If the
  # pool were broken (handle dropped, double-released, etc.) we'd see
  # cuSOLVER status errors before long.
  set.seed(99)
  a <- matrix(rnorm(25), nrow = 5)
  for (i in 1:20) {
    res <- run_qr(a)
    expect_equal(res[[1L]] %*% res[[2L]], a, tolerance = 1e-10)
  }
})

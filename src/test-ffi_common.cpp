#include <testthat.h>

#include <limits>

#include "ffi_common.h"

context("workspace_to_int") {
  test_that("converts a workspace size that fits in int") {
    int lwork = -1;
    expect_true(rpjrt::workspace_to_int(1024.0, "gesdd", lwork).success());
    expect_true(lwork == 1024);
    expect_true(
        rpjrt::workspace_to_int(2147483647.0, "gesdd", lwork).success());
    expect_true(lwork == std::numeric_limits<int>::max());
  }

  test_that("rejects a workspace size past INT_MAX") {
    int lwork = -1;
    expect_false(
        rpjrt::workspace_to_int(2147483648.0, "gesdd", lwork).success());
    // A float cannot hold INT_MAX; the nearest float above it must not pass.
    expect_false(
        rpjrt::workspace_to_int(2147483648.0f, "gesdd", lwork).success());
    expect_false(rpjrt::workspace_to_int(4e9, "gesdd", lwork).success());
    expect_true(lwork == -1);
  }

  test_that("rejects a negative or NaN workspace size") {
    int lwork = -1;
    expect_false(rpjrt::workspace_to_int(-1.0, "gesdd", lwork).success());
    expect_false(rpjrt::workspace_to_int(
                     std::numeric_limits<double>::quiet_NaN(), "gesdd", lwork)
                     .success());
  }
}

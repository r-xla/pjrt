#include <Rcpp.h>

// Reports whether the Catch C++ unit tests were compiled into this build.
// They are gated behind PJRT_BUILD_CPP_TESTS (see configure), which defines
// TESTTHAT_DISABLED when unset. test-cpp.R uses this to skip run_cpp_tests()
// when the tests -- and thus the Catch XML runner -- are absent.
// [[Rcpp::export]]
bool cpp_tests_enabled() {
#ifdef TESTTHAT_DISABLED
  return false;
#else
  return true;
#endif
}

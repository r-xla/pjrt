# The C++ unit tests are compiled out unless PJRT_BUILD_CPP_TESTS was set at
# build time (see configure); when absent, run_cpp_tests() would choke on the
# empty Catch output, so skip.
skip_if_not(cpp_tests_enabled(), "C++ unit tests were compiled out (set PJRT_BUILD_CPP_TESTS=1)")
# run_cpp_tests() requires xml2 to parse the Catch output; skip when it (a
# testthat Suggests dependency) is unavailable, e.g. in noSuggests CI runs.
skip_if_not_installed("xml2")

run_cpp_tests("pjrt")

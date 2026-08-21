// The consumer-facing half of a deliberately shadowed pair.
//
// Rcpp attributes writes `#include "pjrt_types.h"` into everything it
// generates, so both src/RcppExports.cpp and the installed
// inst/include/pjrt_RcppExports.h ask for a header of this name. They must not
// get the same one:
//
//   * src/RcppExports.cpp declares entry points whose signatures name pjrt's
//     own C++ types (Rcpp::XPtr<rpjrt::PJRTBuffer> and friends), so it needs
//     src/pjrt_types.h, which pulls in buffer.h, client.h and the rest.
//
//   * inst/include/pjrt_RcppExports.h is compiled inside *other* packages.
//     Handing them pjrt's internal headers would make its class layouts part of
//     the cross-package contract -- precisely what the interface exists to
//     avoid -- and they would not compile anyway, since those headers are not
//     installed.
//
// A quoted include searches the including file's own directory first, so the
// two files in src/ resolve to src/pjrt_types.h and a downstream package
// resolves to this one. Nothing else is needed here: every signature in
// src/capi_cpp.cpp is built from SEXP, Rcpp::List and standard-library types on
// purpose, so this header only has to exist and to provide what those need.
//
// If a future exported signature does name a pjrt type, that is the signal to
// reconsider the boundary rather than to start installing internal headers.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

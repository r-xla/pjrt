// pjrt's C++ interface for downstream packages: the half of the boundary that
// Rcpp generates.
//
// Rcpp attributes turns the [[Rcpp::export]]s in this file into inline wrappers
// in inst/include/pjrt_RcppExports.h (namespace `pjrt`), which resolve through
// R_GetCCallable exactly as the hand-written C entry points in capi.cpp do. The
// transport is the same; what differs is what the wrapper does around it.
//
// Why these functions and not the others
// --------------------------------------
// The generated wrapper marshals every argument and result through a SEXP
// (Rcpp::wrap / Rcpp::as) and inspects the result on the way back. That buys
// something real: an R error, an interrupt, or a longjump raised in here comes
// back as a *value* and is re-thrown in the caller's own translation unit,
// instead of tearing through its C++ frames without running destructors. The
// hand-written boundary cannot do that -- it traps C++ exceptions, and an R
// longjmp goes straight past it.
//
// It also costs something real: an allocation per marshalled argument and
// result. So the split follows where each side pays off. Everything here
//
//   * already allocates a device buffer or runs a computation, so the
//     marshalling is lost in the noise, and
//   * can genuinely fail -- a PJRT error, or an out-of-memory that runs R's
//     garbage collector and its finalizers, which is R code that can longjmp.
//
// Everything in capi.cpp is the opposite: a non-allocating read of memoized
// metadata that runs once per array input per dispatch and cannot reach R at
// all. Those stay hand-written C, because there the per-call allocation is the
// whole cost and there is no longjmp to guard against.
//
// `rng = false` on each export drops the RNGScope the wrapper would otherwise
// construct per call; none of this touches R's RNG.

// [[Rcpp::interfaces(cpp)]]

#include <Rcpp.h>

#include <cstdint>
#include <string>
#include <vector>

#include "buffer.h"
#include "client.h"
#include "device.h"
#include "pjrt_impl.h"

using rpjrt::PJRTBuffer;
using rpjrt::PJRTClient;
using rpjrt::PJRTDevice;
using rpjrt::PJRTExecuteOptions;
using rpjrt::PJRTLoadedExecutable;

namespace {

// Rcpp::XPtr checks only that a SEXP is an external pointer, never what it
// points at, so the class is verified before any of these casts. Throwing is
// the right move here (unlike in capi.cpp): the generated wrapper catches it
// and hands it back to the caller as a value.
void require_class(SEXP x, const char* cls) {
  if (TYPEOF(x) != EXTPTRSXP || !Rf_inherits(x, cls)) {
    Rcpp::stop("expected a %s", cls);
  }
}

}  // namespace

// Run a compiled executable. `inputs` is a list of PJRTBuffers in program
// order; the result is a list of PJRTBuffers. Donation and input keepalives are
// handled inside.
// [[Rcpp::export(rng = false)]]
SEXP execute(SEXP executable, Rcpp::List inputs, SEXP options) {
  require_class(executable, "PJRTLoadedExecutable");
  require_class(options, "PJRTExecuteOptions");
  // Each element is handed straight to the PJRT C API as a PJRTBuffer; a
  // wrong-classed external pointer there would be reinterpreted blindly and
  // crash, so they are checked rather than trusted.
  const R_xlen_t n = inputs.size();
  for (R_xlen_t i = 0; i < n; ++i) {
    if (TYPEOF(inputs[i]) != EXTPTRSXP ||
        !Rf_inherits(inputs[i], "PJRTBuffer")) {
      Rcpp::stop("`inputs[[%d]]` must be a PJRTBuffer",
                 static_cast<int>(i) + 1);
    }
  }
  return impl_loaded_executable_execute(
      Rcpp::XPtr<PJRTLoadedExecutable>(executable), inputs,
      Rcpp::XPtr<PJRTExecuteOptions>(options));
}

// Upload an R vector or array to a device buffer, with the same conversion and
// column-major handling as pjrt_buffer() / pjrt_scalar(). `data` must be a
// REALSXP, INTSXP or LGLSXP; `dims` is the logical shape (empty for a scalar).
// [[Rcpp::export(rng = false)]]
SEXP buffer_from_r(SEXP client, SEXP device, SEXP data,
                   std::vector<int64_t> dims, std::string dtype) {
  require_class(client, "PJRTClient");
  require_class(device, "PJRTDevice");
  Rcpp::XPtr<PJRTClient> cl(client);
  Rcpp::XPtr<PJRTDevice> dev(device);
  switch (TYPEOF(data)) {
    case REALSXP:
      return impl_client_buffer_from_double(cl, dev, data, dims, dtype);
    case INTSXP:
      return impl_client_buffer_from_integer(cl, dev, data, dims, dtype);
    case LGLSXP:
      return impl_client_buffer_from_logical(cl, dev, data, dims, dtype);
    default:
      Rcpp::stop("`data` must be a double, integer or logical vector");
  }
}

// Allocate an uninitialized buffer -- what an output-donation "phantom" input
// needs.
// [[Rcpp::export(rng = false)]]
SEXP buffer_empty(SEXP client, SEXP device, std::vector<int64_t> dims,
                  std::string dtype) {
  require_class(client, "PJRTClient");
  require_class(device, "PJRTDevice");
  return impl_client_buffer_empty(Rcpp::XPtr<PJRTClient>(client),
                                  Rcpp::XPtr<PJRTDevice>(device), dims, dtype);
}

// Copy a buffer to another device. `cross_client` selects the host round-trip
// needed when source and destination belong to different clients; compare
// pjrt_c_buffer_api() with pjrt_c_client_api() to decide.
// [[Rcpp::export(rng = false)]]
SEXP buffer_copy_to_device(SEXP buffer, SEXP device, SEXP dst_client,
                           bool cross_client) {
  require_class(buffer, "PJRTBuffer");
  require_class(device, "PJRTDevice");
  require_class(dst_client, "PJRTClient");
  return impl_buffer_copy_to_device(
      Rcpp::XPtr<PJRTBuffer>(buffer), Rcpp::XPtr<PJRTDevice>(device),
      Rcpp::XPtr<PJRTClient>(dst_client), cross_client);
}

// A reusable PJRTExecuteOptions. `non_donatable_indices` holds 0-based input
// positions.
// [[Rcpp::export(rng = false)]]
SEXP execution_options(std::vector<int64_t> non_donatable_indices,
                       int launch_id) {
  return impl_execution_options_create(non_donatable_indices, launch_id);
}

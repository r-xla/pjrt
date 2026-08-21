// Tests for pjrt's C interface (inst/include/pjrt/api.h).
//
// These deliberately go the long way round: this translation unit includes the
// public header *without* PJRT_C_API_IMPLEMENTATION, so every call below runs
// through the same R_GetCCallable-resolving stub a downstream package uses.
// That means they cover the registration (`pjrt_register_c_api`, reached via
// [[Rcpp::init]]) and the header, not just the implementations in capi.cpp --
// which is the half that breaks silently, since a symbol that is never
// registered only fails in someone else's package.
//
// Exposed as ordinary [[Rcpp::export]] entry points so tests/testthat/
// test-capi.R can drive them; they are not part of pjrt's R API.

#include <Rcpp.h>

#include "pjrt/api.h"
// The generated half, included exactly as a downstream package would.
#include "pjrt_RcppExports.h"

// The version the header declares against what the registered entry reports.
// [[Rcpp::export]]
Rcpp::IntegerVector impl_capi_versions() {
  return Rcpp::IntegerVector::create(
      Rcpp::Named("header") = PJRT_C_API_VERSION,
      Rcpp::Named("registered") = pjrt_c_api_version());
}

// [[Rcpp::export]]
Rcpp::LogicalVector impl_capi_predicates(SEXP x) {
  return Rcpp::LogicalVector::create(
      Rcpp::Named("buffer") = pjrt_c_is_buffer(x) != 0,
      Rcpp::Named("client") = pjrt_c_is_client(x) != 0,
      Rcpp::Named("device") = pjrt_c_is_device(x) != 0,
      Rcpp::Named("executable") = pjrt_c_is_executable(x) != 0);
}

// Buffer metadata as the dispatcher reads it: dtype, shape, and the interned
// device object.
// [[Rcpp::export]]
Rcpp::List impl_capi_buffer_meta(SEXP buffer) {
  int rank = -1;
  const int64_t* dims = pjrt_c_buffer_dims(buffer, &rank);
  Rcpp::IntegerVector shape(rank < 0 ? 0 : rank);
  for (int i = 0; i < rank; ++i) shape[i] = static_cast<int>(dims[i]);
  const int dtype = pjrt_c_buffer_dtype(buffer);
  const char* nm = pjrt_c_dtype_name(dtype);
  return Rcpp::List::create(
      Rcpp::Named("dtype") = dtype,
      Rcpp::Named("dtype_name") = nm == nullptr ? "" : std::string(nm),
      Rcpp::Named("rank") = rank, Rcpp::Named("shape") = shape,
      Rcpp::Named("device") = pjrt_c_device_for_buffer(buffer));
}

// The dtype vocabulary, both ways.
// [[Rcpp::export]]
Rcpp::List impl_capi_dtype_roundtrip(std::string name) {
  const int code = pjrt_c_dtype_from_name(name.c_str());
  const char* back = code < 0 ? nullptr : pjrt_c_dtype_name(code);
  return Rcpp::List::create(
      Rcpp::Named("code") = code,
      Rcpp::Named("name") = back == nullptr ? "" : std::string(back));
}

// The error channel: a wrong-classed argument must return the sentinel and
// leave a message, and a subsequent success must clear it.
// [[Rcpp::export]]
Rcpp::List impl_capi_error_channel(SEXP not_a_buffer) {
  const int bad = pjrt_c_buffer_dtype(not_a_buffer);
  const char* e1 = pjrt_c_last_error();
  std::string msg = e1 == nullptr ? "" : std::string(e1);
  // A call that succeeds must clear the message.
  const int ok = pjrt_c_dtype_from_name("f32");
  const char* e2 = pjrt_c_last_error();
  // An unknown dtype fails without throwing.
  const int unknown = pjrt_c_dtype_from_name("no_such_dtype");
  const char* e3 = pjrt_c_last_error();
  return Rcpp::List::create(
      Rcpp::Named("bad_dtype") = bad, Rcpp::Named("message") = msg,
      Rcpp::Named("cleared") = e2 == nullptr, Rcpp::Named("good_dtype") = ok,
      Rcpp::Named("unknown_dtype") = unknown,
      Rcpp::Named("unknown_message") = e3 == nullptr ? "" : std::string(e3));
}

// Identity tokens: same device from two routes must agree, and a buffer must
// report the device it actually lives on.
// [[Rcpp::export]]
Rcpp::LogicalVector impl_capi_device_identity(SEXP buffer, SEXP device) {
  SEXP from_buffer = pjrt_c_device_for_buffer(buffer);
  SEXP canonical = pjrt_c_device_canonical(device);
  return Rcpp::LogicalVector::create(
      // pjrt interns devices, so these are the *same object*, not merely equal.
      Rcpp::Named("same_object") = from_buffer == canonical,
      Rcpp::Named("same_token") =
          pjrt_c_buffer_device_ptr(buffer) == pjrt_c_device_ptr(device),
      // A canonical device must be idempotent under further interning.
      Rcpp::Named("idempotent") =
          pjrt_c_device_canonical(canonical) == canonical);
}

// [[Rcpp::export]]
bool impl_capi_same_client(SEXP buffer, SEXP client) {
  return pjrt_c_buffer_api(buffer) == pjrt_c_client_api(client);
}

// Allocation and execution, through the Rcpp-generated half. These need no
// sentinel checking: a failure inside pjrt comes back as a thrown exception.
// [[Rcpp::export]]
SEXP impl_capi_buffer_from_r(SEXP client, SEXP device, SEXP data,
                             std::vector<int64_t> dims, std::string dtype) {
  return pjrt::buffer_from_r(client, device, data, dims, dtype);
}

// [[Rcpp::export]]
SEXP impl_capi_buffer_empty(SEXP client, SEXP device, std::vector<int64_t> dims,
                            std::string dtype) {
  return pjrt::buffer_empty(client, device, dims, dtype);
}

// [[Rcpp::export]]
SEXP impl_capi_execute(SEXP executable, Rcpp::List inputs) {
  Rcpp::Shield<SEXP> opts(pjrt::execution_options(std::vector<int64_t>(), 0));
  return pjrt::execute(executable, inputs, opts);
}

// A failure in the generated half must arrive as a C++ exception the caller can
// catch in its own frame -- not as a longjmp past it, and not as a sentinel.
// Catching it here is the whole point: it proves the error crossed the package
// boundary as a value.
// [[Rcpp::export]]
std::string impl_capi_execute_error(SEXP executable, Rcpp::List bad_inputs) {
  Rcpp::Shield<SEXP> opts(pjrt::execution_options(std::vector<int64_t>(), 0));
  try {
    pjrt::execute(executable, bad_inputs, opts);
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
  return "";
}

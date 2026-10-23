// Device-to-host transfers, materialized as R vectors.
//
// as_array(x, await = FALSE) returns a *host array*: an ALTREP vector (double,
// integer or logical) that stands for the result of a transfer still in flight.
// It carries its dim and class attributes from the start and reads as an
// ordinary R array; the first access to its data waits for the transfer and
// converts the bytes into a plain R vector, which the host array then wraps.
// For a dtype whose elements are already R's own (f64, i32, i64, ui64), PJRT
// writes straight into that vector and there is nothing to convert.
// That first access goes through pjrt's R function host_array_materialize(),
// which checks the values as as_array() does before the host array takes them.
//
// That first access may come from any code that reads the vector's data,
// including C++ code. A failed transfer then raises an R error from inside it
// (see call_pjrt()), so C++ that reads arrays which may be host arrays should
// await() them first.

#include <Rcpp.h>
// Rcpp.h first: the R headers below need its definition of SEXP.
#include <R_ext/Altrep.h>
#include <R_ext/Rdynload.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "bfloat16.h"
#include "buffer.h"
#include "buffer_sexp.h"
#include "event.h"
#include "half/half.hpp"
#include "utils.h"

namespace {

// A device-to-host transfer: where PJRT copies the buffer to (in column-major
// order), and the event that fires once it is done. That is `values`, the
// result vector itself, for a dtype R stores bit for bit (see is_r_native()),
// and `bytes`, to be converted, for any other.
struct HostTransfer {
  std::vector<uint8_t> bytes;
  // Rooted for the transfer's lifetime, not merely referenced: PJRT writes
  // into it until the event fires, even if nothing else refers to it.
  Rcpp::RObject values;
  std::unique_ptr<rpjrt::PJRTEvent> event;  // null once awaited, or if empty
  PJRT_Buffer_Type type;
  std::vector<int64_t> dims;
  std::string check;  // as_array()'s `check` level

  HostTransfer() = default;
  HostTransfer(const HostTransfer&) = delete;
  HostTransfer& operator=(const HostTransfer&) = delete;

  // PJRT writes into `bytes` or `values` until the event fires, so they must
  // outlive it. The body runs before the members are destroyed.
  ~HostTransfer() {
    if (event == nullptr) return;
    try {
      event->await();
    } catch (...) {
    }
  }
};

// 64-bit integers, and ui32 (which has no headroom in R's signed int32), are
// returned as bit64::integer64: a double vector carrying the int64 bit pattern.
bool is_integer64(PJRT_Buffer_Type type) {
  return type == PJRT_Buffer_Type_S64 || type == PJRT_Buffer_Type_U32 ||
         type == PJRT_Buffer_Type_U64;
}

SEXPTYPE r_type(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_S16:
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U8:
    case PJRT_Buffer_Type_U16:
      return INTSXP;
    case PJRT_Buffer_Type_PRED:
      return LGLSXP;
    default:
      return REALSXP;
  }
}

// Whether the dtype's elements are exactly what R stores for it, so that PJRT
// can write the result vector itself: f64 and i32 as they are, and i64 and
// ui64 as the int64 bit patterns of a bit64::integer64.
bool is_r_native(PJRT_Buffer_Type type) {
  return type == PJRT_Buffer_Type_F64 || type == PJRT_Buffer_Type_S32 ||
         type == PJRT_Buffer_Type_S64 || type == PJRT_Buffer_Type_U64;
}

void* vector_dataptr(SEXP x) {
  switch (TYPEOF(x)) {
    case REALSXP:
      return REAL(x);
    case INTSXP:
      return INTEGER(x);
    default:
      return LOGICAL(x);
  }
}

std::unique_ptr<HostTransfer> start_transfer(SEXP buffer_sexp) {
  auto buffer = rpjrt::as_buffer(buffer_sexp);
  auto t = std::make_unique<HostTransfer>();
  t->type = buffer->element_type();
  t->dims = buffer->dimensions();
  const auto numel = number_of_elements(t->dims);
  if (numel > 0) {
    const size_t nbytes = numel * sizeof_pjrt_buffer_type(t->type);
    uint8_t* dst;
    if (is_r_native(t->type)) {
      t->values = Rf_allocVector(r_type(t->type), numel);
      dst = static_cast<uint8_t*>(vector_dataptr(t->values));
    } else {
      t->bytes.resize(nbytes);
      dst = t->bytes.data();
    }
    std::span<uint8_t> host_buffer(dst, nbytes);
    t->event = buffer->buffer_to_host_async(host_buffer, false);
  }
  return t;
}

// Casts each element of the transfer's column-major bytes to `out`'s type.
template <typename T>
void convert(const HostTransfer& t, SEXP out) {
  const T* src = reinterpret_cast<const T*>(t.bytes.data());
  const R_xlen_t n = XLENGTH(out);
  switch (TYPEOF(out)) {
    case REALSXP:
      if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                    std::is_same_v<T, uint32_t>) {
        static_assert(sizeof(double) == sizeof(int64_t),
                      "bit64::integer64 layout requires sizeof(double) == "
                      "sizeof(int64_t)");
        std::copy(src, src + n, reinterpret_cast<int64_t*>(REAL(out)));
      } else {
        std::copy(src, src + n, REAL(out));
      }
      break;
    case INTSXP:
      std::copy(src, src + n, INTEGER(out));
      break;
    default:
      std::copy(src, src + n, LOGICAL(out));
  }
}

// Waits for the transfer and returns its values as a plain R vector, without
// attributes: the vector PJRT wrote, or one converted from its bytes.
SEXP to_vector(HostTransfer& t) {
  if (t.event != nullptr) {
    t.event->await();
    t.event->check_error();
    t.event.reset();
  }
  if (!t.values.isNULL()) return t.values;
  Rcpp::Shield<SEXP> out(
      Rf_allocVector(r_type(t.type), number_of_elements(t.dims)));
  if (XLENGTH(out) == 0) return out;
  switch (t.type) {
    case PJRT_Buffer_Type_F16:
      // Every binary16 value is exactly representable as a double.
      convert<half_float::half>(t, out);
      break;
    case PJRT_Buffer_Type_BF16:
      // Widening bf16 -> double is exact.
      convert<rpjrt::bfloat16>(t, out);
      break;
    case PJRT_Buffer_Type_F32:
      convert<float>(t, out);
      break;
    case PJRT_Buffer_Type_F64:
      convert<double>(t, out);
      break;
    case PJRT_Buffer_Type_S8:
      convert<int8_t>(t, out);
      break;
    case PJRT_Buffer_Type_S16:
      convert<int16_t>(t, out);
      break;
    case PJRT_Buffer_Type_S32:
      convert<int32_t>(t, out);
      break;
    case PJRT_Buffer_Type_S64:
      convert<int64_t>(t, out);
      break;
    case PJRT_Buffer_Type_U8:
    case PJRT_Buffer_Type_PRED:
      convert<uint8_t>(t, out);
      break;
    case PJRT_Buffer_Type_U16:
      convert<uint16_t>(t, out);
      break;
    case PJRT_Buffer_Type_U32:
      convert<uint32_t>(t, out);
      break;
    case PJRT_Buffer_Type_U64:
      convert<uint64_t>(t, out);
      break;
    default:
      Rcpp::stop("Unsupported buffer element type for conversion to host.");
  }
  return out;
}

void set_array_attributes(SEXP x, const HostTransfer& t) {
  if (!t.dims.empty()) {
    Rcpp::IntegerVector dim(t.dims.begin(), t.dims.end());
    Rf_setAttrib(x, R_DimSymbol, dim);
  }
  if (is_integer64(t.type)) {
    Rf_setAttrib(x, R_ClassSymbol, Rf_mkString("integer64"));
  }
}

// A host array's data1 is an external pointer to its HostTransfer, and its
// data2 the materialized vector. The external pointer protects the source
// buffer, which PJRT may still be reading from: on CPU, a buffer made from R
// data aliases memory that only the buffer object keeps alive. Materializing
// sets data2 and drops data1, freeing the transfer's bytes and releasing the
// buffer.
R_altrep_class_t real_class;
R_altrep_class_t integer_class;
R_altrep_class_t logical_class;

bool is_host_array(SEXP x) {
  return R_altrep_inherits(x, real_class) ||
         R_altrep_inherits(x, integer_class) ||
         R_altrep_inherits(x, logical_class);
}

bool is_materialized(SEXP x) { return R_altrep_data2(x) != R_NilValue; }

HostTransfer& transfer(SEXP x) {
  return *static_cast<HostTransfer*>(R_ExternalPtrAddr(R_altrep_data1(x)));
}

R_xlen_t host_array_length(SEXP x) {
  if (is_materialized(x)) return XLENGTH(R_altrep_data2(x));
  return number_of_elements(transfer(x).dims);
}

void* host_array_dataptr(SEXP x, Rboolean) {
  if (!is_materialized(x)) {
    rpjrt::call_pjrt("host_array_materialize", x);
  }
  return vector_dataptr(R_altrep_data2(x));
}

const void* host_array_dataptr_or_null(SEXP x) {
  return is_materialized(x) ? vector_dataptr(R_altrep_data2(x)) : nullptr;
}

Rboolean host_array_inspect(SEXP x, int, int, int,
                            void (*)(SEXP, int, int, int)) {
  Rprintf(" PJRTHostArray (%s)\n",
          is_materialized(x) ? "materialized" : "pending");
  return TRUE;
}

void register_methods(R_altrep_class_t cls) {
  R_set_altrep_Length_method(cls, host_array_length);
  R_set_altrep_Inspect_method(cls, host_array_inspect);
  R_set_altvec_Dataptr_method(cls, host_array_dataptr);
  R_set_altvec_Dataptr_or_null_method(cls, host_array_dataptr_or_null);
}

}  // namespace

// [[Rcpp::init]]
void register_host_array_altrep(DllInfo* dll) {
  real_class = R_make_altreal_class("PJRTHostArrayReal", "pjrt", dll);
  integer_class = R_make_altinteger_class("PJRTHostArrayInteger", "pjrt", dll);
  logical_class = R_make_altlogical_class("PJRTHostArrayLogical", "pjrt", dll);
  register_methods(real_class);
  register_methods(integer_class);
  register_methods(logical_class);
}

// [[Rcpp::export()]]
SEXP impl_buffer_to_host(SEXP buffer_sexp) {
  auto t = start_transfer(buffer_sexp);
  Rcpp::Shield<SEXP> out(to_vector(*t));
  set_array_attributes(out, *t);
  return out;
}

// [[Rcpp::export()]]
SEXP impl_buffer_to_host_async(SEXP buffer_sexp, std::string check) {
  auto t = start_transfer(buffer_sexp);
  t->check = std::move(check);
  R_altrep_class_t cls;
  switch (r_type(t->type)) {
    case REALSXP:
      cls = real_class;
      break;
    case INTSXP:
      cls = integer_class;
      break;
    default:
      cls = logical_class;
  }
  const HostTransfer& ref = *t;
  Rcpp::XPtr<HostTransfer> xptr(t.release(), true, R_NilValue, buffer_sexp);
  Rcpp::Shield<SEXP> out(R_new_altrep(cls, xptr, R_NilValue));
  set_array_attributes(out, ref);
  return out;
}

// For a host array `x` that is not materialized yet: waits for its transfer
// and returns its values (as an R array), dtype and `check` level. NULL for
// any other object.
// [[Rcpp::export()]]
SEXP impl_host_array_pending(SEXP x) {
  if (!is_host_array(x) || is_materialized(x)) return R_NilValue;
  const HostTransfer& t = transfer(x);
  Rcpp::Shield<SEXP> values(to_vector(transfer(x)));
  set_array_attributes(values, t);
  return Rcpp::List::create(
      Rcpp::Named("values") = values,
      Rcpp::Named("dtype") = rpjrt::PJRTElementType(t.type).as_string(),
      Rcpp::Named("check") = t.check);
}

// Makes the host array `x` wrap `values`, from impl_host_array_pending(x).
// [[Rcpp::export()]]
void impl_host_array_set(SEXP x, SEXP values) {
  R_set_altrep_data2(x, values);
  Rcpp::XPtr<HostTransfer>(R_altrep_data1(x)).release();
  R_set_altrep_data1(x, R_NilValue);
}

// Whether the host array `x`'s transfer is done; TRUE for any other object.
// [[Rcpp::export()]]
bool impl_host_array_is_ready(SEXP x) {
  if (!is_host_array(x) || is_materialized(x)) return true;
  const auto& event = transfer(x).event;
  return event == nullptr || event->is_ready();
}

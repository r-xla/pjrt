// Device-to-host transfers, materialized as R vectors.
//
// as_array_async() returns a *host array*: an ALTREP vector (double, integer
// or logical) that stands for the result of a transfer still in flight. It
// carries its dim and class attributes from the start and reads as an ordinary
// R array; the first access to its data waits for the transfer and converts
// the bytes into a plain R vector, which the host array then wraps.
//
// That first access may come from any code that reads the vector's data,
// including C++ code. A failed transfer then raises an R error from inside it
// (see call_pjrt()), so C++ that reads arrays which may be host arrays should
// await() them first.

#include <Rcpp.h>
// Rcpp.h first: the R headers below need its definition of SEXP.
#include <R_ext/Altrep.h>
#include <R_ext/Rdynload.h>

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "bfloat16.h"
#include "buffer.h"
#include "buffer_sexp.h"
#include "event.h"
#include "utils.h"

namespace {

// A device-to-host transfer: the bytes PJRT copies the buffer into (in the
// device layout), and the event that fires once it is done.
struct HostTransfer {
  std::vector<uint8_t> bytes;
  std::unique_ptr<rpjrt::PJRTEvent> event;  // null once awaited, or if empty
  PJRT_Buffer_Type type;
  std::vector<int64_t> dims;
  std::vector<int64_t> minor_to_major;

  HostTransfer() = default;
  HostTransfer(const HostTransfer&) = delete;
  HostTransfer& operator=(const HostTransfer&) = delete;

  // PJRT writes into `bytes` until the event fires, so they must outlive it.
  ~HostTransfer() {
    if (event == nullptr) return;
    try {
      event->await();
    } catch (...) {
    }
  }
};

std::unique_ptr<HostTransfer> start_transfer(SEXP buffer_sexp) {
  auto buffer = rpjrt::as_buffer(buffer_sexp);
  auto t = std::make_unique<HostTransfer>();
  t->type = buffer->element_type();
  t->dims = buffer->dimensions();
  t->minor_to_major = buffer->minor_to_major();
  const auto numel = number_of_elements(t->dims);
  if (numel > 0) {
    t->bytes.resize(numel * sizeof_pjrt_buffer_type(t->type));
    std::span<uint8_t> host_buffer(t->bytes);
    t->event = buffer->buffer_to_host_async(host_buffer);
  }
  return t;
}

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

// Reorders the transfer's device-layout bytes into the column-major `out`,
// casting each element to `out`'s type.
template <typename T>
void convert(const HostTransfer& t, SEXP out) {
  const T* src = reinterpret_cast<const T*>(t.bytes.data());
  std::vector<T> row_major(XLENGTH(out));
  device_to_row_major<T>(src, row_major.data(), t.dims, t.minor_to_major);
  switch (TYPEOF(out)) {
    case REALSXP:
      if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                    std::is_same_v<T, uint32_t>) {
        static_assert(sizeof(double) == sizeof(int64_t),
                      "bit64::integer64 layout requires sizeof(double) == "
                      "sizeof(int64_t)");
        row_to_col_order<T, int64_t>(
            row_major, reinterpret_cast<int64_t*>(REAL(out)), t.dims);
      } else {
        row_to_col_order<T, double>(row_major, REAL(out), t.dims);
      }
      break;
    case INTSXP:
      row_to_col_order<T, int>(row_major, INTEGER(out), t.dims);
      break;
    default:
      row_to_col_order<T, int>(row_major, LOGICAL(out), t.dims);
  }
}

// Waits for the transfer and converts its bytes into a plain R vector, without
// attributes.
SEXP to_vector(HostTransfer& t) {
  if (t.event != nullptr) {
    t.event->await();
    t.event->check_error();
    t.event.reset();
  }
  Rcpp::Shield<SEXP> out(
      Rf_allocVector(r_type(t.type), number_of_elements(t.dims)));
  if (XLENGTH(out) == 0) return out;
  switch (t.type) {
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

R_xlen_t host_array_length(SEXP x) {
  if (is_materialized(x)) return XLENGTH(R_altrep_data2(x));
  return number_of_elements(transfer(x).dims);
}

void* host_array_dataptr(SEXP x, Rboolean) {
  if (!is_materialized(x)) {
    rpjrt::call_pjrt("impl_host_array_materialize", x);
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
SEXP impl_buffer_to_host_async(SEXP buffer_sexp) {
  auto t = start_transfer(buffer_sexp);
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

// Materializes the host array `x`; a no-op for any other object.
// [[Rcpp::export()]]
void impl_host_array_materialize(SEXP x) {
  if (!is_host_array(x) || is_materialized(x)) return;
  R_set_altrep_data2(x, to_vector(transfer(x)));
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

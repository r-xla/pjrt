#include "buffer_sexp.h"

#include <R_ext/Altrep.h>
#include <R_ext/Rdynload.h>

namespace {

R_altrep_class_t carrier_class;
SEXP buffer_class_attr = R_NilValue;

// `xptr` is the buffer's external pointer, or R_NilValue for a buffer that was
// serialized without data.
SEXP new_carrier(SEXP xptr, SEXP keepalive) {
  return R_new_altrep(carrier_class, xptr, keepalive);
}

bool is_carrier(SEXP x) { return R_altrep_inherits(x, carrier_class); }

// The carrier reads as its keepalive's bytes, so that object.size() -- which
// sizes a raw vector by its length -- counts a CPU buffer's memory.
R_xlen_t carrier_length(SEXP x) {
  SEXP keepalive = R_altrep_data2(x);
  return keepalive == R_NilValue ? 0 : XLENGTH(keepalive);
}

void* carrier_dataptr(SEXP x, Rboolean) {
  static Rbyte empty;
  SEXP keepalive = R_altrep_data2(x);
  return keepalive == R_NilValue ? &empty : RAW(keepalive);
}

const void* carrier_dataptr_or_null(SEXP x) {
  return carrier_dataptr(x, FALSE);
}

SEXP carrier_duplicate(SEXP x, Rboolean) { return x; }

// A donated buffer has no data to write; it serializes as NULL and loads as a
// buffer without data.
SEXP carrier_serialized_state(SEXP x) {
  SEXP xptr = R_altrep_data1(x);
  if (xptr == R_NilValue) return R_NilValue;
  auto* buf = static_cast<rpjrt::PJRTBuffer*>(R_ExternalPtrAddr(xptr));
  if (buf == nullptr || buf->buffer == nullptr) return R_NilValue;
  return rpjrt::call_pjrt("buffer_serialized_state", xptr);
}

SEXP carrier_unserialize(SEXP, SEXP state) {
  if (state == R_NilValue) return new_carrier(R_NilValue, R_NilValue);
  SEXP buf = PROTECT(rpjrt::call_pjrt("buffer_unserialize", state));
  SEXP carrier = R_ExternalPtrProtected(buf);
  UNPROTECT(1);
  return carrier;
}

// Moves the buffer that unserialization uploaded into the buffer object `x`
// that R read back: `x` takes over its address, finalizer and carrier, and the
// pointer that held it is left empty.
void adopt_unserialized(SEXP x) {
  SEXP carrier = R_ExternalPtrProtected(x);
  SEXP src = is_carrier(carrier) ? R_altrep_data1(carrier) : R_NilValue;
  if (src == R_NilValue || R_ExternalPtrAddr(src) == nullptr) {
    Rcpp::stop(
        "The PJRTBuffer holds no data: it was serialized after being donated, "
        "or with serialization format version 2");
  }
  R_SetExternalPtrAddr(x, R_ExternalPtrAddr(src));
  R_ClearExternalPtr(src);
  R_SetExternalPtrProtected(src, R_NilValue);
  R_set_altrep_data1(carrier, x);
  Rcpp::XPtr<rpjrt::PJRTBuffer>(x).setDeleteFinalizer();
}

}  // namespace

namespace rpjrt {

SEXP call_pjrt(const char* fun, SEXP arg) {
  SEXP ns = PROTECT(R_FindNamespace(PROTECT(Rf_mkString("pjrt"))));
  SEXP call = PROTECT(Rf_lang2(Rf_install(fun), arg));
  SEXP out = Rf_eval(call, ns);
  UNPROTECT(3);
  return out;
}

}  // namespace rpjrt

// [[Rcpp::init]]
void register_buffer_altrep(DllInfo* dll) {
  carrier_class = R_make_altraw_class("PJRTBufferCarrier", "pjrt", dll);
  R_set_altrep_Length_method(carrier_class, carrier_length);
  R_set_altrep_Duplicate_method(carrier_class, carrier_duplicate);
  R_set_altrep_Serialized_state_method(carrier_class, carrier_serialized_state);
  R_set_altrep_Unserialize_method(carrier_class, carrier_unserialize);
  R_set_altvec_Dataptr_method(carrier_class, carrier_dataptr);
  R_set_altvec_Dataptr_or_null_method(carrier_class, carrier_dataptr_or_null);

  buffer_class_attr = Rf_mkString("PJRTBuffer");
  R_PreserveObject(buffer_class_attr);
  MARK_NOT_MUTABLE(buffer_class_attr);
}

namespace rpjrt {

SEXP wrap_buffer(std::unique_ptr<PJRTBuffer> buffer, SEXP keepalive) {
  Rcpp::XPtr<PJRTBuffer> xptr(buffer.release(), true);
  R_SetExternalPtrProtected(xptr, new_carrier(xptr, keepalive));
  Rf_setAttrib(xptr, R_ClassSymbol, buffer_class_attr);
  return xptr;
}

bool is_buffer(SEXP x) {
  return TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "PJRTBuffer");
}

SEXP buffer_xptr(SEXP x) {
  if (!is_buffer(x)) {
    Rcpp::stop("Expected a PJRTBuffer");
  }
  if (R_ExternalPtrAddr(x) == nullptr) {
    adopt_unserialized(x);
  }
  return x;
}

Rcpp::XPtr<PJRTBuffer> as_buffer(SEXP x) {
  return Rcpp::XPtr<PJRTBuffer>(buffer_xptr(x));
}

SEXP buffer_keepalive(SEXP x) {
  SEXP carrier = R_ExternalPtrProtected(x);
  return is_carrier(carrier) ? R_altrep_data2(carrier) : R_NilValue;
}

void set_buffer_keepalive(SEXP x, SEXP keepalive) {
  R_set_altrep_data2(R_ExternalPtrProtected(x), keepalive);
}

}  // namespace rpjrt

#include "buffer_sexp.h"

#include <R_ext/Altrep.h>
#include <R_ext/Rdynload.h>

namespace {

R_altrep_class_t buffer_class;
SEXP buffer_class_attr = R_NilValue;

// Evaluates `fun(arg)` in pjrt's namespace. The ALTREP methods below run
// inside R's (un)serializer, so they use the C API only and an R error
// longjmps straight through them.
SEXP call_pjrt(const char* fun, SEXP arg) {
  SEXP ns = PROTECT(R_FindNamespace(PROTECT(Rf_mkString("pjrt"))));
  SEXP call = PROTECT(Rf_lang2(Rf_install(fun), arg));
  SEXP out = Rf_eval(call, ns);
  UNPROTECT(3);
  return out;
}

R_xlen_t buffer_length(SEXP) { return 1; }

SEXP buffer_elt(SEXP x, R_xlen_t) { return R_altrep_data1(x); }

void buffer_set_elt(SEXP, R_xlen_t, SEXP) {
  Rf_error("A PJRTBuffer can't be modified");
}

// Buffers are immutable, so a duplicate -- deep or not -- shares the external
// pointer. The default DuplicateEX copies the attributes.
SEXP buffer_duplicate(SEXP x, Rboolean) {
  return R_new_altrep(buffer_class, R_altrep_data1(x), R_NilValue);
}

SEXP buffer_serialized_state(SEXP x) {
  return call_pjrt("buffer_serialized_state", x);
}

SEXP buffer_unserialize(SEXP, SEXP state) {
  return call_pjrt("buffer_unserialize", state);
}

}  // namespace

// [[Rcpp::init]]
void register_buffer_altrep(DllInfo* dll) {
  buffer_class = R_make_altlist_class("PJRTBuffer", "pjrt", dll);
  R_set_altrep_Length_method(buffer_class, buffer_length);
  R_set_altrep_Duplicate_method(buffer_class, buffer_duplicate);
  R_set_altrep_Serialized_state_method(buffer_class, buffer_serialized_state);
  R_set_altrep_Unserialize_method(buffer_class, buffer_unserialize);
  R_set_altlist_Elt_method(buffer_class, buffer_elt);
  R_set_altlist_Set_elt_method(buffer_class, buffer_set_elt);

  buffer_class_attr = Rf_mkString("PJRTBuffer");
  R_PreserveObject(buffer_class_attr);
  MARK_NOT_MUTABLE(buffer_class_attr);
}

namespace rpjrt {

SEXP wrap_buffer(std::unique_ptr<PJRTBuffer> buffer, SEXP keepalive) {
  Rcpp::XPtr<PJRTBuffer> xptr(buffer.release(), true, R_NilValue, keepalive);
  SEXP out = PROTECT(R_new_altrep(buffer_class, xptr, R_NilValue));
  Rf_setAttrib(out, R_ClassSymbol, buffer_class_attr);
  UNPROTECT(1);
  return out;
}

bool is_buffer(SEXP x) { return R_altrep_inherits(x, buffer_class); }

SEXP buffer_xptr(SEXP x) {
  if (!is_buffer(x)) {
    Rcpp::stop("Expected a PJRTBuffer");
  }
  return R_altrep_data1(x);
}

Rcpp::XPtr<PJRTBuffer> as_buffer(SEXP x) {
  return Rcpp::XPtr<PJRTBuffer>(buffer_xptr(x));
}

}  // namespace rpjrt

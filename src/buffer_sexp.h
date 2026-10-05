#pragma once

#include <Rcpp.h>

#include <memory>

#include "buffer.h"

namespace rpjrt {

// The R representation of a PJRTBuffer: a length-1 ALTLIST of class
// "PJRTBuffer" whose only element is the external pointer that owns the
// buffer (its finalizer deletes it; its protected slot holds the CPU
// keepalive). An external pointer cannot survive serialization, but an ALTREP
// object can provide its own: the ALTREP class serializes a buffer as its
// host bytes and re-uploads them on unserialization, so saveRDS() and
// friends round-trip buffers.
//
// A length-1 list rather than a bare external pointer wrapper keeps the
// handle visible to base R: identical() compares the external pointers, and a
// copy that drops the ALTREP class (anything bypassing Duplicate) still holds
// the pointer rather than nothing.

// Wraps `buffer` (taking ownership) in a new R buffer object. `keepalive` goes
// into the external pointer's protected slot.
SEXP wrap_buffer(std::unique_ptr<PJRTBuffer> buffer,
                 SEXP keepalive = R_NilValue);

// Whether `x` is a buffer object created by wrap_buffer().
bool is_buffer(SEXP x);

// The external pointer inside a buffer object; errors if `x` is not one.
SEXP buffer_xptr(SEXP x);

// The PJRTBuffer behind a buffer object; errors if `x` is not one.
Rcpp::XPtr<PJRTBuffer> as_buffer(SEXP x);

}  // namespace rpjrt

#pragma once

#include <Rcpp.h>

#include <memory>

#include "buffer.h"

namespace rpjrt {

// The R representation of a PJRTBuffer: an external pointer of class
// "PJRTBuffer" that owns the buffer (its finalizer deletes it). Its protected
// slot holds the buffer's *carrier*, an ALTRAW object pointing back at the
// external pointer and holding the CPU keepalive (the RAWSXP whose bytes a
// zero-copy buffer aliases); it reads as the keepalive's bytes.
//
// The carrier gives the buffer its serialization. R writes an external
// pointer's protected slot (but not its address) and puts the pointer in its
// reference table, so the carrier's ALTREP methods write the buffer's host
// bytes once per serialize() call, however often the buffer is referenced, and
// on unserialization upload them into a fresh buffer that every reference
// shares. The pointer read back has a NULL address; buffer_xptr() moves the
// fresh buffer into it on first use.
//
// Never read or write the protected slot directly: use buffer_keepalive() and
// set_buffer_keepalive().

// Wraps `buffer` (taking ownership) in a new R buffer object, keeping
// `keepalive` alive with it.
SEXP wrap_buffer(std::unique_ptr<PJRTBuffer> buffer,
                 SEXP keepalive = R_NilValue);

// Whether `x` is a buffer object.
bool is_buffer(SEXP x);

// The external pointer of a buffer object -- `x` itself, after moving an
// unserialized buffer into it; errors if `x` is not a buffer object or holds no
// buffer.
SEXP buffer_xptr(SEXP x);

// The PJRTBuffer behind a buffer object; errors as buffer_xptr() does.
Rcpp::XPtr<PJRTBuffer> as_buffer(SEXP x);

// The keepalive of the buffer object `x`; R_NilValue if it has none.
SEXP buffer_keepalive(SEXP x);

void set_buffer_keepalive(SEXP x, SEXP keepalive);

}  // namespace rpjrt

#pragma once

#include <Rcpp.h>

#include <cstddef>

namespace rpjrt {

// Pin `obj` against garbage collection until its release is processed, and
// return the token to hand to queue_release(). Main R thread only. The pin
// lives on Rcpp's precious list, whose removal is O(1).
SEXP preserve(SEXP obj);

// Unpin a token from preserve() right away. Main R thread only.
void release(SEXP token);

// Queue a token from preserve() for deferred release. This is thread-safe and
// can be called from any thread (e.g., PJRT callback threads). The actual
// release happens later on the main R thread.
void queue_release(SEXP token);

// Process any pending releases. This MUST be called from the main R thread.
// It is safe to call this function even if there are no pending releases.
void process_pending_releases();

// Number of objects currently waiting in the release queue. Exposed for
// tests/tooling to observe that the queue drains.
std::size_t pending_release_count();

}  // namespace rpjrt

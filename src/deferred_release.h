#pragma once

#include <cstddef>

#include <Rcpp.h>

namespace rpjrt {

// Queue an R object for deferred release. This is thread-safe and can be
// called from any thread (e.g., PJRT callback threads).
// The actual R_ReleaseObject call will happen later on the main R thread.
void queue_release(SEXP obj);

// Process any pending releases. This MUST be called from the main R thread.
// It is safe to call this function even if there are no pending releases.
void process_pending_releases();

// Number of objects currently waiting in the release queue. Exposed for
// tests/tooling to observe that the queue drains.
std::size_t pending_release_count();

}  // namespace rpjrt

// Include Rcpp first to avoid conflicts with R macros
#include "deferred_release.h"

#include <Rcpp.h>

#include <mutex>
#include <vector>

namespace rpjrt {

// Thread-safe queue of preserve() tokens pending release
static std::mutex g_release_mutex;
static std::vector<SEXP> g_pending_releases;

SEXP preserve(SEXP obj) { return Rcpp::Rcpp_precious_preserve(obj); }

void release(SEXP token) { Rcpp::Rcpp_precious_remove(token); }

void queue_release(SEXP token) {
  std::lock_guard<std::mutex> lock(g_release_mutex);
  g_pending_releases.push_back(token);
}

void process_pending_releases() {
  // Swap out the pending list under the lock to minimize lock contention
  std::vector<SEXP> to_release;
  {
    std::lock_guard<std::mutex> lock(g_release_mutex);
    to_release.swap(g_pending_releases);
  }

  // Now release all tokens on the main R thread (no lock held)
  for (SEXP token : to_release) release(token);
}

std::size_t pending_release_count() {
  std::lock_guard<std::mutex> lock(g_release_mutex);
  return g_pending_releases.size();
}

}  // namespace rpjrt

// Include Rcpp first to avoid conflicts with R macros
#include <Rcpp.h>

#include <mutex>
#include <vector>

#include "deferred_release.h"

namespace rpjrt {

// Thread-safe queue for R objects pending release
static std::mutex g_release_mutex;
static std::vector<SEXP> g_pending_releases;

void queue_release(SEXP obj) {
  std::lock_guard<std::mutex> lock(g_release_mutex);
  g_pending_releases.push_back(obj);
}

void process_pending_releases() {
  // Swap out the pending list under the lock to minimize lock contention
  std::vector<SEXP> to_release;
  {
    std::lock_guard<std::mutex> lock(g_release_mutex);
    to_release.swap(g_pending_releases);
  }

  // Now release all objects on the main R thread (no lock held)
  for (SEXP obj : to_release) {
    R_ReleaseObject(obj);
  }
}

}  // namespace rpjrt

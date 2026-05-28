#include "gc.h"

#include <Rcpp.h>

#include <atomic>

#include "deferred_release.h"

namespace rpjrt {

static std::atomic<std::size_t> g_gc_call_count{0};

void call_r_gc() {
  Rcpp::Function r_gc("gc");
  r_gc(Rcpp::Named("full") = true, Rcpp::Named("verbose") = false);
  R_RunPendingFinalizers();
  process_pending_releases();
  g_gc_call_count.fetch_add(1, std::memory_order_relaxed);
}

std::size_t gc_call_count() {
  return g_gc_call_count.load(std::memory_order_relaxed);
}

}  // namespace rpjrt

// [[Rcpp::export]]
std::size_t impl_gc_call_count() { return rpjrt::gc_call_count(); }

// [[Rcpp::export]]
void impl_call_r_gc() { rpjrt::call_r_gc(); }

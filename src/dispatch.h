// The Dispatcher: the per-jit object an R-side `Dispatcher` external pointer
// points at. Declared here (rather than in dispatch.cpp) so the type is visible
// to Rcpp attributes, which lets the exported entry points take and return
// Rcpp::XPtr<Dispatcher> directly -- the same way every other pjrt handle
// (PJRTClient, PJRTBuffer, ...) is declared in its own header.
//
// The dispatch logic itself lives in dispatch.cpp.

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "dispatch_engine.h"
#include "dispatch_key.h"
#include "lru_cache.h"

namespace rpjrt {

// Per-jit Dispatcher: the engine, the executable cache, the R compile callback
// (invoked on a miss), and the dispatch policies. Every R object it holds --
// the callbacks, and the cache entries' own -- is held in an Rcpp type, so it
// is rooted for as long as the dispatcher (or the entry) lives and dropped when
// that ends. There is nothing to release by hand.
class Dispatcher {
 public:
  Dispatcher(std::size_t capacity, SEXP miss_fn,
             std::unordered_set<std::string> static_names,
             std::unique_ptr<Engine> engine, std::string backend,
             bool move_inputs, std::optional<Rcpp::Function> default_device_fn)
      : cache_(capacity),
        miss_fn_(miss_fn),
        default_device_fn_(std::move(default_device_fn)),
        static_names_(std::move(static_names)),
        engine_(std::move(engine)),
        backend_(std::move(backend)),
        move_inputs_(move_inputs) {}

  // Holds R objects and an engine on behalf of one R-side external pointer.
  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  // The backend's current default device, as an R object -- the device a call
  // with no array leaves runs on. Resolved afresh per such call: the default
  // can change mid-session, and an entry compiled under one must not serve
  // another. Nullopt when the dispatcher was given no resolver.
  const std::optional<Rcpp::Function>& default_device_fn() const {
    return default_device_fn_;
  }

  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq>& cache() {
    return cache_;
  }
  const Rcpp::Function& miss_fn() const { return miss_fn_; }
  const std::unordered_set<std::string>& static_names() const {
    return static_names_;
  }
  // Non-const: canonical_device() may grow the engine's device table.
  Engine& engine() { return *engine_; }
  // The `$backend` tag every AnvlArray input must carry.
  const std::string& backend() const { return backend_; }
  // Pin policy: a target device is fixed per entry (jit(device = ) /
  // device_arg), so the key carries no device and the engine places the
  // inputs on the entry's device at execute time. The engine holds this too
  // -- the core needs it for the key, the engine for the placing.
  bool move_inputs() const { return move_inputs_; }

 private:
  LRUCache<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq> cache_;
  Rcpp::Function miss_fn_;
  std::optional<Rcpp::Function> default_device_fn_;
  std::unordered_set<std::string> static_names_;
  std::unique_ptr<Engine> engine_;
  std::string backend_;
  bool move_inputs_;
};

}  // namespace rpjrt

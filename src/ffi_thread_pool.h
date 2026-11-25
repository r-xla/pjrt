// Spreading the matrices of a batch over XLA's intra-op thread pool, which an
// FFI handler on the host binds as `.Ctx<xla::ffi::ThreadPool>()`.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace rpjrt {

namespace detail {

// What the threads of one parallel_for() share. Tasks hold it through a
// shared_ptr, so a task that starts only after parallel_for() returned still
// finds it; `body` is reached only by a thread that took a chunk, which
// parallel_for() waits for.
template <typename Body>
struct ParallelFor {
  ParallelFor(const Body &body, std::int64_t size, std::int64_t chunk)
      : body(body),
        size(size),
        chunk(chunk),
        n_chunks((size + chunk - 1) / chunk),
        pending(n_chunks) {}

  const Body &body;
  const std::int64_t size;
  const std::int64_t chunk;
  const std::int64_t n_chunks;
  std::atomic<std::int64_t> next{0};
  // The chunks that are not done yet.
  std::atomic<std::int64_t> pending;
  std::atomic<bool> failed{false};
  std::mutex mu;
  std::condition_variable done;
  // The first error a chunk returned, guarded by `mu`.
  xla::ffi::Error error;

  // Take chunks until there are none left.
  void work() {
    for (;;) {
      std::int64_t c = next.fetch_add(1);
      if (c >= n_chunks) return;
      // After an error, the remaining chunks are only counted down.
      if (!failed.load()) {
        std::int64_t begin = c * chunk;
        xla::ffi::Error e = run(begin, std::min(begin + chunk, size));
        if (e.failure()) {
          std::lock_guard<std::mutex> lock(mu);
          if (!failed.load()) {
            error = std::move(e);
            failed.store(true);
          }
        }
      }
      if (pending.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> lock(mu);
        done.notify_all();
      }
    }
  }

  // An exception must not escape a thread of XLA's pool.
  xla::ffi::Error run(std::int64_t begin, std::int64_t end) {
    try {
      return body(begin, end);
    } catch (const std::exception &e) {
      return xla::ffi::Error::Internal(e.what());
    } catch (...) {
      return xla::ffi::Error::Internal("unknown exception");
    }
  }
};

}  // namespace detail

// Call `body(begin, end)`, which returns an xla::ffi::Error, on chunks of
// consecutive indices that together cover [0, size) once, spread over at most
// `max_threads` threads of `pool`, and return the first error a chunk
// returned. After an error, chunks that have not started are skipped. A chunk
// can reuse its workspace for every index in it.
//
// XLA promises neither that a scheduled task runs in parallel nor that it runs
// before the handler returns: the handler may itself occupy the only thread of
// the pool. So the calling thread works through the chunks too, and waits only
// for those another thread has taken.
//
// `Pool` is xla::ffi::ThreadPool, or anything with its `Schedule()` and
// `num_threads()`.
template <typename Pool, typename Body>
xla::ffi::Error parallel_for(Pool &pool, std::int64_t size,
                             std::int64_t max_threads, const Body &body) {
  if (size <= 0) return xla::ffi::Error::Success();
  std::int64_t workers = std::max<std::int64_t>(
      std::min({pool.num_threads(), max_threads, size}), 1);
  // A few chunks per thread, so that one which starts late, or a matrix that
  // takes longer, delays the others less.
  std::int64_t n_chunks = std::min<std::int64_t>(size, workers * 4);
  std::int64_t chunk = (size + n_chunks - 1) / n_chunks;

  auto state = std::make_shared<detail::ParallelFor<Body>>(body, size, chunk);
  for (std::int64_t i = 1; i < workers; i++) {
    pool.Schedule([state] { state->work(); });
  }
  state->work();

  std::unique_lock<std::mutex> lock(state->mu);
  state->done.wait(lock, [&] { return state->pending.load() == 0; });
  return state->error;
}

}  // namespace rpjrt

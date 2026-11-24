#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "ffi_thread_pool.h"
#include "testthat/testthat.h"

namespace {

using xla::ffi::Error;

constexpr std::int64_t kAnyThreads = std::numeric_limits<std::int64_t>::max();

// Runs every task on a thread of its own, joined when the pool goes.
class ThreadedPool {
 public:
  explicit ThreadedPool(std::int64_t n) : n_(n) {}
  ~ThreadedPool() {
    for (auto &t : threads_) t.join();
  }
  std::int64_t num_threads() const { return n_; }
  template <typename F>
  void Schedule(F &&f) {
    threads_.emplace_back(std::forward<F>(f));
  }
  std::size_t n_tasks() const { return threads_.size(); }

 private:
  std::int64_t n_;
  std::vector<std::thread> threads_;
};

// Holds on to its tasks until run(): a pool whose threads are all busy.
class DeferredPool {
 public:
  std::int64_t num_threads() const { return 4; }
  template <typename F>
  void Schedule(F &&f) {
    tasks_.emplace_back(std::forward<F>(f));
  }
  void run() {
    for (auto &t : tasks_) t();
  }
  std::size_t n_tasks() const { return tasks_.size(); }

 private:
  std::vector<std::function<void()>> tasks_;
};

// Reports no threads, as xla::ffi::ThreadPool does when XLA gives the handler
// no pool.
class NoPool {
 public:
  std::int64_t num_threads() const { return 0; }
  template <typename F>
  void Schedule(F &&) {
    scheduled = true;
  }
  bool scheduled = false;
};

// Counts how often each index of [0, size) was visited.
struct Visits {
  explicit Visits(std::int64_t size) : counts(size) {}
  std::vector<std::atomic<int>> counts;
  std::atomic<int> chunks{0};

  Error visit(std::int64_t b0, std::int64_t b1) {
    chunks++;
    for (std::int64_t i = b0; i < b1; i++) counts[i]++;
    return Error::Success();
  }
  bool each_once() const {
    for (auto &c : counts) {
      if (c.load() != 1) return false;
    }
    return true;
  }
};

}  // namespace

context("parallel_for") {
  test_that("visits every index once, spread over the pool's threads") {
    Visits v(1000);
    ThreadedPool pool(8);
    Error e = rpjrt::parallel_for(
        pool, 1000, kAnyThreads,
        [&](std::int64_t b0, std::int64_t b1) { return v.visit(b0, b1); });
    expect_true(e.success());
    expect_true(v.each_once());
  }

  test_that("finishes on the calling thread when the pool runs nothing") {
    Visits v(100);
    DeferredPool pool;
    Error e = rpjrt::parallel_for(
        pool, 100, kAnyThreads,
        [&](std::int64_t b0, std::int64_t b1) { return v.visit(b0, b1); });
    expect_true(e.success());
    expect_true(v.each_once());
    expect_true(pool.n_tasks() == 3u);
    // The tasks that start late find no chunk left, and never call the body.
    int chunks = v.chunks.load();
    pool.run();
    expect_true(v.chunks.load() == chunks);
  }

  test_that("uses no more threads than max_threads") {
    Visits v(100);
    ThreadedPool pool(8);
    Error e = rpjrt::parallel_for(
        pool, 100, 3,
        [&](std::int64_t b0, std::int64_t b1) { return v.visit(b0, b1); });
    expect_true(e.success());
    expect_true(v.each_once());
    expect_true(pool.n_tasks() == 2u);

    Visits w(100);
    ThreadedPool alone(8);
    e = rpjrt::parallel_for(
        alone, 100, 1,
        [&](std::int64_t b0, std::int64_t b1) { return w.visit(b0, b1); });
    expect_true(e.success());
    expect_true(w.each_once());
    expect_true(alone.n_tasks() == 0u);
  }

  test_that("runs on the calling thread without a pool") {
    Visits v(10);
    NoPool pool;
    Error e = rpjrt::parallel_for(
        pool, 10, kAnyThreads,
        [&](std::int64_t b0, std::int64_t b1) { return v.visit(b0, b1); });
    expect_true(e.success());
    expect_true(v.each_once());
    expect_false(pool.scheduled);
  }

  test_that("calls nothing for an empty batch") {
    NoPool pool;
    bool called = false;
    Error e = rpjrt::parallel_for(pool, 0, kAnyThreads,
                                  [&](std::int64_t, std::int64_t) {
                                    called = true;
                                    return Error::Success();
                                  });
    expect_true(e.success());
    expect_false(called);
  }

  test_that("returns the first error and skips the chunks after it") {
    DeferredPool pool;
    int chunks = 0;
    Error e = rpjrt::parallel_for(pool, 100, kAnyThreads,
                                  [&](std::int64_t, std::int64_t) {
                                    chunks++;
                                    return Error::Internal("chunk failed");
                                  });
    expect_true(e.failure());
    expect_true(e.message() == "chunk failed");
    expect_true(chunks == 1);
  }

  test_that("returns an exception of the body as an error") {
    ThreadedPool pool(4);
    Error e = rpjrt::parallel_for(pool, 100, kAnyThreads,
                                  [&](std::int64_t b0, std::int64_t) -> Error {
                                    if (b0 == 0)
                                      throw std::runtime_error("out of memory");
                                    return Error::Success();
                                  });
    expect_true(e.failure());
    expect_true(e.message() == "out of memory");
  }
}

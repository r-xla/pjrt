#pragma once

#include <functional>
#include <memory>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

// Wrapper class for PJRT_Event - represents an async operation completion.
// This is internal infrastructure for the async API.
class PJRTEvent {
 public:
  PJRTEvent(PJRT_Event* event, std::shared_ptr<PJRT_Api> api);
  ~PJRTEvent();

  // Non-copyable but movable
  PJRTEvent(const PJRTEvent&) = delete;
  PJRTEvent& operator=(const PJRTEvent&) = delete;
  PJRTEvent(PJRTEvent&& other) noexcept;
  PJRTEvent& operator=(PJRTEvent&& other) noexcept;

  // Non-blocking check if the event has completed
  bool is_ready() const;

  // Blocking wait until the event completes
  void await() const;

  // Check for errors after event completes (call only after is_ready() or
  // await())
  void check_error() const;

  // Register a callback to be called when the event is ready.
  // The callback receives any error that occurred (or nullptr if success).
  // Note: The callback runs on a PJRT thread, not the R main thread.
  // The callback must be safe to call from any thread.
  void on_ready(std::function<void(PJRT_Error*)> callback);

  // Get the underlying PJRT_Event pointer (for internal use)
  PJRT_Event* get() const { return event_; }

 private:
  PJRT_Event* event_;
  std::shared_ptr<PJRT_Api> api_;
};

}  // namespace rpjrt

#include "event.h"

#include <utility>

#include "utils.h"

namespace rpjrt {

PJRTEvent::PJRTEvent(PJRTHandle<PJRT_Event> event,
                     std::shared_ptr<PJRT_Api> api)
    : api_(std::move(api)), event_(std::move(event)) {}

bool PJRTEvent::is_ready() const {
  PJRT_Event_IsReady_Args args{};
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.event = event_.get();
  check_err(api_.get(), api_->PJRT_Event_IsReady_(&args));
  return args.is_ready;
}

void PJRTEvent::await() const {
  PJRT_Event_Await_Args args{};
  args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  args.event = event_.get();
  check_err(api_.get(), api_->PJRT_Event_Await_(&args));
}

void PJRTEvent::check_error() const {
  PJRT_Event_Error_Args args{};
  args.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
  args.event = event_.get();
  PJRT_Error* error = api_->PJRT_Event_Error_(&args);
  if (error != nullptr) {
    check_err(api_.get(), error);
  }
}

// Helper struct to pass both callback and api to the C callback wrapper
struct OnReadyCallbackData {
  std::function<void(PJRT_Error*)> callback;
  std::shared_ptr<PJRT_Api> api;
};

// C callback wrapper for on_ready
static void on_ready_callback_wrapper(PJRT_Error* error, void* user_arg) {
  auto* data = static_cast<OnReadyCallbackData*>(user_arg);

  // Invoke the user's callback
  data->callback(error);

  // Per PJRT spec: "Ownership of `error` is passed to the callback.
  // The callback must destroy `error` via `PJRT_Error_Destroy`."
  destroy_error(data->api.get(), error);

  // Per PJRT spec: "The caller retains ownership of `user_arg`."
  // We allocated it, so we must delete it.
  delete data;
}

void PJRTEvent::on_ready(std::function<void(PJRT_Error*)> callback) {
  // Allocate callback data on heap so it survives until called.
  //
  // Note: Per PJRT spec, "The caller retains ownership of `user_arg`."
  // We delete the data in the callback wrapper after invocation.
  // If the callback is never invoked (e.g., event destroyed without
  // completing), this would leak. In practice, PJRT events should always
  // complete before destruction.
  auto* data = new OnReadyCallbackData{std::move(callback), api_};

  PJRT_Event_OnReady_Args args{};
  args.struct_size = sizeof(PJRT_Event_OnReady_Args);
  args.event = event_.get();
  args.callback = on_ready_callback_wrapper;
  args.user_arg = data;

  PJRT_Error* err = api_->PJRT_Event_OnReady_(&args);
  if (err != nullptr) {
    delete data;  // Clean up if registration failed
    check_err(api_.get(), err);
  }
}

}  // namespace rpjrt

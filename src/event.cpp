#include "event.h"

#include "utils.h"

namespace rpjrt {

PJRTEvent::PJRTEvent(PJRT_Event* event, std::shared_ptr<PJRT_Api> api)
    : event_(event), api_(api) {}

PJRTEvent::~PJRTEvent() {
  if (event_ != nullptr) {
    PJRT_Event_Destroy_Args args{};
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.event = event_;
    // Ignore errors during destruction
    api_->PJRT_Event_Destroy_(&args);
  }
}

PJRTEvent::PJRTEvent(PJRTEvent&& other) noexcept
    : event_(other.event_), api_(std::move(other.api_)) {
  other.event_ = nullptr;
}

PJRTEvent& PJRTEvent::operator=(PJRTEvent&& other) noexcept {
  if (this != &other) {
    // Destroy current event if exists
    if (event_ != nullptr) {
      PJRT_Event_Destroy_Args args{};
      args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      args.event = event_;
      api_->PJRT_Event_Destroy_(&args);
    }
    event_ = other.event_;
    api_ = std::move(other.api_);
    other.event_ = nullptr;
  }
  return *this;
}

bool PJRTEvent::is_ready() const {
  PJRT_Event_IsReady_Args args{};
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.event = event_;
  check_err(api_.get(), api_->PJRT_Event_IsReady_(&args));
  return args.is_ready;
}

void PJRTEvent::await() const {
  PJRT_Event_Await_Args args{};
  args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  args.event = event_;
  check_err(api_.get(), api_->PJRT_Event_Await_(&args));
}

void PJRTEvent::check_error() const {
  PJRT_Event_Error_Args args{};
  args.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
  args.event = event_;
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
  if (error != nullptr) {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    data->api->PJRT_Error_Destroy_(&destroy_args);
  }

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
  args.event = event_;
  args.callback = on_ready_callback_wrapper;
  args.user_arg = data;

  PJRT_Error* err = api_->PJRT_Event_OnReady_(&args);
  if (err != nullptr) {
    delete data;  // Clean up if registration failed
    check_err(api_.get(), err);
  }
}

}  // namespace rpjrt

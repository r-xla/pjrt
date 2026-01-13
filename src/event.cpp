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

// C callback wrapper for on_ready
static void on_ready_callback_wrapper(PJRT_Error* error, void* user_arg) {
  auto* callback =
      static_cast<std::function<void(PJRT_Error*)>*>(user_arg);
  (*callback)(error);
  delete callback;  // Clean up the allocated callback
}

void PJRTEvent::on_ready(std::function<void(PJRT_Error*)> callback) {
  // Allocate callback on heap so it survives until called
  auto* callback_ptr = new std::function<void(PJRT_Error*)>(std::move(callback));

  PJRT_Event_OnReady_Args args{};
  args.struct_size = sizeof(PJRT_Event_OnReady_Args);
  args.event = event_;
  args.callback = on_ready_callback_wrapper;
  args.user_arg = callback_ptr;

  PJRT_Error* err = api_->PJRT_Event_OnReady_(&args);
  if (err != nullptr) {
    delete callback_ptr;  // Clean up if registration failed
    check_err(api_.get(), err);
  }
}

}  // namespace rpjrt

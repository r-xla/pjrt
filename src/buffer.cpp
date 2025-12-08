#include "buffer.h"

#include <Rcpp.h>
#include <exception>

#include "utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

PJRTElementType::PJRTElementType(PJRT_Buffer_Type type) : element_type_(type) {}

PJRT_Buffer_Type PJRTElementType::get_type() const { return element_type_; }

int PJRTElementType::as_integer() const {
  return static_cast<int>(element_type_);
}

std::string PJRTElementType::as_string() const {
  switch (element_type_) {
    case PJRT_Buffer_Type_INVALID:
      return "INVALID";
    case PJRT_Buffer_Type_PRED:
      return "pred";
    case PJRT_Buffer_Type_S8:
      return "i8";
    case PJRT_Buffer_Type_S16:
      return "i16";
    case PJRT_Buffer_Type_S32:
      return "i32";
    case PJRT_Buffer_Type_S64:
      return "i64";
    case PJRT_Buffer_Type_U8:
      return "ui8";
    case PJRT_Buffer_Type_U16:
      return "ui16";
    case PJRT_Buffer_Type_U32:
      return "ui32";
    case PJRT_Buffer_Type_U64:
      return "ui64";
    case PJRT_Buffer_Type_F32:
      return "f32";
    case PJRT_Buffer_Type_F64:
      return "f64";
    default:
      Rcpp::stop("Unknown element type: %d", as_integer());
  }
}

PJRTMemory::PJRTMemory(PJRT_Memory *memory, std::shared_ptr<PJRT_Api> api)
    : memory(memory), api(api) {}

std::string PJRTMemory::debug_string() {
  PJRT_Memory_DebugString_Args args{};
  args.struct_size = sizeof(PJRT_Memory_DebugString_Args);
  args.memory = this->memory;
  check_err(this->api.get(), this->api->PJRT_Memory_DebugString_(&args));
  return std::string(args.debug_string, args.debug_string_size);
}

int PJRTMemory::id() {
  PJRT_Memory_Id_Args args{};
  args.struct_size = sizeof(PJRT_Memory_Id_Args);
  args.memory = this->memory;
  check_err(this->api.get(), this->api->PJRT_Memory_Id_(&args));
  return args.id;
}

std::string PJRTMemory::kind() {
  PJRT_Memory_Kind_Args args{};
  args.struct_size = sizeof(PJRT_Memory_Kind_Args);
  args.memory = this->memory;
  check_err(this->api.get(), this->api->PJRT_Memory_Kind_(&args));
  return std::string(args.kind, args.kind_size);
}

std::string PJRTMemory::to_string() {
  PJRT_Memory_ToString_Args args{};
  args.struct_size = sizeof(PJRT_Memory_ToString_Args);
  args.memory = this->memory;
  check_err(this->api.get(), this->api->PJRT_Memory_ToString_(&args));
  return std::string(args.to_string, args.to_string_size);
}

PJRTBufferMemoryLayout::PJRTBufferMemoryLayout(PJRT_Buffer_MemoryLayout layout)
    : layout(layout) {}

PJRTBuffer::PJRTBuffer(PJRT_Buffer *buffer, std::shared_ptr<PJRT_Api> api)
    : buffer(buffer), api(api) {}

PJRTBuffer::~PJRTBuffer() {
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Destroy_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Destroy_(&args));
}

std::vector<int64_t> PJRTBuffer::dimensions() {
  PJRT_Buffer_Dimensions_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Dimensions_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Dimensions_(&args));

  return std::vector<int64_t>(args.dims, args.dims + args.num_dims);
}

std::unique_ptr<PJRTMemory> PJRTBuffer::memory() {
  PJRT_Buffer_Memory_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Memory_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Memory_(&args));

  return std::make_unique<PJRTMemory>(args.memory, this->api);
}

std::unique_ptr<PJRTBufferMemoryLayout> PJRTBuffer::memory_layout() {
  PJRT_Buffer_GetMemoryLayout_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_GetMemoryLayout_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_GetMemoryLayout_(&args));

  return std::make_unique<PJRTBufferMemoryLayout>(args.layout);
}

PJRT_Buffer_Type PJRTBuffer::element_type() {
  PJRT_Buffer_ElementType_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_ElementType_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_ElementType_(&args));

  return args.type;
}

std::unique_ptr<PJRTDevice> PJRTBuffer::device() {
  PJRT_Buffer_Device_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Device_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Device_(&args));

  return std::make_unique<PJRTDevice>(args.device, this->api);
}

// Copy a device buffer to the host and wait for the copy to complete.
void BufferToHostAndWait(const PJRT_Api *api,
                         PJRT_Buffer_ToHostBuffer_Args *args) {
  check_err(api, api->PJRT_Buffer_ToHostBuffer_(args));

  PJRT_Event_Await_Args event_args = {0};
  event_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  event_args.event = args->event;
  check_err(api, api->PJRT_Event_Await_(&event_args));

  PJRT_Event_Destroy_Args destroy_args = {0};
  destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy_args.event = args->event;
  check_err(api, api->PJRT_Event_Destroy_(&destroy_args));
}

void PJRTBuffer::buffer_to_host(std::span<uint8_t> &host_buffer) {
  PJRT_Buffer_ToHostBuffer_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args);
  args.src = this->buffer;
  args.dst = host_buffer.data();
  args.dst_size = host_buffer.size();

  // Perform the copy and wait for it to complete
  BufferToHostAndWait(this->api.get(), &args);
}

PJRTLazyBuffer::EventHandle::EventHandle(PJRT_Event *event,
                                         std::shared_ptr<PJRT_Api> api,
                                         std::shared_ptr<void> owner,
                                         PJRT_Event **event_slot,
                                         std::shared_ptr<std::thread> worker,
                                         std::shared_ptr<std::exception_ptr>
                                             execution_error)
    : event(event),
      api(std::move(api)),
      owner(std::move(owner)),
      event_slot(event_slot),
      worker(std::move(worker)),
      execution_error(std::move(execution_error)) {}

void PJRTLazyBuffer::EventHandle::AwaitAndDestroy() {
  std::call_once(once, [&]() {
    if (worker && worker->joinable()) {
      worker->join();
    }

    std::exception_ptr pending_error =
        execution_error ? *execution_error : std::exception_ptr();
    if (event == nullptr && event_slot != nullptr) {
      event = *event_slot;
    }
    if (event == nullptr) {
      if (pending_error) std::rethrow_exception(pending_error);
      return;
    }

    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = event;
    check_err(this->api.get(), this->api->PJRT_Event_Await_(&await_args));

    PJRT_Event_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = event;
    check_err(this->api.get(), this->api->PJRT_Event_Destroy_(&destroy_args));
    event = nullptr;

    if (pending_error) std::rethrow_exception(pending_error);
  });
}

PJRTLazyBuffer::PJRTLazyBuffer(PJRT_Buffer *buffer, PJRT_Event *event,
                               std::shared_ptr<PJRT_Api> api)
    : pending_buffer(buffer),
      buffer_slot(nullptr),
      event(std::make_shared<EventHandle>(event, api)),
      api(std::move(api)) {}

PJRTLazyBuffer::PJRTLazyBuffer(PJRT_Buffer *buffer,
                               std::shared_ptr<EventHandle> shared_event,
                               std::shared_ptr<PJRT_Api> api,
                               PJRT_Buffer **buffer_slot)
    : pending_buffer(buffer),
      buffer_slot(buffer_slot),
      event(std::move(shared_event)),
      api(std::move(api)) {}

PJRTLazyBuffer::~PJRTLazyBuffer() { ensure_ready(); }

void PJRTLazyBuffer::ensure_ready() {
  if (resolved) return;

  if (event) event->AwaitAndDestroy();

  if (pending_buffer == nullptr && buffer_slot != nullptr) {
    pending_buffer = *buffer_slot;
  }

  if (pending_buffer == nullptr) return;

  resolved = std::make_unique<PJRTBuffer>(pending_buffer, this->api);
}

bool PJRTLazyBuffer::is_ready() {
  if (resolved) return true;

  if (pending_buffer == nullptr && buffer_slot != nullptr) {
    pending_buffer = *buffer_slot;
  }

  if (!event) return pending_buffer != nullptr;
  if (event->event == nullptr && event->event_slot != nullptr) {
    event->event = *event->event_slot;
  }
  if (event->event == nullptr) return false;

  PJRT_Event_IsReady_Args args{};
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.event = event->event;
  check_err(this->api.get(), this->api->PJRT_Event_IsReady_(&args));
  return args.is_ready;
}

PJRTBuffer *PJRTLazyBuffer::operator->() {
  ensure_ready();
  return resolved.get();
}

PJRTBuffer &PJRTLazyBuffer::operator*() {
  ensure_ready();
  return *resolved;
}

std::unique_ptr<PJRTBuffer> PJRTLazyBuffer::materialize() {
  ensure_ready();
  if (pending_buffer == nullptr && buffer_slot != nullptr) {
    pending_buffer = *buffer_slot;
  }
  if (!resolved && pending_buffer != nullptr) {
    resolved = std::make_unique<PJRTBuffer>(pending_buffer, this->api);
  }
  pending_buffer = nullptr;
  return std::move(resolved);
}

std::vector<int64_t> PJRTLazyBuffer::dimensions() {
  return this->operator->()->dimensions();
}

std::unique_ptr<PJRTMemory> PJRTLazyBuffer::memory() {
  return this->operator->()->memory();
}

std::unique_ptr<PJRTBufferMemoryLayout> PJRTLazyBuffer::memory_layout() {
  return this->operator->()->memory_layout();
}

PJRT_Buffer_Type PJRTLazyBuffer::element_type() {
  return this->operator->()->element_type();
}

std::unique_ptr<PJRTDevice> PJRTLazyBuffer::device() {
  return this->operator->()->device();
}

std::shared_ptr<PJRT_Api> PJRTLazyBuffer::get_api() {
  return this->operator->()->get_api();
}

void PJRTLazyBuffer::buffer_to_host(std::span<uint8_t> &host_buffer) {
  this->operator->()->buffer_to_host(host_buffer);
}

PJRT_Buffer *PJRTLazyBuffer::raw_buffer() {
  ensure_ready();
  return resolved ? resolved->buffer : nullptr;
}

}  // namespace rpjrt

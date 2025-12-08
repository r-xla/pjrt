#pragma once

#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "device.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

class PJRTElementType {
 public:
  explicit PJRTElementType(PJRT_Buffer_Type type);

  PJRT_Buffer_Type get_type() const;

  int as_integer() const;

  std::string as_string() const;

 private:
  PJRT_Buffer_Type element_type_;
};

class PJRTMemory {
 public:
  PJRT_Memory* memory;
  PJRTMemory(PJRT_Memory* memory, std::shared_ptr<PJRT_Api> api);
  std::string debug_string();
  int id();
  std::string kind();
  std::string to_string();

 private:
  std::shared_ptr<PJRT_Api> api;
};

class PJRTBufferMemoryLayout {
 public:
  PJRT_Buffer_MemoryLayout layout;
  PJRTBufferMemoryLayout(PJRT_Buffer_MemoryLayout layout);
};

class PJRTBuffer {
 public:
  PJRTBuffer(PJRT_Buffer* buffer, std::shared_ptr<PJRT_Api> api);
  PJRT_Buffer* buffer;
  ~PJRTBuffer();
  std::vector<int64_t> dimensions();
  std::unique_ptr<PJRTMemory> memory();
  std::unique_ptr<PJRTBufferMemoryLayout> memory_layout();
  PJRT_Buffer_Type element_type();
  std::unique_ptr<PJRTDevice> device();
  std::shared_ptr<PJRT_Api> get_api() const { return api; }
 void buffer_to_host(std::span<uint8_t>& host_buffer);

 private:
  std::shared_ptr<PJRT_Api> api;
};

class PJRTLazyBuffer {
 public:
  struct EventHandle {
    EventHandle(PJRT_Event* event, std::shared_ptr<PJRT_Api> api,
                std::shared_ptr<void> owner = {},
                PJRT_Event** event_slot = nullptr,
                std::shared_ptr<std::thread> worker = {},
                std::shared_ptr<std::exception_ptr> execution_error = {});
    void AwaitAndDestroy();
    PJRT_Event* event;
    std::once_flag once;
    std::shared_ptr<PJRT_Api> api;
    std::shared_ptr<void> owner;
    PJRT_Event** event_slot;
    std::shared_ptr<std::thread> worker;
    std::shared_ptr<std::exception_ptr> execution_error;
  };

  PJRTLazyBuffer(PJRT_Buffer* buffer, PJRT_Event* event,
                 std::shared_ptr<PJRT_Api> api);
  PJRTLazyBuffer(PJRT_Buffer* buffer,
                 std::shared_ptr<EventHandle> shared_event,
                 std::shared_ptr<PJRT_Api> api,
                 PJRT_Buffer** buffer_slot = nullptr);
  ~PJRTLazyBuffer();

  PJRTBuffer* operator->();
  PJRTBuffer& operator*();

  bool is_ready();
  std::unique_ptr<PJRTBuffer> materialize();
  std::vector<int64_t> dimensions();
  std::unique_ptr<PJRTMemory> memory();
  std::unique_ptr<PJRTBufferMemoryLayout> memory_layout();
  PJRT_Buffer_Type element_type();
  std::unique_ptr<PJRTDevice> device();
  std::shared_ptr<PJRT_Api> get_api();
  void buffer_to_host(std::span<uint8_t>& host_buffer);
  PJRT_Buffer* raw_buffer();

 private:
  void ensure_ready();

  PJRT_Buffer* pending_buffer;
  PJRT_Buffer** buffer_slot;
  std::shared_ptr<EventHandle> event;
  std::unique_ptr<PJRTBuffer> resolved;
  std::shared_ptr<PJRT_Api> api;
};

}  // namespace rpjrt

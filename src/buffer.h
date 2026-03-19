#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "device.h"
#include "event.h"
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
  // Returns event - caller must keep host_buffer alive until event completes
  std::unique_ptr<PJRTEvent> buffer_to_host_async(
      std::span<uint8_t>& host_buffer);

  // Set the completion event for this buffer (e.g., from execution)
  void set_completion_event(std::shared_ptr<PJRTEvent> event);

  // Non-blocking check if the buffer's operation is complete
  bool is_ready() const;

  // Block until the buffer's operation is complete, then check for errors
  void await();

 private:
  std::shared_ptr<PJRT_Api> api;
  std::shared_ptr<PJRTEvent> completion_event_;
};

// Holds result of an async device-to-host transfer.
// Owns the host data and the completion event.
class PJRTHostData {
 public:
  PJRTHostData(std::unique_ptr<std::vector<uint8_t>> data,
               std::unique_ptr<PJRTEvent> event);

  // Non-blocking check if the transfer is complete
  bool is_ready() const;

  // Block until the transfer is complete, then check for errors
  void await();

  // Access the raw data (only valid after await)
  const std::vector<uint8_t>& data() const { return *data_; }

 private:
  std::unique_ptr<std::vector<uint8_t>> data_;
  std::unique_ptr<PJRTEvent> event_;
};

}  // namespace rpjrt

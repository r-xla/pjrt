#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "device.h"
#include "pjrt.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

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

}  // namespace rpjrt

#pragma once

#include <memory>
#include <vector>

#include "device.h"
#include "pjrt.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

class PJRTMemory {
 public:
  PJRT_Memory* memory;
  PJRTMemory(PJRT_Memory* memory, std::shared_ptr<PJRT_API> api);

 private:
  std::shared_ptr<PJRT_API> api;
};

class PJRTBufferMemoryLayout {
 public:
  PJRT_Buffer_MemoryLayout layout;
  PJRTBufferMemoryLayout(PJRT_Buffer_MemoryLayout layout);
};

class PJRTBuffer {
 public:
  PJRTBuffer(PJRT_Buffer* buffer, std::shared_ptr<PJRT_API> api);
  PJRT_Buffer* buffer;
  ~PJRTBuffer();
  std::vector<int64_t> dimensions();
  std::unique_ptr<PJRTMemory> memory();
  std::unique_ptr<PJRTBufferMemoryLayout> memory_layout();
  PJRT_Buffer_Type element_type();
  std::unique_ptr<PJRTDevice> device();

 private:
  std::shared_ptr<PJRT_API> api;
};

}  // namespace rpjrt

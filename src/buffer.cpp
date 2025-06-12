#include "buffer.h"

#include "pjrt.h"
#include "utils.h"

namespace rpjrt {

PJRTMemory::PJRTMemory(PJRT_Memory *memory, std::shared_ptr<PJRT_Api> api)
    : memory(memory), api(api) {}

PJRTBufferMemoryLayout::PJRTBufferMemoryLayout(PJRT_Buffer_MemoryLayout layout)
    : layout(layout) {}

PJRTBuffer::PJRTBuffer(PJRT_Buffer *buffer, std::shared_ptr<PJRT_Api> api)
    : buffer(buffer), api(api) {}

PJRTBuffer::~PJRTBuffer() {
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Destroy_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Destroy(&args));
}

std::vector<int64_t> PJRTBuffer::dimensions() {
  PJRT_Buffer_Dimensions_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Dimensions_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Dimensions(&args));

  return std::vector<int64_t>(args.dims, args.dims + args.num_dims);
}

std::unique_ptr<PJRTMemory> PJRTBuffer::memory() {
  PJRT_Buffer_Memory_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Memory_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Memory(&args));

  return std::make_unique<PJRTMemory>(args.memory, this->api);
}

std::unique_ptr<PJRTBufferMemoryLayout> PJRTBuffer::memory_layout() {
  PJRT_Buffer_GetMemoryLayout_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_GetMemoryLayout_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_GetMemoryLayout(&args));

  return std::make_unique<PJRTBufferMemoryLayout>(args.layout);
}

PJRT_Buffer_Type PJRTBuffer::element_type() {
  PJRT_Buffer_ElementType_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_ElementType_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_ElementType(&args));

  return args.type;
}

std::unique_ptr<PJRTDevice> PJRTBuffer::device() {
  PJRT_Buffer_Device_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Device_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Device(&args));

  return std::make_unique<PJRTDevice>(args.device);
}

}  // namespace rpjrt
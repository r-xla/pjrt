#include "buffer.h"

#include <Rcpp.h>

#include "deferred_release.h"
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

PJRT_Buffer *PJRTBuffer::checked_buffer() const {
  if (this->buffer == nullptr) {
    Rcpp::stop("called on deleted or donated buffer");
  }
  return this->buffer;
}

PJRTBuffer::~PJRTBuffer() {
  // Drain the deferred release queue while we're on the main R thread.
  // This ensures R objects preserved for zero-copy transfers get released
  // when buffers are garbage collected.
  process_pending_releases();

  // A null `buffer` means this wrapper held a donated input whose PJRT
  // handle was invalidated by Execute and explicitly nulled by the
  // keepalive-transfer logic. There's nothing left to destroy.
  if (this->buffer == nullptr) return;

  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Destroy_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_Destroy_(&args));
}

const std::vector<int64_t> &PJRTBuffer::dimensions() {
  if (!meta_cached_) cache_meta();
  return cached_dims_;
}

std::unique_ptr<PJRTMemory> PJRTBuffer::memory() {
  PJRT_Buffer_Memory_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Memory_Args);
  args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_Memory_(&args));

  return std::make_unique<PJRTMemory>(args.memory, this->api);
}

std::unique_ptr<PJRTBufferMemoryLayout> PJRTBuffer::memory_layout() {
  PJRT_Buffer_GetMemoryLayout_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_GetMemoryLayout_Args);
  args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_GetMemoryLayout_(&args));

  return std::make_unique<PJRTBufferMemoryLayout>(args.layout);
}

std::vector<int64_t> PJRTBuffer::minor_to_major() {
  const auto ndim = dimensions().size();

  // For scalars and vectors there is only one possible element order, and
  // readback treats them as dense, so the physical layout is irrelevant.
  // Return the trivial row-major permutation without inspecting the layout —
  // this also means an exotic-but-irrelevant layout never blocks a 0-D/1-D
  // readback.
  std::vector<int64_t> row_major(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    row_major[i] = static_cast<int64_t>(ndim - 1 - i);
  }
  if (ndim <= 1) return row_major;

  PJRT_Buffer_GetMemoryLayout_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_GetMemoryLayout_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_GetMemoryLayout_(&args));

  // Our readback can only faithfully reorder a dense, untiled layout expressed
  // as a minor-to-major permutation. The other representations would require
  // machinery we don't implement — stride-walking for a strided layout, or
  // de-tiling (which also changes the physical byte count via padding) for a
  // tiled one — so we error rather than silently returning wrong data.
  //
  // In practice none of these occur on the platforms pjrt supports: CPU, CUDA,
  // and Metal all hand back dense untiled layouts (tiling is a TPU feature).
  // The checks therefore only fire if a future/exotic backend produces a
  // layout we cannot honor, turning silent corruption into a clear error.
  if (args.layout.type != PJRT_Buffer_MemoryLayout_Type_Tiled) {
    Rcpp::stop(
        "Unsupported strided buffer memory layout on readback; only dense "
        "untiled layouts are supported (CPU/CUDA/Metal).");
  }
  if (args.layout.tiled.num_tiles > 0) {
    Rcpp::stop(
        "Unsupported tiled buffer memory layout on readback; only dense "
        "untiled layouts are supported (tiling is a TPU feature, which pjrt "
        "does not support).");
  }
  if (args.layout.tiled.minor_to_major_size != ndim) {
    Rcpp::stop(
        "Buffer memory layout rank (%d) does not match buffer rank (%d).",
        static_cast<int>(args.layout.tiled.minor_to_major_size),
        static_cast<int>(ndim));
  }
  return std::vector<int64_t>(
      args.layout.tiled.minor_to_major,
      args.layout.tiled.minor_to_major + args.layout.tiled.minor_to_major_size);
}

PJRT_Buffer_Type PJRTBuffer::element_type() {
  if (!meta_cached_) cache_meta();
  return cached_type_;
}

std::unique_ptr<PJRTDevice> PJRTBuffer::device() {
  PJRT_Buffer_Device_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_Device_Args);
  args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_Device_(&args));

  return std::make_unique<PJRTDevice>(args.device, this->api);
}

// Populate the immutable-metadata cache with one PJRT C API call each for
// dtype, shape, and device. Called on the first element_type()/dimensions()/
// device_ptr() read; the values never change, so it runs at most once.
void PJRTBuffer::cache_meta() {
  PJRT_Buffer_ElementType_Args type_args{};
  type_args.struct_size = sizeof(PJRT_Buffer_ElementType_Args);
  type_args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_ElementType_(&type_args));
  cached_type_ = type_args.type;

  PJRT_Buffer_Dimensions_Args dim_args{};
  dim_args.struct_size = sizeof(PJRT_Buffer_Dimensions_Args);
  dim_args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_Dimensions_(&dim_args));
  cached_dims_.assign(dim_args.dims, dim_args.dims + dim_args.num_dims);

  PJRT_Buffer_Device_Args dev_args{};
  dev_args.struct_size = sizeof(PJRT_Buffer_Device_Args);
  dev_args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_Device_(&dev_args));
  cached_device_ = dev_args.device;

  meta_cached_ = true;
}

PJRT_Device *PJRTBuffer::device_ptr() {
  if (!meta_cached_) cache_meta();
  return cached_device_;
}

std::unique_ptr<PJRTEvent> PJRTBuffer::buffer_to_host_async(
    std::span<uint8_t> &host_buffer) {
  PJRT_Buffer_ToHostBuffer_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args);
  args.src = checked_buffer();
  args.dst = host_buffer.data();
  args.dst_size = host_buffer.size();

  // Start the copy but don't wait. The bytes arrive in the buffer's *device*
  // layout (the CPU runtime ignores a requested host_layout), so callers must
  // consult minor_to_major() to interpret them; see device_to_row_major().
  check_err(this->api.get(), this->api->PJRT_Buffer_ToHostBuffer_(&args));

  // Return the event - caller is responsible for waiting and keeping
  // host_buffer alive
  if (args.event != nullptr) {
    return std::make_unique<PJRTEvent>(args.event, this->api);
  }
  return nullptr;
}

PJRTEvent PJRTBuffer::ready_event() {
  PJRT_Buffer_ReadyEvent_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_ReadyEvent_Args);
  args.buffer = checked_buffer();
  check_err(this->api.get(), this->api->PJRT_Buffer_ReadyEvent_(&args));
  return PJRTEvent(args.event, this->api);
}

bool PJRTBuffer::is_ready() {
  auto event = ready_event();
  return event.is_ready();
}

void PJRTBuffer::await() {
  auto event = ready_event();
  event.await();
  event.check_error();
}

std::unique_ptr<PJRTBuffer> PJRTBuffer::copy_to_device(PJRTDevice &dst_device) {
  PJRT_Buffer_CopyToDevice_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_CopyToDevice_Args);
  args.buffer = checked_buffer();
  args.dst_device = dst_device.device;
  check_err(this->api.get(), this->api->PJRT_Buffer_CopyToDevice_(&args));
  return std::make_unique<PJRTBuffer>(args.dst_buffer, this->api);
}

bool PJRTBuffer::is_deleted() {
  if (this->buffer == nullptr) return true;
  PJRT_Buffer_IsDeleted_Args args{};
  args.struct_size = sizeof(PJRT_Buffer_IsDeleted_Args);
  args.buffer = this->buffer;
  check_err(this->api.get(), this->api->PJRT_Buffer_IsDeleted_(&args));
  return args.is_deleted;
}

// PJRTHostData implementation

PJRTHostData::PJRTHostData(std::unique_ptr<std::vector<uint8_t>> data,
                           std::unique_ptr<PJRTEvent> event)
    : data_(std::move(data)), event_(std::move(event)) {}

bool PJRTHostData::is_ready() const {
  if (!event_) return true;
  return event_->is_ready();
}

void PJRTHostData::await() {
  if (!event_) return;
  event_->await();
  event_->check_error();
}

}  // namespace rpjrt

#include "utils.h"

#include <memory>
#include <optional>

void check_err(const PJRT_Api *api, PJRT_Error *err) {
  if (err) {
    PJRT_Error_Message_Args args;
    args.error = err;
    args.struct_size = sizeof(PJRT_Error_Message_Args);
    api->PJRT_Error_Message_(&args);
    throw std::runtime_error(args.message);
  }
}

size_t sizeof_pjrt_buffer_type(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_F64:
      return 8;
    case PJRT_Buffer_Type_S8:
      return 1;
    case PJRT_Buffer_Type_S16:
      return 2;
    case PJRT_Buffer_Type_S32:
      return 4;
    case PJRT_Buffer_Type_S64:
      return 8;
    case PJRT_Buffer_Type_U8:
      return 1;
    case PJRT_Buffer_Type_U16:
      return 2;
    case PJRT_Buffer_Type_U32:
      return 4;
    case PJRT_Buffer_Type_U64:
      return 8;
    case PJRT_Buffer_Type_PRED:
      return 1;
    default:
      throw std::runtime_error("Unsupported PJRT buffer type");
  }
}

bool format_is_irrelevant(const std::vector<int64_t> &dims) {
  // there is at most one non-1 dimension

  int64_t non_1_dim = 0;
  for (int64_t dim : dims) {
    if (dim != 1) {
      non_1_dim++;
    }
  }
  return non_1_dim <= 1;
}

std::optional<std::vector<int64_t>> get_byte_strides(
    const std::vector<int64_t> &dims, bool row_major, size_t sizeof_type) {
  std::optional<std::vector<int64_t>> byte_strides_opt;
  if (!row_major) {
    // For column-major (R's native format), we need to specify byte_strides
    auto byte_strides = std::vector<int64_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      byte_strides[i] = sizeof_type;
      for (size_t j = 0; j < i; ++j) {
        byte_strides[i] *= dims[j];
      }
    }
    byte_strides_opt = byte_strides;
  }
  return byte_strides_opt;
}

#include "utils.h"

#include <optional>
#include <stdexcept>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"

void destroy_error(const PJRT_Api *api, PJRT_Error *err) {
  if (err == nullptr) return;
  PJRT_Error_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_Error_Destroy_Args);
  args.error = err;
  api->PJRT_Error_Destroy_(&args);
}

void check_err(const PJRT_Api *api, PJRT_Error *err) {
  if (err) {
    PJRT_Error_Message_Args args{};
    args.struct_size = sizeof(PJRT_Error_Message_Args);
    args.error = err;
    api->PJRT_Error_Message_(&args);
    std::string message(args.message, args.message_size);
    // The PJRT_Error is owned by the caller and must be destroyed even on the
    // failure path, before we unwind via the exception.
    destroy_error(api, err);
    throw std::runtime_error(message);
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

std::vector<int64_t> dims2strides(std::vector<int64_t> dims, bool row_major) {
  std::vector<int64_t> strides(dims.size(), 1);

  if (dims.size() <= 1) {
    return strides;
  }

  if (row_major) {
    for (int i = dims.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  } else {
    for (int i = 1; i < dims.size(); ++i) {
      strides[i] = strides[i - 1] * dims[i - 1];
    }
  }
  return strides;
}

std::vector<int64_t> id2indices(int lid, const std::vector<int64_t> strides) {
  std::vector<int64_t> idx(strides.size());
  for (size_t k = 0; k < strides.size(); ++k) {
    const int64_t s = strides[k];
    idx[k] = lid / s;
    lid %= s;
  }
  return idx;
}

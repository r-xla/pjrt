#include "utils.h"

#include <unistd.h>

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

StderrCapture begin_stderr_capture() {
  StderrCapture cap;
  std::fflush(stderr);
  FILE *tmp = std::tmpfile();
  if (tmp == nullptr) {
    return cap;  // capture disabled
  }
  int saved = dup(STDERR_FILENO);
  if (saved == -1) {
    std::fclose(tmp);
    return cap;
  }
  if (dup2(fileno(tmp), STDERR_FILENO) == -1) {
    close(saved);
    std::fclose(tmp);
    return cap;
  }
  cap.saved_fd = saved;
  cap.tmp = tmp;
  return cap;
}

void end_stderr_capture(StderrCapture &cap, bool replay) {
  if (cap.saved_fd == -1) {
    return;  // capture was disabled; nothing to restore
  }
  FILE *tmp = static_cast<FILE *>(cap.tmp);
  std::fflush(stderr);  // push any buffered XLA output into the temp sink
  dup2(cap.saved_fd, STDERR_FILENO);
  close(cap.saved_fd);
  cap.saved_fd = -1;

  if (replay) {
    std::fseek(tmp, 0, SEEK_SET);
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), tmp)) > 0) {
      std::fwrite(buf, 1, n, stderr);
    }
    std::fflush(stderr);
  }
  std::fclose(tmp);  // closes the fd and deletes the temp file
  cap.tmp = nullptr;
}

}  // namespace rpjrt

void check_err(const PJRT_Api *api, PJRT_Error *err) {
  if (err) {
    PJRT_Error_Message_Args args{};
    args.struct_size = sizeof(PJRT_Error_Message_Args);
    args.error = err;
    api->PJRT_Error_Message_(&args);
    std::string message(args.message, args.message_size);
    destroy_error(api, err);
    throw std::runtime_error(message);
  }
}

PJRT_Error_Code get_error_code(const PJRT_Api *api, PJRT_Error *err) {
  PJRT_Error_GetCode_Args args{};
  args.struct_size = sizeof(PJRT_Error_GetCode_Args);
  args.error = err;
  PJRT_Error *inner = api->PJRT_Error_GetCode_(&args);
  if (inner != nullptr) {
    // GetCode itself failed; drop the inner error and report UNKNOWN so the
    // caller surfaces the original error via check_err.
    destroy_error(api, inner);
    return PJRT_Error_Code_UNKNOWN;
  }
  return args.code;
}

void destroy_error(const PJRT_Api *api, PJRT_Error *err) {
  if (err == nullptr) return;
  PJRT_Error_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_Error_Destroy_Args);
  args.error = err;
  api->PJRT_Error_Destroy_(&args);
}

PJRT_Buffer_Type string_to_pjrt_buffer_type(const std::string &dtype) {
  if (dtype == "f32") return PJRT_Buffer_Type_F32;
  if (dtype == "f64") return PJRT_Buffer_Type_F64;
  if (dtype == "i8") return PJRT_Buffer_Type_S8;
  if (dtype == "i16") return PJRT_Buffer_Type_S16;
  if (dtype == "i32") return PJRT_Buffer_Type_S32;
  if (dtype == "i64") return PJRT_Buffer_Type_S64;
  if (dtype == "ui8") return PJRT_Buffer_Type_U8;
  if (dtype == "ui16") return PJRT_Buffer_Type_U16;
  if (dtype == "ui32") return PJRT_Buffer_Type_U32;
  if (dtype == "ui64") return PJRT_Buffer_Type_U64;
  if (dtype == "pred") return PJRT_Buffer_Type_PRED;
  throw std::runtime_error("Unsupported type: " + dtype);
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

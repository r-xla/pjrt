#include "utils.h"

#include <unistd.h>

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <sys/mman.h>  // memfd_create
#endif

#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

namespace {

// A single capture sink, reused for the lifetime of the process rather than
// recreated per call (creating/unlinking a temp file each time dominated the
// cost — see the benchmark in the memory-management design doc). On Linux it is
// an anonymous RAM-backed file (`memfd_create`), which matters because the
// capture is only ever armed on non-CPU backends and CUDA is Linux-only; on
// other platforms it falls back to an unlinked `tmpfile()`. try_alloc runs only
// on the main R thread, so this shared sink never has concurrent users.
int g_sink_fd = -1;
bool g_sink_tried = false;

// Open the reusable sink fd, or -1 if unavailable (capture then no-ops).
int open_sink_fd() {
#if defined(__linux__)
  {
    int fd = memfd_create("rpjrt_stderr_capture", MFD_CLOEXEC);
    if (fd != -1) return fd;
    // fall through to the portable path if memfd is unavailable at runtime
  }
#endif
  // tmpfile() gives an already-unlinked temp file; dup a raw fd we own and
  // close the FILE* (the inode stays alive via our fd until we close it).
  FILE *f = std::tmpfile();
  if (f == nullptr) return -1;
  int fd = dup(fileno(f));
  std::fclose(f);
  return fd;
}

}  // namespace

StderrCapture begin_stderr_capture() {
  StderrCapture cap;
  if (!g_sink_tried) {
    g_sink_tried = true;
    g_sink_fd = open_sink_fd();
  }
  if (g_sink_fd == -1) {
    return cap;  // capture disabled; leave stderr untouched
  }

  // Reset the shared sink to empty: ftruncate clears the contents but not the
  // file offset, so lseek back to the start as well.
  if (ftruncate(g_sink_fd, 0) == -1) {
    return cap;  // capture disabled for this call
  }
  lseek(g_sink_fd, 0, SEEK_SET);

  std::fflush(stderr);
  int saved = dup(STDERR_FILENO);
  if (saved == -1) {
    return cap;
  }
  if (dup2(g_sink_fd, STDERR_FILENO) == -1) {
    close(saved);
    return cap;
  }
  cap.saved_fd = saved;
  return cap;
}

void end_stderr_capture(StderrCapture &cap, bool replay) {
  if (cap.saved_fd == -1) {
    return;  // capture was disabled; nothing to restore
  }
  std::fflush(stderr);  // push any libc-buffered stderr into the sink
  dup2(cap.saved_fd, STDERR_FILENO);
  close(cap.saved_fd);
  cap.saved_fd = -1;

  if (replay) {
    lseek(g_sink_fd, 0, SEEK_SET);
    char buf[4096];
    ssize_t n;
    while ((n = read(g_sink_fd, buf, sizeof(buf))) > 0) {
      ssize_t off = 0;
      while (off < n) {
        ssize_t w = write(STDERR_FILENO, buf + off, n - off);
        if (w <= 0) break;
        off += w;
      }
    }
  }
  // The sink is left open for reuse; it is reset on the next begin.
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

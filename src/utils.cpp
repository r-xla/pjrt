#include "utils.h"

#include <R_ext/Print.h>  // REprintf
#include <Rinternals.h>   // SEXP, R_ExternalPtrProtected
#include <unistd.h>

// R's headers #define `error` (-> Rf_error), which collides with the `.error`
// members on PJRT's C-API structs used throughout this file. Drop the macro; we
// never call R's error() here (failures throw std::runtime_error instead).
#undef error

#include <cstdio>
#include <cstdlib>  // getenv
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
// other platforms it falls back to an unlinked `tmpfile()`.
//
// Both are process-global because the sink must outlive any single call and is
// shared by every try_alloc caller (there is one stderr for the process). Plain
// globals are safe here only because try_alloc runs solely on the main R
// thread, so the sink never has concurrent users.

// The reusable sink's file descriptor, or -1 if we don't have one (capture then
// no-ops). Created lazily on first use and never closed — it lives for the
// process and is just reset (ftruncate to 0) between captures.
int g_sink_fd = -1;

// Whether we have already attempted to create the sink. Needed because
// `g_sink_fd == -1` alone is ambiguous — it could mean "not created yet" or
// "tried and failed". This flag records that the one-time attempt happened, so
// a permanent failure is decided once instead of re-running open_sink_fd() (and
// its failing syscalls) on every allocation.
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
  // Portable fallback (e.g. macOS, where memfd_create does not exist).
  //
  // We want a bare file descriptor — a raw OS handle (an int) used with
  // read/lseek/ftruncate/dup2 — to match the memfd path, so the rest of the
  // capture code never special-cases this branch. But std::tmpfile() returns a
  // FILE*: the C stdio wrapper that adds userspace buffering and the f* API
  // (fread/fwrite/...) on top of an underlying fd. So we convert FILE* -> fd:
  //
  //   1. tmpfile() creates an already-unlinked temp file (no name; auto-freed
  //      once the last fd to it closes — same self-cleaning as memfd).
  //   2. dup(fileno(f)) makes a second, independent fd referring to the *same*
  //      open file as the FILE*'s own fd.
  //   3. fclose(f) drops the FILE* wrapper and closes its fd, but the file
  //      survives: the kernel reference-counts the open file, and our dup'd fd
  //      still references it.
  //
  // The result is a raw fd we solely own, with no lingering FILE* to manage.
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

  // fflush(nullptr) flushes every output stream, stderr included. We must not
  // name `stderr` here: referencing the C stream symbol in compiled code is
  // flagged by R CMD check.
  std::fflush(nullptr);
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
  // Push any libc-buffered stderr into the sink. fflush(nullptr) flushes all
  // output streams — naming `stderr` would be flagged by R CMD check.
  std::fflush(nullptr);
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

void debug_inform(const char *msg) {
  const char *dbg = std::getenv("PJRT_DEBUG");
  if (dbg != nullptr && dbg[0] != '\0') {
    REprintf("%s\n", msg);
  }
}

}  // namespace rpjrt

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

// Test-only: read the SEXP stored in an external pointer's protected slot. Used
// to assert the keepalive invariant — which RAWSXP a CPU buffer's XPtr pins.
// [[Rcpp::export()]]
SEXP impl_test_xptr_prot(SEXP x) { return R_ExternalPtrProtected(x); }

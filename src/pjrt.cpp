#include <Rcpp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "buffer.h"
#include "buffer_printer.h"
#include "client.h"
#include "deferred_release.h"
#include "pjrt_types.h"
#include "plugin.h"
#include "utils.h"

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTPlugin> impl_plugin_load(const std::string &path) {
  auto ptr = std::make_unique<rpjrt::PJRTPlugin>(path);
  Rcpp::XPtr<rpjrt::PJRTPlugin> xptr(ptr.release(), true);
  xptr.attr("class") = "PJRTPlugin";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTClient> impl_plugin_client_create(
    Rcpp::XPtr<rpjrt::PJRTPlugin> plugin,
    Rcpp::Nullable<Rcpp::List> options = R_NilValue) {
  std::vector<std::pair<std::string, int64_t>> client_options;

  if (options.isNotNull()) {
    Rcpp::List opts(options);
    Rcpp::CharacterVector names = opts.names();
    // We have ensured in the R side, that all optins are integers of length 1
    for (int i = 0; i < opts.size(); ++i) {
      std::string key = Rcpp::as<std::string>(names[i]);
      int64_t value = Rcpp::as<int64_t>(opts[i]);
      client_options.emplace_back(key, value);
    }
  }

  auto client = plugin->client_create(client_options);
  Rcpp::XPtr<rpjrt::PJRTClient> xptr(client.release(), true);
  xptr.attr("class") = "PJRTClient";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTProgram> impl_program_load(const std::string &fname,
                                                 const std::string &format) {
  rpjrt::PJRTProgramFormat fmt;
  if (format == "hlo") {
    fmt = rpjrt::HLO;
  } else if (format == "mlir") {
    fmt = rpjrt::MLIR;
  } else {
    throw std::runtime_error("Unknown program format: " + format);
  }

  auto program = std::make_unique<rpjrt::PJRTProgram>(fname, fmt);
  Rcpp::XPtr<rpjrt::PJRTProgram> xptr(program.release(), true);
  xptr.attr("class") = "PJRTProgram";
  return xptr;
}

// [[Rcpp::export()]]
std::string impl_program_repr(Rcpp::XPtr<rpjrt::PJRTProgram> program,
                              int n = 10) {
  return program->repr(n);
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuildOptions> impl_build_options_create(
    const int num_replicas = 1, const int num_partitions = 1,
    const int device_ordinal = -1) {
  auto build_options = std::make_unique<rpjrt::PJRTBuildOptions>(
      num_replicas, num_partitions, device_ordinal);
  Rcpp::XPtr<rpjrt::PJRTBuildOptions> xptr(build_options.release(), true);
  xptr.attr("class") = "PJRTBuildOptions";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTCompileOptions> impl_compile_options_create(
    Rcpp::XPtr<rpjrt::PJRTBuildOptions> build_options) {
  auto compile_options =
      std::make_unique<rpjrt::PJRTCompileOptions>(build_options->clone());
  Rcpp::XPtr<rpjrt::PJRTCompileOptions> xptr(compile_options.release(), true);
  xptr.attr("class") = "PJRTCompileOptions";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> impl_client_program_compile(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    Rcpp::XPtr<rpjrt::PJRTProgram> program,
    Rcpp::XPtr<rpjrt::PJRTCompileOptions> compile_options) {
  auto executable = client->compile(*program, *compile_options, *device);
  Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> xptr(executable.release(), true);

  xptr.attr("class") = "PJRTLoadedExecutable";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTDevice> impl_loaded_executable_device(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable) {
  auto devices = executable->addressable_devices();
  if (devices.empty()) {
    Rcpp::stop("Loaded executable has no addressable devices");
  }
  // We only support single-device executables; return the first device.
  // PJRTDevice does not own this pointer (the client does), so use a weak ref.
  Rcpp::XPtr<rpjrt::PJRTDevice> xptr(
      new rpjrt::PJRTDevice(devices[0], executable->api), true);
  xptr.attr("class") = "PJRTDevice";
  return xptr;
}

// Copy R data into a pre-allocated typed destination buffer, performing
// type conversion as needed. T is the PJRT-side element type.
template <typename T>
void convert_r_data_to_typed(SEXP data, T *dst, int len) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (TYPEOF(data) == REALSXP) {
      std::copy(REAL(data), REAL(data) + len, dst);
    } else if (TYPEOF(data) == INTSXP) {
      for (int i = 0; i < len; ++i) {
        dst[i] = static_cast<T>(INTEGER(data)[i]);
      }
    } else {
      Rcpp::stop("Cannot convert R type %d to floating point", TYPEOF(data));
    }
  } else if constexpr (std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, int16_t> ||
                       std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, int64_t> ||
                       std::is_same_v<T, uint16_t> ||
                       std::is_same_v<T, uint32_t> ||
                       std::is_same_v<T, uint64_t>) {
    std::copy(INTEGER(data), INTEGER(data) + len, dst);
  } else if constexpr (std::is_same_v<T, bool>) {
    std::copy(LOGICAL(data), LOGICAL(data) + len, dst);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    if (TYPEOF(data) == LGLSXP) {
      for (int i = 0; i < len; ++i) {
        dst[i] = LOGICAL(data)[i] ? 1 : 0;
      }
    } else if (TYPEOF(data) == INTSXP) {
      std::copy(INTEGER(data), INTEGER(data) + len, dst);
    } else {
      Rcpp::stop("Unsupported R type: %d", TYPEOF(data));
    }
  } else {
    Rcpp::stop("Unsupported R type: %d", TYPEOF(data));
  }
}

// Helper to copy R data to a heap-allocated vector with type conversion
template <typename T>
std::unique_ptr<std::vector<T>> copy_r_data_to_vec(SEXP data) {
  int len = Rf_length(data);
  auto vec = std::make_unique<std::vector<T>>(len);
  convert_r_data_to_typed<T>(data, vec->data(), len);
  return vec;
}

// Build a CPU buffer backed by a fresh RAWSXP. `fill` receives a pointer to
// `total_bytes` of writable storage and must populate it with the buffer's
// host bytes (a conversion, a memcpy, or nothing for uninitialized contents).
// PJRT then aliases those bytes via kMutableZeroCopy and the RAWSXP is parked
// in the buffer XPtr's protected slot, so R's GC both keeps it alive exactly as
// long as the PJRTBuffer XPtr is reachable AND accounts for its memory. Every
// CPU buffer-creation path funnels through here, so no CPU buffer's host
// storage is ever invisible to R's GC.
//
// We use kMutableZeroCopy rather than kImmutableZeroCopy because the buffer may
// later be donated to pjrt_execute() as an aliased output, which the immutable
// variant explicitly forbids. PJRT only aliases the bytes: when the XPtr is
// reclaimed the finalizer runs PJRT_Buffer_Destroy first (releasing PJRT's
// alias), then the RAWSXP becomes collectable — no double-free, since PJRT
// never owned the host bytes.
template <typename Fill>
Rcpp::XPtr<rpjrt::PJRTBuffer> make_cpu_buffer(
    Rcpp::XPtr<rpjrt::PJRTClient> &client, size_t total_bytes,
    const std::vector<int64_t> &dims,
    const std::optional<std::vector<int64_t>> &byte_strides,
    PJRT_Buffer_Type dtype, PJRT_Device *device, Fill fill) {
  // PROTECT across buffer_from_host_async: PJRT allocation may trigger R's GC.
  // Once the XPtr holds raw_sexp in its prot slot it stays reachable.
  SEXP raw_sexp = PROTECT(Rf_allocVector(RAWSXP, total_bytes));
  fill(RAW(raw_sexp));
  auto result = client->buffer_from_host_async(
      RAW(raw_sexp), dims, byte_strides, dtype, device,
      PJRT_HostBufferSemantics_kMutableZeroCopy);
  Rcpp::XPtr<rpjrt::PJRTBuffer> buffer_xptr(result.buffer.release(), true,
                                            R_NilValue, raw_sexp);
  buffer_xptr.attr("class") = "PJRTBuffer";
  UNPROTECT(1);
  return buffer_xptr;
}

// Async buffer creation - handles data lifetime internally via on_ready
// callback
template <typename T>
Rcpp::XPtr<rpjrt::PJRTBuffer> create_buffer_from_array_async(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
    const std::vector<int64_t> &dims, PJRT_Buffer_Type dtype,
    bool row_major = false, PJRT_Device *device = nullptr) {
  int len = Rf_length(data);
  if (len == 0) {
    if (!std::any_of(dims.begin(), dims.end(),
                     [](int64_t dim) { return dim == 0; })) {
      Rcpp::stop("Data must be a non-empty vector.");
    }
  }

  auto byte_strides_opt = get_byte_strides(dims, row_major, sizeof(T));

  if (client->is_cpu()) {
    return make_cpu_buffer(client, static_cast<size_t>(len) * sizeof(T), dims,
                           byte_strides_opt, dtype, device, [&](void *dst) {
                             convert_r_data_to_typed<T>(
                                 data, reinterpret_cast<T *>(dst), len);
                           });
  }

  // Non-CPU: copy into a std::vector that stays alive until the transfer
  // event completes.
  auto data_vec = copy_r_data_to_vec<T>(data);

  auto result = client->buffer_from_host_async(data_vec->data(), dims,
                                               byte_strides_opt, dtype, device);

  if (result.event) {
    auto *raw_ptr = data_vec.release();
    result.event->on_ready([raw_ptr](PJRT_Error *error) { delete raw_ptr; });
  }

  Rcpp::XPtr<rpjrt::PJRTBuffer> buffer_xptr(result.buffer.release(), true);
  buffer_xptr.attr("class") = "PJRTBuffer";
  return buffer_xptr;
}

// Buffer creation for types whose R in-memory layout already matches the dtype
// byte-for-byte (double->f64, int->i32, integer64->i64/ui64), so no per-element
// conversion is needed.
//
// On CPU the bytes are copied into a fresh RAWSXP that backs the buffer for its
// lifetime (see make_cpu_buffer). We copy rather than alias R's own `data`
// vector so that PJRT can take a mutable lifetime alias without exposing the
// user's object to in-place donation writes from pjrt_execute().
//
// On non-CPU it is genuinely zero-copy: R's data is handed straight to PJRT and
// the R object is kept alive only until the transfer completes.
Rcpp::XPtr<rpjrt::PJRTBuffer> create_buffer_from_array_async_no_convert(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, void *data_ptr,
    const std::vector<int64_t> &dims, PJRT_Buffer_Type dtype,
    size_t element_size, bool row_major = false,
    PJRT_Device *device = nullptr) {
  int len = Rf_length(data);
  if (len == 0) {
    if (!std::any_of(dims.begin(), dims.end(),
                     [](int64_t dim) { return dim == 0; })) {
      Rcpp::stop("Data must be a non-empty vector.");
    }
  }

  auto byte_strides_opt = get_byte_strides(dims, row_major, element_size);

  if (client->is_cpu()) {
    size_t total_bytes = static_cast<size_t>(len) * element_size;
    return make_cpu_buffer(client, total_bytes, dims, byte_strides_opt, dtype,
                           device, [&](void *dst) {
                             if (total_bytes > 0)
                               std::memcpy(dst, data_ptr, total_bytes);
                           });
  }

  // Non-CPU: hand R's data straight to PJRT (zero-copy).
  auto result = client->buffer_from_host_async(data_ptr, dims, byte_strides_opt,
                                               dtype, device);

  if (result.event) {
    R_PreserveObject(data);
    result.event->on_ready(
        [data](PJRT_Error *error) { rpjrt::queue_release(data); });
  }

  Rcpp::XPtr<rpjrt::PJRTBuffer> buffer_xptr(result.buffer.release(), true);
  buffer_xptr.attr("class") = "PJRTBuffer";
  return buffer_xptr;
}

Rcpp::XPtr<rpjrt::PJRTBuffer> create_buffer_from_raw(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
    const std::vector<int64_t> &dims, PJRT_Buffer_Type dtype,
    bool row_major = false, PJRT_Device *device = nullptr) {
  auto byte_strides_opt =
      get_byte_strides(dims, row_major, sizeof_pjrt_buffer_type(dtype));

  if (client->is_cpu()) {
    // Copy the raw bytes into a fresh RAWSXP; don't alias the caller's vector.
    size_t total_bytes = static_cast<size_t>(Rf_length(data));
    return make_cpu_buffer(client, total_bytes, dims, byte_strides_opt, dtype,
                           device, [&](void *dst) {
                             if (total_bytes > 0)
                               std::memcpy(dst, RAW(data), total_bytes);
                           });
  }

  auto result = client->buffer_from_host_async(RAW(data), dims,
                                               byte_strides_opt, dtype, device);

  if (result.event) {
    R_PreserveObject(data);
    result.event->on_ready(
        [data](PJRT_Error *error) { rpjrt::queue_release(data); });
  }

  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(result.buffer.release());
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_raw(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype,
    bool row_major = false) {
  if (dtype == "f32") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_F32,
                                  row_major, device->device);
  } else if (dtype == "f64") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_F64,
                                  row_major, device->device);
  } else if (dtype == "i8") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_S8,
                                  row_major, device->device);
  } else if (dtype == "i16") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_S16,
                                  row_major, device->device);
  } else if (dtype == "i32") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_S32,
                                  row_major, device->device);
  } else if (dtype == "i64") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_S64,
                                  row_major, device->device);
  } else if (dtype == "ui8") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_U8,
                                  row_major, device->device);
  } else if (dtype == "ui16") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_U16,
                                  row_major, device->device);
  } else if (dtype == "ui32") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_U32,
                                  row_major, device->device);
  } else if (dtype == "ui64") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_U64,
                                  row_major, device->device);
  } else if (dtype == "pred") {
    return create_buffer_from_raw(client, data, dims, PJRT_Buffer_Type_PRED,
                                  row_major, device->device);
  } else {
    Rcpp::stop("Unsupported type: %s", dtype.c_str());
  }
}

// Allocate a buffer with unspecified contents. On CPU, backs the buffer with
// a RAWSXP stashed in the XPtr's protected slot so that R's GC keeps the host
// bytes alive for the buffer's lifetime and counts the memory. On other
// platforms, allocates a host vector, transfers, and releases it when the
// transfer completes.
// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_empty(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    std::vector<int64_t> dims, std::string dtype) {
  const PJRT_Buffer_Type pjrt_dtype = string_to_pjrt_buffer_type(dtype);
  const size_t element_size = sizeof_pjrt_buffer_type(pjrt_dtype);
  const int64_t numel = number_of_elements(dims);
  const size_t total_bytes = static_cast<size_t>(numel) * element_size;
  auto byte_strides_opt =
      get_byte_strides(dims, /*row_major=*/false, element_size);

  if (client->is_cpu()) {
    return make_cpu_buffer(client, total_bytes, dims, byte_strides_opt,
                           pjrt_dtype, device->device, [](void *) {});
  }

  auto data_vec = std::make_unique<std::vector<uint8_t>>(total_bytes);
  auto result = client->buffer_from_host_async(
      data_vec->data(), dims, byte_strides_opt, pjrt_dtype, device->device);
  if (result.event) {
    auto *raw_ptr = data_vec.release();
    result.event->on_ready([raw_ptr](PJRT_Error *error) { delete raw_ptr; });
  }
  Rcpp::XPtr<rpjrt::PJRTBuffer> buffer_xptr(result.buffer.release(), true);
  buffer_xptr.attr("class") = "PJRTBuffer";
  return buffer_xptr;
}

// Core conversion: raw bytes (row-major) -> R array (column-major) with type
// casting. Used by both sync and async buffer-to-host paths.
template <typename T>
SEXP raw_to_array_impl(const uint8_t *raw_data,
                       const std::vector<int64_t> &dimensions, int r_type,
                       const std::vector<int64_t> &minor_to_major) {
  const auto numel = number_of_elements(dimensions);

  if (numel == 0) {
    SEXP out = PROTECT(Rf_allocVector(r_type, 0));
    if (dimensions.size() > 0) {
      Rcpp::IntegerVector dims(dimensions.begin(), dimensions.end());
      Rf_setAttrib(out, R_DimSymbol, dims);
    }
    UNPROTECT(1);
    return out;
  }

  // Reinterpret the device-layout bytes and reorder them to logical row-major
  // (a no-op copy when the device layout is already row-major).
  const T *typed_data = reinterpret_cast<const T *>(raw_data);
  std::vector<T> temp_vec(numel);
  device_to_row_major<T>(typed_data, temp_vec.data(), dimensions,
                         minor_to_major);

  SEXP out = PROTECT(Rf_allocVector(r_type, numel));
  void *out_data;

  if (r_type == REALSXP) {
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                  std::is_same_v<T, uint32_t>) {
      // Integer dtype that can't fit in R's signed int32 -> "pseudo-double":
      // REALSXP slots carry the int64 bit pattern (the storage layout used by
      // bit64::integer64). The R caller attaches the integer64 class. Widening
      // uint32_t to int64_t is value-preserving (u32 max < 2^53).
      static_assert(sizeof(double) == sizeof(int64_t),
                    "bit64::integer64 layout requires sizeof(double) == "
                    "sizeof(int64_t)");
      int64_t *out_data_i64 = reinterpret_cast<int64_t *>(REAL(out));
      row_to_col_order<T, int64_t>(temp_vec, out_data_i64, dimensions);
    } else {
      out_data = REAL(out);
      row_to_col_order<T, double>(temp_vec, static_cast<double *>(out_data),
                                  dimensions);
    }
  } else if (r_type == INTSXP) {
    out_data = INTEGER(out);
    row_to_col_order<T, int>(temp_vec, static_cast<int *>(out_data),
                             dimensions);
  } else if (r_type == LGLSXP) {
    out_data = LOGICAL(out);
    row_to_col_order<T, int>(temp_vec, static_cast<int *>(out_data),
                             dimensions);
  } else {
    UNPROTECT(1);
    Rcpp::stop("Unsupported R type: %d", r_type);
  }

  if (dimensions.size() > 0) {
    SEXP dim_attr = PROTECT(Rf_allocVector(INTSXP, dimensions.size()));
    int *dim_data = INTEGER(dim_attr);
    std::copy(dimensions.begin(), dimensions.end(), dim_data);
    Rf_setAttrib(out, R_DimSymbol, dim_attr);
    UNPROTECT(1);
  }

  UNPROTECT(1);
  return out;
}

// [[Rcpp::export()]]
Rcpp::RawVector impl_buffer_to_raw(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                   Rcpp::XPtr<rpjrt::PJRTBuffer> buffer,
                                   bool row_major = false) {
  const auto dimensions = buffer->dimensions();
  const auto element_type = buffer->element_type();

  const auto numel = number_of_elements(dimensions);

  if (numel == 0) {
    return Rcpp::RawVector(0);
  }

  const size_t total_bytes = numel * sizeof_pjrt_buffer_type(element_type);

  Rcpp::RawVector raw_data(total_bytes);

  // The bytes from PJRT arrive in the device layout. When that layout already
  // matches what the caller wants — row-major requested and device is
  // row-major, or the layout is irrelevant (e.g. 1-D) — we can copy straight
  // into the output with no temporary or reorder.
  const auto minor_to_major = buffer->minor_to_major();
  const bool device_row_major = is_row_major(minor_to_major);
  if (format_is_irrelevant(dimensions) || (row_major && device_row_major)) {
    std::span<uint8_t> host_buffer(
        reinterpret_cast<uint8_t *>(raw_data.begin()), total_bytes);
    auto event = buffer->buffer_to_host_async(host_buffer);
    if (event) {
      event->await();
      event->check_error();
    }
    return raw_data;
  }

  // Otherwise read the device-layout bytes into a temporary, normalize them to
  // logical row-major, then emit the requested order (row-major as-is, or
  // column-major via a transpose).
  auto handle_transpose = [&](auto type_tag) {
    using T = decltype(type_tag);
    std::vector<T> device_vec(numel);
    std::span<uint8_t> host_buffer(
        reinterpret_cast<uint8_t *>(device_vec.data()), total_bytes);
    auto event = buffer->buffer_to_host_async(host_buffer);
    if (event) {
      event->await();
      event->check_error();
    }
    std::vector<T> row_major_vec(numel);
    device_to_row_major<T>(device_vec.data(), row_major_vec.data(), dimensions,
                           minor_to_major);
    if (row_major) {
      std::copy(row_major_vec.begin(), row_major_vec.end(),
                reinterpret_cast<T *>(raw_data.begin()));
    } else {
      row_to_col_order<T, T>(
          row_major_vec, reinterpret_cast<T *>(raw_data.begin()), dimensions);
    }
  };

  switch (element_type) {
    case PJRT_Buffer_Type_F32:
      handle_transpose(float{});
      break;
    case PJRT_Buffer_Type_F64:
      handle_transpose(double{});
      break;
    case PJRT_Buffer_Type_S8:
      handle_transpose(int8_t{});
      break;
    case PJRT_Buffer_Type_S16:
      handle_transpose(int16_t{});
      break;
    case PJRT_Buffer_Type_S32:
      handle_transpose(int32_t{});
      break;
    case PJRT_Buffer_Type_S64:
      handle_transpose(int64_t{});
      break;
    case PJRT_Buffer_Type_U8:
      handle_transpose(uint8_t{});
      break;
    case PJRT_Buffer_Type_U16:
      handle_transpose(uint16_t{});
      break;
    case PJRT_Buffer_Type_U32:
      handle_transpose(uint32_t{});
      break;
    case PJRT_Buffer_Type_U64:
      handle_transpose(uint64_t{});
      break;
    case PJRT_Buffer_Type_PRED:
      handle_transpose(uint8_t{});
      break;
    default:
      Rcpp::stop("Unsupported buffer element type for conversion to host.");
  }
  return raw_data;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_buffer_copy_to_device(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    Rcpp::XPtr<rpjrt::PJRTClient> dst_client, bool cross_client) {
  std::unique_ptr<rpjrt::PJRTBuffer> new_buf;
  if (cross_client) {
    // Host roundtrip: device -> host bytes -> new buffer on target client
    auto dims = buffer->dimensions();
    auto dtype = buffer->element_type();
    auto numel = number_of_elements(dims);
    size_t total_bytes = numel * sizeof_pjrt_buffer_type(dtype);

    std::vector<uint8_t> host_bytes(total_bytes);
    std::span<uint8_t> host_span(host_bytes.data(), total_bytes);
    auto event = buffer->buffer_to_host_async(host_span);
    if (event) {
      event->await();
      event->check_error();
    }

    // Row-major strides so the data lands in the same layout
    auto byte_strides =
        get_byte_strides(dims, true, sizeof_pjrt_buffer_type(dtype));
    auto result = dst_client->buffer_from_host_async(
        host_bytes.data(), dims, byte_strides, dtype, device->device);
    if (result.event) {
      result.event->await();
      result.event->check_error();
    }
    new_buf = std::move(result.buffer);
  } else {
    new_buf = buffer->copy_to_device(*device);
  }
  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(new_buf.release(), true);
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
std::string impl_client_platform(Rcpp::XPtr<rpjrt::PJRTClient> client) {
  return client->platform();
}

// [[Rcpp::export()]]
Rcpp::List impl_client_devices(Rcpp::XPtr<rpjrt::PJRTClient> client) {
  auto devs = client->devices();
  Rcpp::List out(devs.size());
  for (size_t i = 0; i < devs.size(); ++i) {
    auto dev = std::make_unique<rpjrt::PJRTDevice>(devs[i], client->api);
    Rcpp::XPtr<rpjrt::PJRTDevice> xptr(dev.release(), true);
    xptr.attr("class") = "PJRTDevice";
    out[i] = xptr;
  }
  return out;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTElementType> impl_buffer_elt_type(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  auto element_type =
      std::make_unique<rpjrt::PJRTElementType>(buffer->element_type());
  Rcpp::XPtr<rpjrt::PJRTElementType> xptr(element_type.release(), true);
  xptr.attr("class") = "PJRTElementType";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTDevice> impl_buffer_device(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  auto device = buffer->device();
  Rcpp::XPtr<rpjrt::PJRTDevice> xptr(device.release(), true);
  xptr.attr("class") = "PJRTDevice";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTMemory> impl_buffer_memory(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  auto memory = buffer->memory();
  Rcpp::XPtr<rpjrt::PJRTMemory> xptr(memory.release(), true);
  xptr.attr("class") = "PJRTMemory";
  return xptr;
}

// [[Rcpp::export()]]
std::string impl_memory_debug_string(Rcpp::XPtr<rpjrt::PJRTMemory> memory) {
  return memory->debug_string();
}

// [[Rcpp::export()]]
int impl_memory_id(Rcpp::XPtr<rpjrt::PJRTMemory> memory) {
  return memory->id();
}

// [[Rcpp::export()]]
std::string impl_memory_kind(Rcpp::XPtr<rpjrt::PJRTMemory> memory) {
  return memory->kind();
}

// [[Rcpp::export()]]
std::string impl_memory_to_string(Rcpp::XPtr<rpjrt::PJRTMemory> memory) {
  return memory->to_string();
}

// [[Rcpp::export()]]
std::string impl_dtype_as_string(
    Rcpp::XPtr<rpjrt::PJRTElementType> element_type) {
  return element_type->as_string();
}

// [[Rcpp::export()]]
Rcpp::IntegerVector impl_buffer_dimensions(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  auto dims = buffer->dimensions();
  Rcpp::IntegerVector result(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    result[i] = static_cast<int>(dims[i]);
  }
  return result;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTExecuteOptions> impl_execution_options_create(
    std::vector<int64_t> non_donatable_input_indices, int launch_id) {
  auto options = std::make_unique<rpjrt::PJRTExecuteOptions>(
      non_donatable_input_indices, launch_id);
  Rcpp::XPtr<rpjrt::PJRTExecuteOptions> xptr(options.release(), true);
  xptr.attr("class") = "PJRTExecuteOptions";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::IntegerVector impl_plugin_pjrt_api_version(
    Rcpp::XPtr<rpjrt::PJRTPlugin> plugin) {
  auto version = plugin->pjrt_api_version();
  return Rcpp::IntegerVector::create(version.first, version.second);
}

// [[Rcpp::export()]]
Rcpp::List impl_plugin_attributes(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin) {
  auto attrs = plugin->attributes();
  Rcpp::List out;
  Rcpp::CharacterVector names(attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    names[i] = attrs[i].first;
    out.push_back(attrs[i].second);
  }
  out.attr("names") = names;
  return out;
}

// [[Rcpp::export()]]
std::string impl_device_to_string(Rcpp::XPtr<rpjrt::PJRTDevice> device) {
  auto api = device->api;
  PJRT_Device_GetDescription_Args desc_args{};
  desc_args.struct_size = sizeof(PJRT_Device_GetDescription_Args);
  desc_args.device = device->device;
  check_err(api.get(), api->PJRT_Device_GetDescription_(&desc_args));

  PJRT_DeviceDescription_ToString_Args str_args{};
  str_args.struct_size = sizeof(PJRT_DeviceDescription_ToString_Args);
  str_args.device_description = desc_args.device_description;
  check_err(api.get(), api->PJRT_DeviceDescription_ToString_(&str_args));
  return std::string(str_args.to_string, str_args.to_string_size);
}

// [[Rcpp::export()]]
void impl_buffer_print(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, int max_rows,
                       int max_width, int max_rows_slice) {
  buffer_print(buffer, max_rows, max_width, max_rows_slice);
}

// [[Rcpp::export()]]
Rcpp::CharacterVector impl_format_array(SEXP data, int max_rows, int max_width,
                                        int max_rows_slice) {
  PJRT_Buffer_Type element_type;
  const void *data_ptr;
  std::vector<uint8_t> logical_data;

  if (Rf_isReal(data)) {
    element_type = PJRT_Buffer_Type_F64;
    data_ptr = REAL(data);
  } else if (Rf_isInteger(data)) {
    element_type = PJRT_Buffer_Type_S32;
    data_ptr = INTEGER(data);
  } else if (Rf_isLogical(data)) {
    int n = Rf_length(data);
    logical_data.resize(n);
    int *lgl = LOGICAL(data);
    for (int i = 0; i < n; i++) {
      logical_data[i] = static_cast<uint8_t>(lgl[i] != 0);
    }
    element_type = PJRT_Buffer_Type_PRED;
    data_ptr = logical_data.data();
  } else {
    Rcpp::stop("Unsupported R type for formatting.");
  }

  SEXP dim_attr = Rf_getAttrib(data, R_DimSymbol);
  std::vector<int64_t> dimensions;
  if (!Rf_isNull(dim_attr)) {
    int ndim = Rf_length(dim_attr);
    int *dim_ptr = INTEGER(dim_attr);
    for (int i = 0; i < ndim; i++) {
      dimensions.push_back(static_cast<int64_t>(dim_ptr[i]));
    }
  } else {
    int n = Rf_length(data);
    if (n != 1) {
      dimensions.push_back(static_cast<int64_t>(n));
    }
  }

  auto lines = buffer_to_string_lines(data_ptr, dimensions, element_type,
                                      max_rows, max_width, max_rows_slice,
                                      /*row_major=*/false);
  return Rcpp::wrap(lines);
}

// Async status functions for buffers and host data

// [[Rcpp::export()]]
bool impl_buffer_is_ready(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  bool ready = buffer->is_ready();
  if (ready) {
    rpjrt::process_pending_releases();
  }
  return ready;
}

// [[Rcpp::export()]]
void impl_buffer_await(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  buffer->await();
  rpjrt::process_pending_releases();
}

// [[Rcpp::export()]]
bool impl_host_data_is_ready(Rcpp::XPtr<rpjrt::PJRTHostData> data) {
  return data->is_ready();
}

// [[Rcpp::export()]]
void impl_host_data_await(Rcpp::XPtr<rpjrt::PJRTHostData> data) {
  data->await();
}

// Process any pending R object releases that were queued by PJRT callbacks.
// [[Rcpp::export()]]
void impl_process_pending_releases() { rpjrt::process_pending_releases(); }

// Number of objects waiting in the deferred-release queue (tests/tooling).
// [[Rcpp::export()]]
std::size_t impl_pending_release_count() {
  return rpjrt::pending_release_count();
}

// Test-only: model a completed zero-copy upload by preserving `x` and queueing
// it for deferred release, exactly as the on_ready callback does. Lets tests
// exercise the drain on CPU, where no path populates the queue naturally.
// [[Rcpp::export()]]
void impl_test_enqueue_release(SEXP x) {
  R_PreserveObject(x);
  rpjrt::queue_release(x);
}

// [[Rcpp::export()]]
SEXP impl_raw_to_array(Rcpp::XPtr<rpjrt::PJRTHostData> host_data,
                       const std::string &dtype, Rcpp::IntegerVector dims,
                       Rcpp::IntegerVector minor_to_major) {
  std::vector<int64_t> dimensions(dims.begin(), dims.end());
  std::vector<int64_t> m2m(minor_to_major.begin(), minor_to_major.end());

  // Handle null/empty data
  const uint8_t *raw_data = nullptr;
  if (host_data.get() != nullptr && !host_data->data().empty()) {
    raw_data = host_data->data().data();
  }

  if (dtype == "f32") {
    return raw_to_array_impl<float>(raw_data, dimensions, REALSXP, m2m);
  } else if (dtype == "f64") {
    return raw_to_array_impl<double>(raw_data, dimensions, REALSXP, m2m);
  } else if (dtype == "i8") {
    return raw_to_array_impl<int8_t>(raw_data, dimensions, INTSXP, m2m);
  } else if (dtype == "i16") {
    return raw_to_array_impl<int16_t>(raw_data, dimensions, INTSXP, m2m);
  } else if (dtype == "i32") {
    return raw_to_array_impl<int32_t>(raw_data, dimensions, INTSXP, m2m);
  } else if (dtype == "i64") {
    return raw_to_array_impl<int64_t>(raw_data, dimensions, REALSXP, m2m);
  } else if (dtype == "ui8") {
    return raw_to_array_impl<uint8_t>(raw_data, dimensions, INTSXP, m2m);
  } else if (dtype == "ui16") {
    return raw_to_array_impl<uint16_t>(raw_data, dimensions, INTSXP, m2m);
  } else if (dtype == "ui32") {
    // u32 -> integer64 storage: signed int32 has no headroom for ui32 values
    // >= 2^31; widen to integer64 (53 bits of headroom) so the R caller gets
    // the full unsigned magnitude. R-side classes the result as integer64.
    return raw_to_array_impl<uint32_t>(raw_data, dimensions, REALSXP, m2m);
  } else if (dtype == "ui64") {
    return raw_to_array_impl<uint64_t>(raw_data, dimensions, REALSXP, m2m);
  } else if (dtype == "pred") {
    return raw_to_array_impl<uint8_t>(raw_data, dimensions, LGLSXP, m2m);
  } else {
    Rcpp::stop("Unsupported dtype: %s", dtype.c_str());
  }
}

// [[Rcpp::export()]]
Rcpp::List impl_buffer_to_host_async(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  const auto dimensions = buffer->dimensions();
  const auto element_type = buffer->element_type();
  const auto numel = number_of_elements(dimensions);

  // Get element type as string
  rpjrt::PJRTElementType elt_type(element_type);
  std::string dtype = elt_type.as_string();

  Rcpp::IntegerVector dims(dimensions.begin(), dimensions.end());

  // The bytes arrive in the device layout; carry it through so value() can
  // reorder them to a column-major R array.
  const auto m2m_vec = buffer->minor_to_major();
  Rcpp::IntegerVector minor_to_major(m2m_vec.begin(), m2m_vec.end());

  // Handle empty buffers
  if (numel == 0) {
    auto empty_vec = std::make_unique<std::vector<uint8_t>>();
    auto empty_data =
        std::make_unique<rpjrt::PJRTHostData>(std::move(empty_vec), nullptr);
    Rcpp::XPtr<rpjrt::PJRTHostData> data_xptr(empty_data.release(), true);
    return Rcpp::List::create(Rcpp::Named("data") = data_xptr,
                              Rcpp::Named("dtype") = dtype,
                              Rcpp::Named("dims") = dims,
                              Rcpp::Named("minor_to_major") = minor_to_major);
  }

  const size_t total_bytes = numel * sizeof_pjrt_buffer_type(element_type);

  auto data_vec = std::make_unique<std::vector<uint8_t>>(total_bytes);
  std::span<uint8_t> host_buffer(data_vec->data(), total_bytes);

  // Start async copy (data will be in row-major order)
  auto event = buffer->buffer_to_host_async(host_buffer);

  // Wrap data + event in PJRTHostData (owns both)
  auto host_data = std::make_unique<rpjrt::PJRTHostData>(std::move(data_vec),
                                                         std::move(event));
  Rcpp::XPtr<rpjrt::PJRTHostData> data_xptr(host_data.release(), true);

  return Rcpp::List::create(Rcpp::Named("data") = data_xptr,
                            Rcpp::Named("dtype") = dtype,
                            Rcpp::Named("dims") = dims,
                            Rcpp::Named("minor_to_major") = minor_to_major);
}

static std::string device_to_string(PJRT_Device *device, PJRT_Api *api) {
  PJRT_Device_GetDescription_Args desc_args{};
  desc_args.struct_size = sizeof(PJRT_Device_GetDescription_Args);
  desc_args.device = device;
  check_err(api, api->PJRT_Device_GetDescription_(&desc_args));

  PJRT_DeviceDescription_ToString_Args str_args{};
  str_args.struct_size = sizeof(PJRT_DeviceDescription_ToString_Args);
  str_args.device_description = desc_args.device_description;
  check_err(api, api->PJRT_DeviceDescription_ToString_(&str_args));

  return std::string(str_args.to_string, str_args.to_string_size);
}

// [[Rcpp::export()]]
Rcpp::List impl_loaded_executable_aliases(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable) {
  const auto &aliases = executable->input_output_aliases();
  Rcpp::IntegerVector inputs(aliases.size());
  Rcpp::IntegerVector outputs(aliases.size());
  for (size_t i = 0; i < aliases.size(); ++i) {
    inputs[i] = aliases[i].input_index;
    outputs[i] = aliases[i].output_index;
  }
  return Rcpp::List::create(Rcpp::Named("input") = inputs,
                            Rcpp::Named("output") = outputs);
}

// [[Rcpp::export()]]
Rcpp::List impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input,
    Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options) {
  rpjrt::process_pending_releases();
  auto api = executable->api;
  auto exec_devices = executable->addressable_devices();

  std::vector<rpjrt::PJRTBuffer *> inputs(input.size());
  for (auto i = 0; i < input.size(); i++) {
    auto elt = input[i];
    auto buffer = Rcpp::as<Rcpp::XPtr<rpjrt::PJRTBuffer>>(elt);
    inputs[i] = buffer.get();

    auto buf_device = buffer->device();
    bool on_exec_device = false;
    for (auto *dev : exec_devices) {
      if (buf_device->device == dev) {
        on_exec_device = true;
        break;
      }
    }
    if (!on_exec_device) {
      auto buf_dev_str = device_to_string(buf_device->device, api.get());
      auto exec_dev_str = device_to_string(exec_devices[0], api.get());
      Rcpp::stop(
          "Input %d is on device '%s', but the executable was "
          "compiled for device '%s'.",
          i + 1, buf_dev_str.c_str(), exec_dev_str.c_str());
    }
  }

  // Collect the input buffer XPtrs that need to stay alive for the duration of
  // the async Execute. A CPU buffer's bytes live in a RAWSXP parked in the
  // XPtr's prot slot, which PJRT only aliases (zero-copy); the Execute reads
  // those bytes on a background thread, but nothing else keeps a *dropped*
  // input alive until the computation finishes with it. Pinning the whole XPtr
  // keeps the buffer -- and transitively its RAWSXP -- reachable, so an
  // un-awaited Execute can't read freed memory. Device inputs (CUDA/Metal) are
  // PJRT-owned and carry a NilValue prot slot, so they are skipped: PJRT
  // already defers their device-memory free until pending ops complete.
  std::vector<SEXP> input_keepalives;
  for (auto i = 0; i < input.size(); ++i) {
    SEXP xptr = VECTOR_ELT(input, i);
    if (R_ExternalPtrProtected(xptr) != R_NilValue) {
      input_keepalives.push_back(xptr);
    }
  }

  // Pin the inputs BEFORE launching the async Execute, so the keepalive is
  // already in place when PJRT's background threads begin reading the aliased
  // host bytes. R_PreserveObject runs here on the main thread; the matching
  // release is deferred to the completion event below. If Execute itself
  // throws, unpin first so the objects don't leak.
  for (SEXP k : input_keepalives) R_PreserveObject(k);

  rpjrt::AsyncExecuteResult result;
  try {
    result = executable->execute_async(inputs, *execution_options);
  } catch (...) {
    for (SEXP k : input_keepalives) R_ReleaseObject(k);
    throw;
  }

  // Wrap buffers (each already has its completion event set)
  Rcpp::List buffers(result.buffers.size());
  for (size_t i = 0; i < result.buffers.size(); ++i) {
    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(result.buffers[i].release(), true);
    xptr.attr("class") = "PJRTBuffer";
    buffers[i] = xptr;
  }

  // For each input->output alias declared in the program, migrate the RAWSXP
  // keepalive from the donated input XPtr to the aliased output XPtr, and null
  // the donated input's PJRT_Buffer* so its finalizer is a no-op (PJRT already
  // invalidated the handle during Execute).
  //
  // We only migrate when PJRT *actually* donated (confirmed via is_deleted),
  // because tf.aliasing_output lowers to a may-alias: PJRT may copy instead of
  // donate, leaving the input valid. Migrating unconditionally in the copy case
  // would null a live buffer — leaking device memory and double-freeing it.
  const auto &aliases = executable->input_output_aliases();
  for (const auto &alias : aliases) {
    if (alias.input_index < 0 ||
        static_cast<size_t>(alias.input_index) >=
            static_cast<size_t>(input.size()) ||
        alias.output_index < 0 ||
        static_cast<size_t>(alias.output_index) >=
            static_cast<size_t>(buffers.size())) {
      continue;
    }
    SEXP in_xptr_sexp = VECTOR_ELT(input, alias.input_index);
    auto *in_buf =
        static_cast<rpjrt::PJRTBuffer *>(R_ExternalPtrAddr(in_xptr_sexp));
    if (!in_buf->is_deleted()) continue;

    SEXP out_xptr_sexp = VECTOR_ELT(buffers, alias.output_index);
    SEXP keepalive = R_ExternalPtrProtected(in_xptr_sexp);
    R_SetExternalPtrProtected(out_xptr_sexp, keepalive);
    R_SetExternalPtrProtected(in_xptr_sexp, R_NilValue);
    in_buf->buffer = nullptr;
  }

  // Release the input keepalives (pinned before Execute, above) once the
  // execution has finished reading all inputs. Without the pin, a dropped
  // zero-copy input could be collected -- freeing its backing RAWSXP -- while
  // the async Execute is still reading it (use-after-free). The completion
  // event fires on a PJRT thread and only enqueues the release; the actual
  // R_ReleaseObject runs later on the main thread via the deferred-release
  // queue. The PJRTEvent wrapper is destroyed when `result` goes out of scope,
  // but the registered callback still fires (see PJRTEvent::on_ready).
  if (!input_keepalives.empty()) {
    if (result.complete_event) {
      result.complete_event->on_ready(
          [keepalives = std::move(input_keepalives)](PJRT_Error * /*error*/) {
            for (SEXP k : keepalives) rpjrt::queue_release(k);
          });
    } else {
      // No completion event means nothing will signal when Execute is done with
      // the inputs; release the keepalives now so they don't leak.
      for (SEXP k : input_keepalives) rpjrt::queue_release(k);
    }
  }

  return buffers;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_integer(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype) {
  if (dtype == "i8") {
    return create_buffer_from_array_async<int8_t>(
        client, data, dims, PJRT_Buffer_Type_S8, false, device->device);
  } else if (dtype == "i16") {
    return create_buffer_from_array_async<int16_t>(
        client, data, dims, PJRT_Buffer_Type_S16, false, device->device);
  } else if (dtype == "i32") {
    // Zero-copy optimization: use R's integer data directly (R int = 32-bit)
    static_assert(sizeof(int) == sizeof(int32_t),
                  "R int must be 32-bit for zero-copy");
    return create_buffer_from_array_async_no_convert(
        client, data, INTEGER(data), dims, PJRT_Buffer_Type_S32,
        sizeof(int32_t), false, device->device);
  } else if (dtype == "i64") {
    return create_buffer_from_array_async<int64_t>(
        client, data, dims, PJRT_Buffer_Type_S64, false, device->device);
  } else if (dtype == "ui8") {
    return create_buffer_from_array_async<uint8_t>(
        client, data, dims, PJRT_Buffer_Type_U8, false, device->device);
  } else if (dtype == "ui16") {
    return create_buffer_from_array_async<uint16_t>(
        client, data, dims, PJRT_Buffer_Type_U16, false, device->device);
  } else if (dtype == "ui32") {
    return create_buffer_from_array_async<uint32_t>(
        client, data, dims, PJRT_Buffer_Type_U32, false, device->device);
  } else if (dtype == "ui64") {
    return create_buffer_from_array_async<uint64_t>(
        client, data, dims, PJRT_Buffer_Type_U64, false, device->device);
  } else if (dtype == "f32") {
    return create_buffer_from_array_async<float>(
        client, data, dims, PJRT_Buffer_Type_F32, false, device->device);
  } else if (dtype == "f64") {
    return create_buffer_from_array_async<double>(
        client, data, dims, PJRT_Buffer_Type_F64, false, device->device);
  } else {
    Rcpp::stop("Unsupported type: %s", dtype.c_str());
  }
}

// bit64::integer64 stores int64 values inside a REALSXP (8 bytes per slot),
// so we can hand the underlying buffer to PJRT zero-copy as int64.
// The bit pattern is identical for signed/unsigned 64-bit ints, so the same
// data can be uploaded as either S64 or U64.
// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_integer64(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype) {
  static_assert(sizeof(double) == sizeof(int64_t),
                "bit64::integer64 zero-copy requires sizeof(double) == "
                "sizeof(int64_t)");
  PJRT_Buffer_Type buffer_type;
  if (dtype == "i64") {
    buffer_type = PJRT_Buffer_Type_S64;
  } else if (dtype == "ui64") {
    buffer_type = PJRT_Buffer_Type_U64;
  } else {
    Rcpp::stop("Unsupported type: %s", dtype.c_str());
  }
  return create_buffer_from_array_async_no_convert(
      client, data, REAL(data), dims, buffer_type, sizeof(int64_t), false,
      device->device);
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_logical(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype) {
  if (dtype == "pred") {
    return create_buffer_from_array_async<uint8_t>(
        client, data, dims, PJRT_Buffer_Type_PRED, false, device->device);
  } else {
    Rcpp::stop("Unsupported type: %s", dtype.c_str());
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_double(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype) {
  if (dtype == "f32") {
    return create_buffer_from_array_async<float>(
        client, data, dims, PJRT_Buffer_Type_F32, false, device->device);
  } else if (dtype == "f64") {
    // Zero-copy optimization: use R's double data directly (no type conversion
    // needed)
    return create_buffer_from_array_async_no_convert(
        client, data, REAL(data), dims, PJRT_Buffer_Type_F64, sizeof(double),
        false, device->device);
  } else if (dtype == "pred") {
    Rcpp::LogicalVector data_conv = Rcpp::as<Rcpp::LogicalVector>(data);
    return impl_client_buffer_from_logical(client, device, data_conv, dims,
                                           dtype);
  } else {
    Rcpp::IntegerVector data_conv = Rcpp::as<Rcpp::IntegerVector>(data);
    return impl_client_buffer_from_integer(client, device, data_conv, dims,
                                           dtype);
  }
}

#include <Rcpp.h>

#include <algorithm>
#include <cstdint>

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
    Rcpp::XPtr<rpjrt::PJRTClient> client,
    Rcpp::XPtr<rpjrt::PJRTProgram> program,
    Rcpp::XPtr<rpjrt::PJRTCompileOptions> compile_options) {
  auto executable = client->compile(*program, *compile_options);
  Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> xptr(executable.release(), true);

  xptr.attr("class") = "PJRTLoadedExecutable";
  return xptr;
}

// Helper to copy R data to a heap-allocated vector with type conversion
// The generic type T indicates the target type for PJRT
template <typename T>
std::unique_ptr<std::vector<T>> copy_r_data_to_vec(SEXP data) {
  int len = Rf_length(data);
  auto vec = std::make_unique<std::vector<T>>(len);

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (TYPEOF(data) == REALSXP) {
      std::copy(REAL(data), REAL(data) + len, vec->data());
    } else if (TYPEOF(data) == INTSXP) {
      for (int i = 0; i < len; ++i) {
        (*vec)[i] = static_cast<T>(INTEGER(data)[i]);
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
    std::copy(INTEGER(data), INTEGER(data) + len, vec->data());
  } else if constexpr (std::is_same_v<T, bool>) {
    std::copy(LOGICAL(data), LOGICAL(data) + len, vec->data());
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    if (TYPEOF(data) == LGLSXP) {
      for (int i = 0; i < len; ++i) {
        (*vec)[i] = LOGICAL(data)[i] ? 1 : 0;
      }
    } else if (TYPEOF(data) == INTSXP) {
      std::copy(INTEGER(data), INTEGER(data) + len, vec->data());
    } else {
      Rcpp::stop("Unsupported R type: %d", TYPEOF(data));
    }
  } else {
    Rcpp::stop("Unsupported R type: %d", TYPEOF(data));
  }

  return vec;
}

// Async buffer creation - handles data lifetime internally via on_ready callback
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

  // Copy data to heap-allocated vector (stays alive until event completes)
  auto data_vec = copy_r_data_to_vec<T>(data);
  auto byte_strides_opt = get_byte_strides(dims, row_major, sizeof(T));

  // Start async transfer
  auto result = client->buffer_from_host_async(data_vec->data(), dims,
                                               byte_strides_opt, dtype, device);

  // Keep data alive until transfer completes
  if (result.event) {
    auto *raw_ptr = data_vec.release();
    result.event->on_ready(
        [raw_ptr](PJRT_Error *error) { delete raw_ptr; });
  }
  // If no event, data_vec is freed here (transfer already complete)

  Rcpp::XPtr<rpjrt::PJRTBuffer> buffer_xptr(result.buffer.release(), true);
  buffer_xptr.attr("class") = "PJRTBuffer";
  return buffer_xptr;
}

// Zero-copy async buffer creation for matching types (double->f64, int->i32)
// Uses R's data directly without copying, preserving the R object until
// transfer completes
Rcpp::XPtr<rpjrt::PJRTBuffer> create_buffer_from_array_async_zerocopy(
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

  // Start async transfer using R's data directly
  auto result = client->buffer_from_host_async(data_ptr, dims, byte_strides_opt,
                                               dtype, device);

  // Keep R object alive until transfer completes
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
  auto result = client->buffer_from_host_async(RAW(data), dims,
                                                byte_strides_opt, dtype, device);
  if (result.event) {
    result.event->await();
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

// Core conversion: raw bytes (row-major) -> R array (column-major) with type
// casting. Used by both sync and async buffer-to-host paths.
template <typename T>
SEXP raw_to_array_impl(const uint8_t *raw_data,
                       const std::vector<int64_t> &dimensions, int r_type) {
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

  // Reinterpret raw bytes as typed data and copy to vector for transpose
  const T *typed_data = reinterpret_cast<const T *>(raw_data);
  std::vector<T> temp_vec(typed_data, typed_data + numel);

  SEXP out = PROTECT(Rf_allocVector(r_type, numel));
  void *out_data;

  if (r_type == REALSXP) {
    out_data = REAL(out);
    row_to_col_order<T, double>(temp_vec, static_cast<double *>(out_data),
                                dimensions);
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

  if (row_major || format_is_irrelevant(dimensions)) {
    // optimization: we don't need a temporary buffer because
    // 1. We don't need to cast the data
    // 2. We don't need to transpose the data
    std::span<uint8_t> host_buffer(
        reinterpret_cast<uint8_t *>(raw_data.begin()), total_bytes);
    auto event = buffer->buffer_to_host_async(host_buffer);
    if (event) {
      event->await();
      event->check_error();
    }
    return raw_data;
  }

  auto handle_transpose = [&](auto type_tag) {
    using T = decltype(type_tag);
    std::vector<T> temp_vec(numel);
    std::span<uint8_t> host_buffer(reinterpret_cast<uint8_t *>(temp_vec.data()),
                                   total_bytes);
    auto event = buffer->buffer_to_host_async(host_buffer);
    if (event) {
      event->await();
      event->check_error();
    }
    row_to_col_order<T, T>(temp_vec, reinterpret_cast<T *>(raw_data.begin()),
                           dimensions);
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
std::string impl_device_platform(Rcpp::XPtr<rpjrt::PJRTDevice> device) {
  auto api = device->api;
  PJRT_Device_GetDescription_Args desc_args{};
  desc_args.struct_size = sizeof(PJRT_Device_GetDescription_Args);
  desc_args.device = device->device;
  check_err(api.get(), api->PJRT_Device_GetDescription_(&desc_args));

  PJRT_DeviceDescription_Kind_Args kind_args{};
  kind_args.struct_size = sizeof(PJRT_DeviceDescription_Kind_Args);
  kind_args.device_description = desc_args.device_description;
  check_err(api.get(), api->PJRT_DeviceDescription_Kind_(&kind_args));

  std::string kind(kind_args.device_kind, kind_args.device_kind_size);
  // Convert to lowercase for consistency
  std::transform(kind.begin(), kind.end(), kind.begin(), ::tolower);
  return kind;
}

// [[Rcpp::export()]]
void impl_buffer_print(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, int max_rows,
                       int max_width, int max_rows_slice) {
  buffer_print(buffer, max_rows, max_width, max_rows_slice);
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

// [[Rcpp::export()]]
SEXP impl_raw_to_array(Rcpp::XPtr<rpjrt::PJRTHostData> host_data,
                       const std::string &dtype, Rcpp::IntegerVector dims) {
  std::vector<int64_t> dimensions(dims.begin(), dims.end());

  // Handle null/empty data
  const uint8_t *raw_data = nullptr;
  if (host_data.get() != nullptr && !host_data->data().empty()) {
    raw_data = host_data->data().data();
  }

  if (dtype == "f32") {
    return raw_to_array_impl<float>(raw_data, dimensions, REALSXP);
  } else if (dtype == "f64") {
    return raw_to_array_impl<double>(raw_data, dimensions, REALSXP);
  } else if (dtype == "i8") {
    return raw_to_array_impl<int8_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "i16") {
    return raw_to_array_impl<int16_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "i32") {
    return raw_to_array_impl<int32_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "i64") {
    return raw_to_array_impl<int64_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "ui8") {
    return raw_to_array_impl<uint8_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "ui16") {
    return raw_to_array_impl<uint16_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "ui32") {
    return raw_to_array_impl<uint32_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "ui64") {
    return raw_to_array_impl<uint64_t>(raw_data, dimensions, INTSXP);
  } else if (dtype == "pred") {
    return raw_to_array_impl<uint8_t>(raw_data, dimensions, LGLSXP);
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

  // Handle empty buffers
  if (numel == 0) {
    auto empty_vec = std::make_unique<std::vector<uint8_t>>();
    auto empty_data = std::make_unique<rpjrt::PJRTHostData>(
        std::move(empty_vec), nullptr);
    Rcpp::XPtr<rpjrt::PJRTHostData> data_xptr(empty_data.release(), true);
    return Rcpp::List::create(
        Rcpp::Named("data") = data_xptr,
        Rcpp::Named("dtype") = dtype, Rcpp::Named("dims") = dims);
  }

  const size_t total_bytes = numel * sizeof_pjrt_buffer_type(element_type);

  auto data_vec = std::make_unique<std::vector<uint8_t>>(total_bytes);
  std::span<uint8_t> host_buffer(data_vec->data(), total_bytes);

  // Start async copy (data will be in row-major order)
  auto event = buffer->buffer_to_host_async(host_buffer);

  // Wrap data + event in PJRTHostData (owns both)
  auto host_data =
      std::make_unique<rpjrt::PJRTHostData>(std::move(data_vec), std::move(event));
  Rcpp::XPtr<rpjrt::PJRTHostData> data_xptr(host_data.release(), true);

  return Rcpp::List::create(
      Rcpp::Named("data") = data_xptr,
      Rcpp::Named("dtype") = dtype, Rcpp::Named("dims") = dims);
}

// [[Rcpp::export()]]
Rcpp::List impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input,
    Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options) {
  std::vector<rpjrt::PJRTBuffer *> inputs(input.size());
  for (auto i = 0; i < input.size(); i++) {
    auto elt = input[i];
    auto buffer = Rcpp::as<Rcpp::XPtr<rpjrt::PJRTBuffer>>(elt);
    inputs[i] = buffer.get();
  }

  auto result = executable->execute_async(inputs, *execution_options);

  // Wrap buffers (each already has its completion event set)
  Rcpp::List buffers(result.buffers.size());
  for (size_t i = 0; i < result.buffers.size(); ++i) {
    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(result.buffers[i].release(), true);
    xptr.attr("class") = "PJRTBuffer";
    buffers[i] = xptr;
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
    return create_buffer_from_array_async_zerocopy(
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
    return create_buffer_from_array_async_zerocopy(
        client, data, REAL(data), dims, PJRT_Buffer_Type_F64, sizeof(double),
        false, device->device);
  } else if (dtype == "pred") {
    Rcpp::LogicalVector data_conv = Rcpp::as<Rcpp::LogicalVector>(data);
    return impl_client_buffer_from_logical(client, device, data_conv,
                                           dims, dtype);
  } else {
    Rcpp::IntegerVector data_conv = Rcpp::as<Rcpp::IntegerVector>(data);
    return impl_client_buffer_from_integer(client, device, data_conv,
                                           dims, dtype);
  }
}

#include <Rcpp.h>

#include <algorithm>
#include <cstdint>

#include "buffer.h"
#include "client.h"
#include "pjrt_types.h"
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
    Rcpp::XPtr<rpjrt::PJRTPlugin> plugin) {
  auto client = plugin->client_create();
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

// Helper template function for buffer creation from R data
template <typename T>
Rcpp::XPtr<rpjrt::PJRTBuffer> create_buffer_from_r_data(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
    const std::vector<int64_t> &dims, PJRT_Buffer_Type dtype,
    bool row_major = false) {
  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }

  std::vector<T> buffer(len);

  // Copy data based on R type
  // copy implicitly casts the types
  if (TYPEOF(data) == RAWSXP) {
    std::copy(RAW(data), RAW(data) + len,
              reinterpret_cast<uint8_t *>(buffer.data()));
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    std::copy(REAL(data), REAL(data) + len, buffer.data());
  } else if constexpr (std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, int16_t> ||
                       std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, int64_t> ||
                       std::is_same_v<T, uint16_t> ||
                       std::is_same_v<T, uint32_t> ||
                       std::is_same_v<T, uint64_t>) {
    std::copy(INTEGER(data), INTEGER(data) + len, buffer.data());
  } else if constexpr (std::is_same_v<T, bool>) {
    std::copy(LOGICAL(data), LOGICAL(data) + len, buffer.data());
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    // Special case for uint8_t: could be logical data or unsigned 8-bit
    // integers
    if (TYPEOF(data) == LGLSXP) {
      // Convert logical data to uint8_t
      for (int i = 0; i < len; ++i) {
        buffer[i] = LOGICAL(data)[i] ? 1 : 0;
      }
    } else {
      // Regular integer data
      std::copy(INTEGER(data), INTEGER(data) + len, buffer.data());
    }
  }

  std::optional<std::vector<int64_t>> byte_strides_opt;
  if (!row_major) {
    // For column-major (R's native format), we need to specify byte_strides
    auto byte_strides = std::vector<int64_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      byte_strides[i] = sizeof(T);
      for (size_t j = 0; j < i; ++j) {
        byte_strides[i] *= dims[j];
      }
    }
    byte_strides_opt = byte_strides;
  }

  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
      client->buffer_from_host(buffer.data(), dims, byte_strides_opt, dtype)
          .release());
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_double(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims,
    std::string type) {
  if (type == "f32") {
    return create_buffer_from_r_data<float>(client, data, dims,
                                            PJRT_Buffer_Type_F32);
  } else if (type == "f64") {
    return create_buffer_from_r_data<double>(client, data, dims,
                                             PJRT_Buffer_Type_F64);
  } else {
    Rcpp::stop("Unsupported floating point type: %s", type.c_str());
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_integer(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims,
    std::string type) {
  if (type == "s8") {
    return create_buffer_from_r_data<int8_t>(client, data, dims,
                                             PJRT_Buffer_Type_S8);
  } else if (type == "s16") {
    return create_buffer_from_r_data<int16_t>(client, data, dims,
                                              PJRT_Buffer_Type_S16);
  } else if (type == "s32") {
    return create_buffer_from_r_data<int32_t>(client, data, dims,
                                              PJRT_Buffer_Type_S32);
  } else if (type == "s64") {
    return create_buffer_from_r_data<int64_t>(client, data, dims,
                                              PJRT_Buffer_Type_S64);
  } else if (type == "u8") {
    return create_buffer_from_r_data<uint8_t>(client, data, dims,
                                              PJRT_Buffer_Type_U8);
  } else if (type == "u16") {
    return create_buffer_from_r_data<uint16_t>(client, data, dims,
                                               PJRT_Buffer_Type_U16);
  } else if (type == "u32") {
    return create_buffer_from_r_data<uint32_t>(client, data, dims,
                                               PJRT_Buffer_Type_U32);
  } else if (type == "u64") {
    return create_buffer_from_r_data<uint64_t>(client, data, dims,
                                               PJRT_Buffer_Type_U64);
  } else {
    Rcpp::stop("Unsupported type: %s", type.c_str());
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_logical(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims,
    std::string type) {
  if (type == "pred") {
    return create_buffer_from_r_data<uint8_t>(client, data, dims,
                                              PJRT_Buffer_Type_PRED);
  } else {
    Rcpp::stop("Unsupported type: %s", type.c_str());
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_raw(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims,
    std::string type, bool row_major = false) {
  if (TYPEOF(data) != RAWSXP) {
    Rcpp::stop("Data must be a raw vector");
  }

  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty raw vector.");
  }

  // Support all the same types as the other buffer functions
  if (type == "f32") {
    return create_buffer_from_r_data<float>(client, data, dims,
                                            PJRT_Buffer_Type_F32, row_major);
  } else if (type == "f64") {
    return create_buffer_from_r_data<double>(client, data, dims,
                                             PJRT_Buffer_Type_F64, row_major);
  } else if (type == "s8") {
    return create_buffer_from_r_data<int8_t>(client, data, dims,
                                             PJRT_Buffer_Type_S8, row_major);
  } else if (type == "s16") {
    return create_buffer_from_r_data<int16_t>(client, data, dims,
                                              PJRT_Buffer_Type_S16, row_major);
  } else if (type == "s32") {
    return create_buffer_from_r_data<int32_t>(client, data, dims,
                                              PJRT_Buffer_Type_S32, row_major);
  } else if (type == "s64") {
    return create_buffer_from_r_data<int64_t>(client, data, dims,
                                              PJRT_Buffer_Type_S64, row_major);
  } else if (type == "u8") {
    return create_buffer_from_r_data<uint8_t>(client, data, dims,
                                              PJRT_Buffer_Type_U8, row_major);
  } else if (type == "u16") {
    return create_buffer_from_r_data<uint16_t>(client, data, dims,
                                               PJRT_Buffer_Type_U16, row_major);
  } else if (type == "u32") {
    return create_buffer_from_r_data<uint32_t>(client, data, dims,
                                               PJRT_Buffer_Type_U32, row_major);
  } else if (type == "u64") {
    return create_buffer_from_r_data<uint64_t>(client, data, dims,
                                               PJRT_Buffer_Type_U64, row_major);
  } else if (type == "pred") {
    return create_buffer_from_r_data<uint8_t>(client, data, dims,
                                              PJRT_Buffer_Type_PRED, row_major);
  } else {
    Rcpp::stop("Unsupported type for raw data: %s", type.c_str());
  }
}

// Helper template function for buffer to host conversion
template <typename T>
SEXP convert_buffer_to_r(Rcpp::XPtr<rpjrt::PJRTClient> client,
                         Rcpp::XPtr<rpjrt::PJRTBuffer> buffer,
                         const std::vector<int64_t> &dimensions, int r_type) {
  const auto numel = std::accumulate(dimensions.begin(), dimensions.end(), 1,
                                     std::multiplies<int64_t>());

  // TODO(performance): Can we avoid some copies here.
  // We do two copies:
  // device -> host -> R
  std::vector<T> buffer_data(numel);
  std::span<uint8_t> host_buffer(
      reinterpret_cast<uint8_t *>(buffer_data.data()), numel * sizeof(T));

  client->buffer_to_host(*buffer, host_buffer);

  SEXP out = PROTECT(Rf_allocVector(r_type, numel));
  void *out_data;

  if (r_type == REALSXP) {
    out_data = REAL(out);
    row_to_col_order<T, double>(buffer_data, static_cast<double *>(out_data),
                                dimensions);
  } else if (r_type == INTSXP) {
    out_data = INTEGER(out);
    row_to_col_order<T, int>(buffer_data, static_cast<int *>(out_data),
                             dimensions);
  } else if (r_type == LGLSXP) {
    out_data = LOGICAL(out);
    row_to_col_order<T, int>(buffer_data, static_cast<int *>(out_data),
                             dimensions);
  }

  // Set dimensions only once
  if (dimensions.size() > 0) {
    SEXP dim_attr = PROTECT(Rf_allocVector(INTSXP, dimensions.size()));
    int *dim_data = INTEGER(dim_attr);
    std::copy(dimensions.begin(), dimensions.end(), dim_data);
    Rf_setAttrib(out, R_DimSymbol, dim_attr);
    UNPROTECT(1);  // Unprotect dim_attr
  }

  UNPROTECT(1);  // Unprotect out
  return out;
}

// [[Rcpp::export()]]
SEXP impl_client_buffer_to_host(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  const auto dimensions = buffer->dimensions();
  const auto element_type = buffer->element_type();

  switch (element_type) {
    case PJRT_Buffer_Type_F32:
      return convert_buffer_to_r<float>(client, buffer, dimensions, REALSXP);
    case PJRT_Buffer_Type_F64:
      return convert_buffer_to_r<double>(client, buffer, dimensions, REALSXP);
    case PJRT_Buffer_Type_S8:
      return convert_buffer_to_r<int8_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_S16:
      return convert_buffer_to_r<int16_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_S32:
      return convert_buffer_to_r<int32_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_S64:
      return convert_buffer_to_r<int64_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_U8:
      return convert_buffer_to_r<uint8_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_U16:
      return convert_buffer_to_r<uint16_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_U32:
      return convert_buffer_to_r<uint32_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_U64:
      return convert_buffer_to_r<uint64_t>(client, buffer, dimensions, INTSXP);
    case PJRT_Buffer_Type_PRED:
      return convert_buffer_to_r<uint8_t>(client, buffer, dimensions, LGLSXP);
    default:
      Rcpp::stop("Unsupported buffer element type for conversion to host.");
  }
}

// [[Rcpp::export()]]
Rcpp::RawVector impl_client_buffer_to_raw(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                          Rcpp::XPtr<rpjrt::PJRTBuffer> buffer,
                                          bool row_major = false) {
  const auto dimensions = buffer->dimensions();
  const auto element_type = buffer->element_type();

  const auto numel = std::accumulate(dimensions.begin(), dimensions.end(), 1,
                                     std::multiplies<int64_t>());

  // Get the size of each element in bytes
  size_t element_size = 0;
  switch (element_type) {
    case PJRT_Buffer_Type_F32:
      element_size = sizeof(float);
      break;
    case PJRT_Buffer_Type_F64:
      element_size = sizeof(double);
      break;
    case PJRT_Buffer_Type_S8:
      element_size = sizeof(int8_t);
      break;
    case PJRT_Buffer_Type_S16:
      element_size = sizeof(int16_t);
      break;
    case PJRT_Buffer_Type_S32:
      element_size = sizeof(int32_t);
      break;
    case PJRT_Buffer_Type_S64:
      element_size = sizeof(int64_t);
      break;
    case PJRT_Buffer_Type_U8:
      element_size = sizeof(uint8_t);
      break;
    case PJRT_Buffer_Type_U16:
      element_size = sizeof(uint16_t);
      break;
    case PJRT_Buffer_Type_U32:
      element_size = sizeof(uint32_t);
      break;
    case PJRT_Buffer_Type_U64:
      element_size = sizeof(uint64_t);
      break;
    case PJRT_Buffer_Type_PRED:
      element_size = sizeof(uint8_t);
      break;
    default:
      Rcpp::stop("Unsupported buffer element type for conversion to raw.");
  }

  const size_t total_bytes = numel * element_size;

  Rcpp::RawVector raw_data(total_bytes);

  // Handle each type appropriately with proper transposition
  switch (element_type) {
    case PJRT_Buffer_Type_F32: {
      std::vector<float> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<float, float>(temp_buffer,
                                  reinterpret_cast<float *>(raw_data.begin()),
                                  dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_F64: {
      std::vector<double> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<double, double>(
          temp_buffer, reinterpret_cast<double *>(raw_data.begin()), dimensions,
          row_major);
      break;
    }
    case PJRT_Buffer_Type_S8: {
      std::vector<int8_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<int8_t, int8_t>(
          temp_buffer, reinterpret_cast<int8_t *>(raw_data.begin()), dimensions,
          row_major);
      break;
    }
    case PJRT_Buffer_Type_S16: {
      std::vector<int16_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<int16_t, int16_t>(
          temp_buffer, reinterpret_cast<int16_t *>(raw_data.begin()),
          dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_S32: {
      std::vector<int32_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<int32_t, int32_t>(
          temp_buffer, reinterpret_cast<int32_t *>(raw_data.begin()),
          dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_S64: {
      std::vector<int64_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<int64_t, int64_t>(
          temp_buffer, reinterpret_cast<int64_t *>(raw_data.begin()),
          dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_U8: {
      std::vector<uint8_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(temp_buffer.data(), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<uint8_t, uint8_t>(temp_buffer, raw_data.begin(), dimensions,
                                      row_major);
      break;
    }
    case PJRT_Buffer_Type_U16: {
      std::vector<uint16_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<uint16_t, uint16_t>(
          temp_buffer, reinterpret_cast<uint16_t *>(raw_data.begin()),
          dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_U32: {
      std::vector<uint32_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<uint32_t, uint32_t>(
          temp_buffer, reinterpret_cast<uint32_t *>(raw_data.begin()),
          dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_U64: {
      std::vector<uint64_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(
          reinterpret_cast<uint8_t *>(temp_buffer.data()), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<uint64_t, uint64_t>(
          temp_buffer, reinterpret_cast<uint64_t *>(raw_data.begin()),
          dimensions, row_major);
      break;
    }
    case PJRT_Buffer_Type_PRED: {
      std::vector<uint8_t> temp_buffer(numel);
      std::span<uint8_t> host_buffer(temp_buffer.data(), total_bytes);
      client->buffer_to_host(*buffer, host_buffer);
      convert_order<uint8_t, uint8_t>(temp_buffer, raw_data.begin(), dimensions,
                                      row_major);
      break;
    }
    default:
      Rcpp::stop("Unsupported buffer element type for conversion to raw.");
  }

  return raw_data;
}

// [[Rcpp::export()]]
std::string impl_client_platform_name(Rcpp::XPtr<rpjrt::PJRTClient> client) {
  return client->platform_name();
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input,
    Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options) {
  std::vector<rpjrt::PJRTBuffer *> inputs(input.size());
  for (auto i = 0; i < input.size(); i++) {
    auto elt = input[i];
    auto buffer = Rcpp::as<Rcpp::XPtr<rpjrt::PJRTBuffer>>(elt);
    inputs[i] = buffer.get();
  }

  auto outs = executable->execute(inputs, *execution_options);
  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(outs[0].release(), true);
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTElementType> impl_buffer_element_type(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  auto element_type =
      std::make_unique<rpjrt::PJRTElementType>(buffer->element_type());
  Rcpp::XPtr<rpjrt::PJRTElementType> xptr(element_type.release(), true);
  xptr.attr("class") = "PJRTElementType";
  return xptr;
}

// [[Rcpp::export]]
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
std::string impl_element_type_as_string(
    Rcpp::XPtr<rpjrt::PJRTElementType> element_type) {
  return element_type->as_string();
}

// [[Rcpp::export()]]
std::vector<int64_t> impl_buffer_dimensions(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  return buffer->dimensions();
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

#include <Rcpp.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>

#include "buffer.h"
#include "client.h"
#include "pjrt_types.h"

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTPlugin> impl_plugin_load(const std::string &path) {
  auto ptr = std::make_unique<rpjrt::PJRTPlugin>(path);
  Rcpp::XPtr<rpjrt::PJRTPlugin> xptr(ptr.release(), true);
  xptr.attr("class") = "PJRTPlugin";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTClient>
impl_plugin_client_create(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin) {
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
Rcpp::XPtr<rpjrt::PJRTBuildOptions>
impl_build_options_create(const int num_replicas = 1,
                          const int num_partitions = 1,
                          const int device_ordinal = -1) {
  auto build_options = std::make_unique<rpjrt::PJRTBuildOptions>(
      num_replicas, num_partitions, device_ordinal);
  Rcpp::XPtr<rpjrt::PJRTBuildOptions> xptr(build_options.release(), true);
  xptr.attr("class") = "PJRTBuildOptions";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTCompileOptions>
impl_compile_options_create(Rcpp::XPtr<rpjrt::PJRTBuildOptions> build_options) {
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
Rcpp::XPtr<rpjrt::PJRTBuffer>
create_buffer_from_r_data(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                          const std::vector<int64_t> &dims,
                          PJRT_Buffer_Type dtype) {
  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }

  std::vector<T> buffer(len);

  // Copy data based on R type
  // copy implicitly casts the types
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    std::copy(REAL(data), REAL(data) + len, buffer.data());
  } else if constexpr (std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, int16_t> ||
                       std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, int64_t>) {
    std::copy(INTEGER(data), INTEGER(data) + len, buffer.data());
  } else if constexpr (std::is_same_v<T, uint8_t> ||
                       std::is_same_v<T, uint16_t> ||
                       std::is_same_v<T, uint32_t> ||
                       std::is_same_v<T, uint64_t>) {
    std::copy(INTEGER(data), INTEGER(data) + len, buffer.data());
  } else if constexpr (std::is_same_v<T, bool>) {
    std::copy(LOGICAL(data), LOGICAL(data) + len, buffer.data());
  }

  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
      client->buffer_from_host(buffer.data(), dims, dtype).release());
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_double(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                               std::vector<int64_t> dims, int precision) {
  switch (precision) {
  case 32:
    return create_buffer_from_r_data<float>(client, data, dims,
                                            PJRT_Buffer_Type_F32);
  case 64:
    return create_buffer_from_r_data<double>(client, data, dims,
                                             PJRT_Buffer_Type_F64);
  default:
    Rcpp::stop("Unsupported floating point precision: %d. Only 32 (single) "
               "and 64 (double) are supported.",
               precision);
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_integer(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                                std::vector<int64_t> dims, int precision,
                                bool is_signed) {
  if (is_signed) {
    switch (precision) {
    case 8:
      return create_buffer_from_r_data<int8_t>(client, data, dims,
                                               PJRT_Buffer_Type_S8);
    case 16:
      return create_buffer_from_r_data<int16_t>(client, data, dims,
                                                PJRT_Buffer_Type_S16);
    case 32:
      return create_buffer_from_r_data<int32_t>(client, data, dims,
                                                PJRT_Buffer_Type_S32);
    case 64:
      return create_buffer_from_r_data<int64_t>(client, data, dims,
                                                PJRT_Buffer_Type_S64);
    default:
      Rcpp::stop("Unsupported signed integer precision: %d. Only 8, 16, 32, 64 "
                 "are supported.",
                 precision);
    }
  } else {
    switch (precision) {
    case 8:
      return create_buffer_from_r_data<uint8_t>(client, data, dims,
                                                PJRT_Buffer_Type_U8);
    case 16:
      return create_buffer_from_r_data<uint16_t>(client, data, dims,
                                                 PJRT_Buffer_Type_U16);
    case 32:
      return create_buffer_from_r_data<uint32_t>(client, data, dims,
                                                 PJRT_Buffer_Type_U32);
    case 64:
      return create_buffer_from_r_data<uint64_t>(client, data, dims,
                                                 PJRT_Buffer_Type_U64);
    default:
      Rcpp::stop("Unsupported unsigned integer precision: %d. Only 8, 16, 32, "
                 "64 are supported.",
                 precision);
    }
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_logical(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                                std::vector<int64_t> dims) {
  return create_buffer_from_r_data<uint8_t>(client, data, dims,
                                            PJRT_Buffer_Type_PRED);
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
    std::copy(buffer_data.begin(), buffer_data.end(),
              static_cast<double *>(out_data));
  } else if (r_type == INTSXP) {
    out_data = INTEGER(out);
    std::copy(buffer_data.begin(), buffer_data.end(),
              static_cast<int *>(out_data));
  } else if (r_type == LGLSXP) {
    out_data = LOGICAL(out);
    for (size_t i = 0; i < numel; ++i) {
      static_cast<int *>(out_data)[i] = buffer_data[i] ? 1 : 0;
    }
  }

  // Set dimensions only once
  if (dimensions.size() > 0) {
    SEXP dim_attr = PROTECT(Rf_allocVector(INTSXP, dimensions.size()));
    int *dim_data = INTEGER(dim_attr);
    std::copy(dimensions.begin(), dimensions.end(), dim_data);
    Rf_setAttrib(out, R_DimSymbol, dim_attr);
    UNPROTECT(1); // Unprotect dim_attr
  }

  UNPROTECT(1); // Unprotect out
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
std::string impl_client_platform_name(Rcpp::XPtr<rpjrt::PJRTClient> client) {
  return client->platform_name();
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input) {
  std::vector<rpjrt::PJRTBuffer *> inputs(input.size());
  for (auto i = 0; i < input.size(); i++) {
    auto elt = input[i];
    auto buffer = Rcpp::as<Rcpp::XPtr<rpjrt::PJRTBuffer>>(elt);
    inputs[i] = buffer.get();
  }

  auto outs = executable->execute(inputs);
  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(outs[0].release(), true);
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTElementType>
impl_buffer_element_type(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  auto element_type =
      std::make_unique<rpjrt::PJRTElementType>(buffer->element_type());
  Rcpp::XPtr<rpjrt::PJRTElementType> xptr(element_type.release(), true);
  xptr.attr("class") = "PJRTElementType";
  return xptr;
}

// [[Rcpp::export()]]
std::string
impl_element_type_as_string(Rcpp::XPtr<rpjrt::PJRTElementType> element_type) {
  return element_type->as_string();
}

// [[Rcpp::export()]]
std::vector<int64_t>
impl_buffer_dimensions(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  return buffer->dimensions();
}

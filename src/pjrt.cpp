#include <Rcpp.h>

#include <algorithm>
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

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_scalar_buffer_from_host(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                    SEXP data) {
  // Check that length is 1
  if (Rf_length(data) != 1) {
    Rcpp::stop("Data must be a scalar (length 1 vector).");
  }

  // Handle numeric scalar (REALSXP)
  if (TYPEOF(data) == REALSXP) {
    float dfloat = (float)*(REAL(data));
    void *data_ptr = static_cast<void *>(&dfloat);

    Rcpp::XPtr xptr(
        client->buffer_from_host(data_ptr, std::nullopt, PJRT_Buffer_Type_F32)
            .release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  }

  // Handle logical scalar (LGLSXP)
  if (TYPEOF(data) == LGLSXP) {
    bool dbool = (bool)*(LOGICAL(data));
    void *data_ptr = static_cast<void *>(&dbool);

    Rcpp::XPtr xptr(
        client->buffer_from_host(data_ptr, std::nullopt, PJRT_Buffer_Type_PRED)
            .release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  }

  Rcpp::stop(
      "Data must be a numeric scalar (REALSXP) or logical scalar (LGLSXP).");
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_double(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                               SEXP dims, std::string precision) {
  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }
  // Determine dims
  std::optional<std::vector<int64_t>> dims_vec;
  if (dims != R_NilValue) {
    int dim_len = Rf_length(dims);
    dims_vec = std::vector<int64_t>(dim_len);
    std::copy(INTEGER(dims), INTEGER(dims) + dim_len, dims_vec->data());
  } else {
    dims_vec = std::vector<int64_t>{len};
  }
  // Only support native C++ types: float (32) and double (64)
  PJRT_Buffer_Type dtype;
  void *data_ptr = nullptr;
  if (precision == "32") {
    dtype = PJRT_Buffer_Type_F32;
    std::vector<float> f32(len);
    for (int i = 0; i < len; ++i)
      f32[i] = static_cast<float>(REAL(data)[i]);
    data_ptr = f32.data();
    // Need to keep f32 alive until after buffer creation
    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
        client->buffer_from_host(data_ptr, dims_vec, dtype).release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  } else if (precision == "64") {
    dtype = PJRT_Buffer_Type_F64;
    std::vector<double> f64(len);
    for (int i = 0; i < len; ++i)
      f64[i] = static_cast<double>(REAL(data)[i]);
    data_ptr = f64.data();
    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
        client->buffer_from_host(data_ptr, dims_vec, dtype).release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  } else {
    Rcpp::stop("Unsupported floating point precision: %s. Only '32' (float) "
               "and '64' (double) are supported.",
               precision);
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_integer(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                                SEXP dims, int precision, bool is_signed) {
  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }

  // Determine dims
  std::optional<std::vector<int64_t>> dims_vec;
  if (dims != R_NilValue) {
    int dim_len = Rf_length(dims);
    dims_vec = std::vector<int64_t>(dim_len);
    std::copy(INTEGER(dims), INTEGER(dims) + dim_len, dims_vec->data());
  } else {
    dims_vec = std::vector<int64_t>{len};
  }

  // Only support native C++ integer types
  PJRT_Buffer_Type dtype;
  void *data_ptr = nullptr;

  if (is_signed) {
    if (precision == 8) {
      dtype = PJRT_Buffer_Type_S8;
      std::vector<int8_t> i8(len);
      for (int i = 0; i < len; ++i)
        i8[i] = static_cast<int8_t>(INTEGER(data)[i]);
      data_ptr = i8.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else if (precision == 16) {
      dtype = PJRT_Buffer_Type_S16;
      std::vector<int16_t> i16(len);
      for (int i = 0; i < len; ++i)
        i16[i] = static_cast<int16_t>(INTEGER(data)[i]);
      data_ptr = i16.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else if (precision == 32) {
      dtype = PJRT_Buffer_Type_S32;
      std::vector<int32_t> i32(len);
      for (int i = 0; i < len; ++i)
        i32[i] = static_cast<int32_t>(INTEGER(data)[i]);
      data_ptr = i32.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else if (precision == 64) {
      dtype = PJRT_Buffer_Type_S64;
      std::vector<int64_t> i64(len);
      for (int i = 0; i < len; ++i)
        i64[i] = static_cast<int64_t>(INTEGER(data)[i]);
      data_ptr = i64.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else {
      Rcpp::stop("Unsupported signed integer precision: %d. Only 8, 16, 32, 64 "
                 "are supported.",
                 precision);
    }
  } else {
    if (precision == 8) {
      dtype = PJRT_Buffer_Type_U8;
      std::vector<uint8_t> u8(len);
      for (int i = 0; i < len; ++i)
        u8[i] = static_cast<uint8_t>(INTEGER(data)[i]);
      data_ptr = u8.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else if (precision == 16) {
      dtype = PJRT_Buffer_Type_U16;
      std::vector<uint16_t> u16(len);
      for (int i = 0; i < len; ++i)
        u16[i] = static_cast<uint16_t>(INTEGER(data)[i]);
      data_ptr = u16.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else if (precision == 32) {
      dtype = PJRT_Buffer_Type_U32;
      std::vector<uint32_t> u32(len);
      for (int i = 0; i < len; ++i)
        u32[i] = static_cast<uint32_t>(INTEGER(data)[i]);
      data_ptr = u32.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else if (precision == 64) {
      dtype = PJRT_Buffer_Type_U64;
      std::vector<uint64_t> u64(len);
      for (int i = 0; i < len; ++i)
        u64[i] = static_cast<uint64_t>(INTEGER(data)[i]);
      data_ptr = u64.data();
      Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
          client->buffer_from_host(data_ptr, dims_vec, dtype).release());
      xptr.attr("class") = "PJRTBuffer";
      return xptr;
    } else {
      Rcpp::stop("Unsupported unsigned integer precision: %d. Only 8, 16, 32, "
                 "64 are supported.",
                 precision);
    }
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_logical(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data,
                                SEXP dims) {
  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }
  // Determine dims
  std::optional<std::vector<int64_t>> dims_vec;
  if (dims != R_NilValue) {
    int dim_len = Rf_length(dims);
    dims_vec = std::vector<int64_t>(dim_len);
    std::copy(INTEGER(dims), INTEGER(dims) + dim_len, dims_vec->data());
  } else {
    dims_vec = std::vector<int64_t>{len};
  }
  // Copy logical data to uint8_t array
  std::vector<uint8_t> bool_data(len);
  for (int i = 0; i < len; ++i) {
    bool_data[i] = LOGICAL(data)[i] ? 1 : 0;
  }
  void *data_ptr = bool_data.data();
  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
      client->buffer_from_host(data_ptr, dims_vec, PJRT_Buffer_Type_PRED)
          .release());
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_floating_point(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                       SEXP data, SEXP dims, int precision) {
  int len = Rf_length(data);
  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }
  // Determine dims
  std::optional<std::vector<int64_t>> dims_vec;
  if (dims != R_NilValue) {
    int dim_len = Rf_length(dims);
    dims_vec = std::vector<int64_t>(dim_len);
    std::copy(INTEGER(dims), INTEGER(dims) + dim_len, dims_vec->data());
  } else {
    dims_vec = std::vector<int64_t>{len};
  }
  // Only support native C++ types: float (32) and double (64)
  PJRT_Buffer_Type dtype;
  void *data_ptr = nullptr;
  if (precision == 32) {
    dtype = PJRT_Buffer_Type_F32;
    std::vector<float> f32(len);
    for (int i = 0; i < len; ++i)
      f32[i] = static_cast<float>(REAL(data)[i]);
    data_ptr = f32.data();
    // Need to keep f32 alive until after buffer creation
    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
        client->buffer_from_host(data_ptr, dims_vec, dtype).release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  } else if (precision == 64) {
    dtype = PJRT_Buffer_Type_F64;
    std::vector<double> f64(len);
    for (int i = 0; i < len; ++i)
      f64[i] = static_cast<double>(REAL(data)[i]);
    data_ptr = f64.data();
    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
        client->buffer_from_host(data_ptr, dims_vec, dtype).release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  } else {
    Rcpp::stop("Unsupported floating point precision: %d. Only 32 (float) and "
               "64 (double) are supported.",
               precision);
  }
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer>
impl_client_buffer_from_host(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data) {
  auto len = Rf_length(data);

  if (len == 0) {
    Rcpp::stop("Data must be a non-empty vector.");
  }

  // Get the dimensions from the dim attribute
  std::optional<std::vector<int64_t>> dims;
  SEXP dim_attr = Rf_getAttrib(data, R_DimSymbol);
  if (dim_attr != R_NilValue) {
    int dim_len = Rf_length(dim_attr);
    dims = std::vector<int64_t>(dim_len);
    std::copy(INTEGER(dim_attr), INTEGER(dim_attr) + dim_len, dims->data());
  } else {
    // If no dimensions are provided, we assume it's a flat vector
    dims = std::vector<int64_t>{len};
  }

  // Handle numeric vector (REALSXP)
  if (TYPEOF(data) == REALSXP) {
    // We have no way around it, we need to copy data to another vector
    std::vector<float> d(len);
    std::copy(REAL(data), REAL(data) + len, d.data());

    void *data_ptr = static_cast<void *>(d.data());

    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
        client->buffer_from_host(data_ptr, dims, PJRT_Buffer_Type_F32)
            .release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  }

  // Handle logical vector (LGLSXP)
  if (TYPEOF(data) == LGLSXP) {
    // Copy logical data to a bool array
    std::vector<bool> d(len);
    for (int i = 0; i < len; ++i) {
      d[i] = LOGICAL(data)[i];
    }

    // Create a temporary array for the data pointer
    std::vector<uint8_t> bool_data(len);
    for (int i = 0; i < len; ++i) {
      bool_data[i] = d[i] ? 1 : 0;
    }

    void *data_ptr = static_cast<void *>(bool_data.data());

    Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
        client->buffer_from_host(data_ptr, dims, PJRT_Buffer_Type_PRED)
            .release());
    xptr.attr("class") = "PJRTBuffer";
    return xptr;
  }

  Rcpp::stop(
      "Data must be a numeric vector (REALSXP) or logical vector (LGLSXP).");
}

// [[Rcpp::export()]]
SEXP impl_client_buffer_to_host(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  // Allocate a buffer of the required length
  const auto dimensions = buffer->dimensions();
  const auto numel = std::accumulate(dimensions.begin(), dimensions.end(), 1,
                                     std::multiplies<int64_t>());

  // Get the element type to determine how to handle the data
  PJRT_Buffer_Type element_type = buffer->element_type();

  if (element_type == PJRT_Buffer_Type_F32) {
    // Handle float data
    std::vector<float> float_buffer(numel);
    std::span<uint8_t> host_buffer(
        reinterpret_cast<uint8_t *>(float_buffer.data()),
        numel * sizeof(float));

    client->buffer_to_host(*buffer, host_buffer);

    // Convert the float buffer to an R numeric vector
    SEXP out = PROTECT(Rf_allocVector(REALSXP, numel));
    double *out_data = REAL(out);
    std::copy(float_buffer.begin(), float_buffer.end(), out_data);

    if (dimensions.size() > 1) {
      // Set the dimensions attribute if the buffer is multi-dimensional
      SEXP dim_attr = PROTECT(Rf_allocVector(INTSXP, dimensions.size()));
      int *dim_data = INTEGER(dim_attr);
      std::copy(dimensions.begin(), dimensions.end(), dim_data);
      Rf_setAttrib(out, R_DimSymbol, dim_attr);
      UNPROTECT(1); // Unprotect dim_attr
    }

    UNPROTECT(1);
    return out;
  } else if (element_type == PJRT_Buffer_Type_PRED) {
    // Handle boolean data
    std::vector<uint8_t> bool_buffer(numel);
    std::span<uint8_t> host_buffer(bool_buffer.data(), numel * sizeof(uint8_t));

    client->buffer_to_host(*buffer, host_buffer);

    // Convert the boolean buffer to an R logical vector
    SEXP out = PROTECT(Rf_allocVector(LGLSXP, numel));
    int *out_data = LOGICAL(out);
    for (size_t i = 0; i < numel; ++i) {
      out_data[i] = bool_buffer[i] ? 1 : 0;
    }

    if (dimensions.size() > 1) {
      // Set the dimensions attribute if the buffer is multi-dimensional
      SEXP dim_attr = PROTECT(Rf_allocVector(INTSXP, dimensions.size()));
      int *dim_data = INTEGER(dim_attr);
      std::copy(dimensions.begin(), dimensions.end(), dim_data);
      Rf_setAttrib(out, R_DimSymbol, dim_attr);
      UNPROTECT(1); // Unprotect dim_attr
    }

    UNPROTECT(1);
    return out;
  } else {
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

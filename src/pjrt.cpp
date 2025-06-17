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

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_scalar_buffer_from_host(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data) {
  // Check that data is a scalar and REALSXP, using TYPEOF
  if (TYPEOF(data) != REALSXP) {
    Rcpp::stop("Data must be a numeric scalar (REALSXP).");
  }

  // Check that length is 1
  if (Rf_length(data) != 1) {
    Rcpp::stop("Data must be a scalar (length 1 vector).");
  }

  float dfloat = (float)*(REAL(data));
  void *data_ptr = static_cast<void *>(&dfloat);

  Rcpp::XPtr xptr(
      client->buffer_from_host(data_ptr, std::nullopt, PJRT_Buffer_Type_F32)
          .release());
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_host(
    Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data) {
  // Check that data is a numeric vector (REALSXP)
  if (TYPEOF(data) != REALSXP) {
    Rcpp::stop("Data must be a numeric vector (REALSXP).");
  }

  auto len = Rf_length(data);

  if (len == 0) {
    Rcpp::stop("Data must be a non-empty numeric vector.");
  }

  // We have no way around it, we need to copy data to another vector
  std::vector<float> d(len);
  std::copy(REAL(data), REAL(data) + len, d.data());

  // Now get the dimensions from the dim attribute
  std::optional<std::vector<int64_t>> dims;
  SEXP dim_attr = Rf_getAttrib(data, R_DimSymbol);
  if (dim_attr != R_NilValue) {
    if (TYPEOF(dim_attr) != INTSXP) {
      Rcpp::stop("Dimensions must be an integer vector.");
    }
    int dim_len = Rf_length(dim_attr);
    dims = std::vector<int64_t>(dim_len);
    std::copy(INTEGER(dim_attr), INTEGER(dim_attr) + dim_len, dims->data());
  } else {
    // If no dimensions are provided, we assume it's a flat vector
    dims = std::vector<int64_t>{len};
  }

  void *data_ptr = static_cast<void *>(d.data());

  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(
      client->buffer_from_host(data_ptr, dims, PJRT_Buffer_Type_F32).release());
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}

// [[Rcpp::export()]]
SEXP impl_client_buffer_to_host(Rcpp::XPtr<rpjrt::PJRTClient> client,
                                Rcpp::XPtr<rpjrt::PJRTBuffer> buffer) {
  // Allocate a buffer of the required length
  const auto dimensions = buffer->dimensions();
  const auto numel = std::accumulate(dimensions.begin(), dimensions.end(), 1,
                                     std::multiplies<int64_t>());

  // Create a float buffer to hold data
  std::vector<float> float_buffer(numel);
  std::span<uint8_t> host_buffer(
      reinterpret_cast<uint8_t *>(float_buffer.data()), numel * sizeof(float));

  client->buffer_to_host(*buffer, host_buffer);

  // Convert the float buffer to an R numeric vector
  // Allocate an R numeric vector of the same length
  SEXP out = PROTECT(Rf_allocVector(REALSXP, numel));
  double *out_data = REAL(out);

  std::copy(float_buffer.begin(), float_buffer.end(), out_data);

  UNPROTECT(1);
  return out;
}

// [[Rcpp::export()]]
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable,
    Rcpp::XPtr<rpjrt::PJRTBuffer> input) {
  auto outs = executable->execute(input.get());
  Rcpp::XPtr<rpjrt::PJRTBuffer> xptr(outs[0].release(), true);
  xptr.attr("class") = "PJRTBuffer";
  return xptr;
}
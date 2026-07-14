// Declarations for the pjrt.cpp entry points that are reused directly from C++
// (not only through their generated R bindings). The dispatcher's PjrtEngine
// calls these to run executables, allocate/upload buffers, and copy buffers
// across devices, reusing pjrt.cpp's keepalive/donation/options logic instead
// of duplicating it. The definitions live in pjrt.cpp, which also includes this
// header so the two stay in sync.
//
// Deliberately separate from pjrt.h, which declares the low-level PJRT C API
// loader typedef and must stay free of Rcpp / execution headers.

#pragma once

#include <Rcpp.h>

#include <cstdint>
#include <string>
#include <vector>

#include "buffer.h"
#include "client.h"
#include "device.h"

Rcpp::List impl_loaded_executable_execute(
    Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input,
    Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options);
Rcpp::XPtr<rpjrt::PJRTExecuteOptions> impl_execution_options_create(
    std::vector<int64_t> non_donatable_input_indices, int launch_id);
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_empty(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    std::vector<int64_t> dims, std::string dtype);
// As impl_client_buffer_empty(), but taking the dtype already parsed. Callers
// that hold a PJRT_Buffer_Type skip the per-call string round-trip.
Rcpp::XPtr<rpjrt::PJRTBuffer> client_buffer_empty(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    std::vector<int64_t> dims, PJRT_Buffer_Type pjrt_dtype);
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_double(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype);
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_integer(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype);
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_logical(
    Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    SEXP data, std::vector<int64_t> dims, std::string dtype);
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_buffer_copy_to_device(
    Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, Rcpp::XPtr<rpjrt::PJRTDevice> device,
    Rcpp::XPtr<rpjrt::PJRTClient> dst_client, bool cross_client);

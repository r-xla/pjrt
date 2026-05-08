#include "xla/ffi/api/ffi.h"

#include <iostream>
#include <string_view>

#include "buffer.h"
#include "buffer_printer.h"
#include "plugin.h"
#include "utils.h"
#include "xla/ffi/api/api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"

#ifndef _WIN32
#include "ffi_cuda.h"
#endif

using namespace xla::ffi;

namespace rpjrt {

PJRT_FFI_Extension *get_pjrt_ffi_extension(PJRTPlugin *plugin) {
  auto extension = plugin->api->extension_start;
  while (extension != nullptr) {
    if (extension->type == PJRT_Extension_Type_FFI) {
      if (extension->struct_size != sizeof(PJRT_FFI_Extension)) {
        throw std::runtime_error(
            "PJRT FFI Extension struct size mismatch: expected " +
            std::to_string(sizeof(PJRT_FFI_Extension)) + ", got " +
            std::to_string(extension->struct_size));
      }
      return reinterpret_cast<PJRT_FFI_Extension *>(extension);
    }
    extension = extension->next;
  }
  throw std::runtime_error("PJRT FFI Extension not found");
}

PJRT_Buffer_Type to_pjrt_type(xla::ffi::DataType dtype) {
  switch (dtype) {
    case xla::ffi::DataType::F32:
      return PJRT_Buffer_Type_F32;
    case xla::ffi::DataType::F64:
      return PJRT_Buffer_Type_F64;
    case xla::ffi::DataType::S8:
      return PJRT_Buffer_Type_S8;
    case xla::ffi::DataType::S16:
      return PJRT_Buffer_Type_S16;
    case xla::ffi::DataType::S32:
      return PJRT_Buffer_Type_S32;
    case xla::ffi::DataType::S64:
      return PJRT_Buffer_Type_S64;
    case xla::ffi::DataType::U8:
      return PJRT_Buffer_Type_U8;
    case xla::ffi::DataType::U16:
      return PJRT_Buffer_Type_U16;
    case xla::ffi::DataType::U32:
      return PJRT_Buffer_Type_U32;
    case xla::ffi::DataType::U64:
      return PJRT_Buffer_Type_U64;
    case xla::ffi::DataType::PRED:
      return PJRT_Buffer_Type_PRED;
    default:
      throw std::runtime_error("Unsupported buffer element type for printing.");
  }
}

// Implement a custom call as a C++ function. Note that we can use `Buffer` type
// defined by XLA FFI that gives us access to buffer data type and shape.
xla::ffi::Error do_test_call() {
  return xla::ffi::Error(xla::ffi::ErrorCode::kOk,
                         "Custom call executed successfully");
}

XLA_FFI_DEFINE_HANDLER_AUTO(test_handler, do_test_call);

constexpr std::string_view kPrintHeaderAttr = "print_header";
constexpr std::string_view kPrintFooterAttr = "print_footer";

std::string dtype_display_name(xla::ffi::DataType dtype) {
  PJRT_Buffer_Type pjrt_type = to_pjrt_type(dtype);
  PJRTElementType elt(pjrt_type);
  std::string name = elt.as_string();
  if (name == "pred") return "bool";
  return name;
}

// Shared helper: prints buffer data (already on host) using
// buffer_to_string_lines.
xla::ffi::Error print_host_buffer(const void *data, Dictionary attrs,
                                  AnyBuffer buffer) {
  std::string_view header = "PJRTBuffer";
  if (attrs.contains(kPrintHeaderAttr)) {
    auto print_header = attrs.get<std::string_view>(kPrintHeaderAttr);
    if (print_header) {
      header = *print_header;
    }
  }

  if (!data) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kDataLoss,
                           "Could not find untyped data.");
  }

  const auto dimensions_span = buffer.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(),
                                  dimensions_span.end());

  PJRT_Buffer_Type element_type;
  try {
    element_type = to_pjrt_type(buffer.element_type());
  } catch (const std::exception &e) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument, e.what());
  }

  auto lines = buffer_to_string_lines(data, dimensions, element_type);
  if (!header.empty()) {
    Rcpp::Rcout << header << "\n";
  }

  for (const auto &line : lines) {
    Rcpp::Rcout << line << '\n';
  }

  if (attrs.contains(kPrintFooterAttr)) {
    auto print_footer = attrs.get<std::string_view>(kPrintFooterAttr);
    if (print_footer && !print_footer->empty()) {
      Rcpp::Rcout << *print_footer << "\n";
    }
  } else {
    Rcpp::Rcout << "[ " << dtype_display_name(buffer.element_type()) << "{";
    const auto dims = buffer.dimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i != 0) {
        Rcpp::Rcout << ',';
      }
      Rcpp::Rcout << dims[i];
    }
    Rcpp::Rcout << "} ]" << "\n";
  }

  return xla::ffi::Error(xla::ffi::ErrorCode::kOk,
                         "Custom call executed successfully");
}

xla::ffi::Error do_print_call(Dictionary attrs, AnyBuffer buffer) {
  return print_host_buffer(buffer.untyped_data(), attrs, buffer);
}

XLA_FFI_DEFINE_HANDLER_AUTO(print_handler, do_print_call);

xla::ffi::Error do_print_call_cuda(void *stream, Dictionary attrs,
                                   AnyBuffer buffer) {
#ifdef _WIN32
  return xla::ffi::Error(xla::ffi::ErrorCode::kUnimplemented,
                         "CUDA print_tensor is not supported on Windows");
#else
  auto &g = get_cuda_libs();
  if (!g.loaded) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "CUDA driver not available");
  }

  std::size_t size = buffer.size_bytes();
  std::vector<char> host(size);

  CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(buffer.untyped_data());

  PJRT_RETURN_IF_GPU_ERROR(g.memcpy_dtoh(host.data(), device_ptr, size, stream),
                           "cuMemcpyDtoHAsync_v2");
  PJRT_RETURN_IF_GPU_ERROR(g.stream_synchronize(stream), "cuStreamSynchronize");

  return print_host_buffer(host.data(), attrs, buffer);
#endif
}

XLA_FFI_DEFINE_HANDLER(print_handler_cuda, do_print_call_cuda,
                       xla::ffi::Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Attrs<Dictionary>()
                           .Arg<AnyBuffer>());

void register_ffi_handlers(PJRTPlugin *plugin,
                           const std::string &platform_name) {
  auto ffi_extension = get_pjrt_ffi_extension(plugin);

  PJRT_FFI_Register_Handler_Args args{};
  args.struct_size = sizeof(PJRT_FFI_Register_Handler_Args);
  args.handler = (void *)test_handler;
  args.target_name = "test_handler";
  args.target_name_size = strlen(args.target_name);
  args.platform_name = platform_name.c_str();
  args.platform_name_size = strlen(args.platform_name);

  check_err(plugin->api.get(), ffi_extension->register_handler(&args));
}

}  // namespace rpjrt

// [[Rcpp::export]]
void impl_register_custom_call(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin,
                               const std::string &target_name, SEXP handler_ptr,
                               const std::string &platform_name) {
  if (TYPEOF(handler_ptr) != EXTPTRSXP) {
    throw std::runtime_error("handler must be an external pointer");
  }
  void *handler = R_ExternalPtrAddr(handler_ptr);
  if (handler == nullptr) {
    throw std::runtime_error("handler external pointer is NULL");
  }

  auto ffi_extension = get_pjrt_ffi_extension(plugin.get());

  PJRT_FFI_Register_Handler_Args args{};
  args.struct_size = sizeof(PJRT_FFI_Register_Handler_Args);
  args.handler = handler;
  args.target_name = target_name.c_str();
  args.target_name_size = target_name.size();
  args.platform_name = platform_name.c_str();
  args.platform_name_size = platform_name.size();

  check_err(plugin->api.get(), ffi_extension->register_handler(&args));
}

// [[Rcpp::export]]
SEXP get_print_handler() {
  return R_MakeExternalPtr((void *)rpjrt::print_handler, R_NilValue,
                           R_NilValue);
}

// [[Rcpp::export]]
SEXP get_print_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::print_handler_cuda, R_NilValue,
                           R_NilValue);
}

// [[Rcpp::export]]
bool test_get_extension(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin,
                        const std::string &platform_name) {
  if (rpjrt::get_pjrt_ffi_extension(plugin.get()) != nullptr) {
    register_ffi_handlers(plugin.get(), platform_name);
    return true;
  }

  return false;
}

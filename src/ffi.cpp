#include "xla/ffi/api/ffi.h"

#include "plugin.h"
#include "utils.h"
#include "xla/ffi/api/api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"

using namespace xla::ffi;

namespace rpjrt {

PJRT_FFI_Extension* get_pjrt_ffi_extension(PJRTPlugin* plugin) {
  auto extension = plugin->api->extension_start;
  while (extension != nullptr) {
    if (extension->type == PJRT_Extension_Type_FFI) {
      if (extension->struct_size != sizeof(PJRT_FFI_Extension)) {
        throw std::runtime_error(
            "PJRT FFI Extension struct size mismatch: expected " +
            std::to_string(sizeof(PJRT_FFI_Extension)) + ", got " +
            std::to_string(extension->struct_size));
      }
      return reinterpret_cast<PJRT_FFI_Extension*>(extension);
    }
    extension = extension->next;
  }
  throw std::runtime_error("PJRT FFI Extension not found");
}

// Implement a custom call as a C++ function. Note that we can use `Buffer` type
// defined by XLA FFI that gives us access to buffer data type and shape.
xla::ffi::Error do_test_call() {
  return xla::ffi::Error(xla::ffi::ErrorCode::kOk,
                         "Custom call executed successfully");
}

XLA_FFI_DEFINE_HANDLER_AUTO(test_handler, do_test_call);

xla::ffi::Error do_print_call(AnyBuffer buffer) {
  const void* arg_data = buffer.untyped_data();
  const auto dtype  = buffer.element_type();
  return xla::ffi::Error(xla::ffi::ErrorCode::kOk,
                         "Custom call executed successfully");
}

void register_ffi_handlers(PJRTPlugin* plugin,
                           const std::string& platform_name) {
  auto ffi_extension = get_pjrt_ffi_extension(plugin);

  PJRT_FFI_Register_Handler_Args args{};
  args.struct_size = sizeof(PJRT_FFI_Register_Handler_Args);
  args.handler = (void*)test_handler;
  args.target_name = "test_handler";
  args.target_name_size = strlen(args.target_name);
  args.platform_name = platform_name.c_str();
  args.platform_name_size = strlen(args.platform_name);

  check_err(plugin->api.get(), ffi_extension->register_handler(&args));
}

}  // namespace rpjrt

// [[Rcpp::export]]
bool test_get_extension(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin,
                        const std::string& platform_name) {
  if (rpjrt::get_pjrt_ffi_extension(plugin.get()) != nullptr) {
    register_ffi_handlers(plugin.get(), platform_name);
    return true;
  }

  return false;
}

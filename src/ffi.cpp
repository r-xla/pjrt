#include "plugin.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "utils.h"

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

// Constrain custom call arguments to 1-dimensional buffers of F32 data type.
using BufferF32 = xla::ffi::BufferR1<xla::ffi::DataType::F32>;

// Implement a custom call as a C++ function. Note that we can use `Buffer` type
// defined by XLA FFI that gives us access to buffer data type and shape.
xla::ffi::Error do_custom_call() {
  // Check that dimensions are compatible.
  throw std::runtime_error(
      "Custom call not implemented yet. This is a placeholder function.");
}

XLA_FFI_DEFINE_HANDLER_AUTO(test_handler, do_custom_call);

void register_ffi_handlers(PJRTPlugin* plugin) {
  auto ffi_extension = get_pjrt_ffi_extension(plugin);

  PJRT_FFI_Register_Handler_Args args{};
  args.struct_size = sizeof(PJRT_FFI_Register_Handler_Args);
  args.handler = (void*)test_handler;
  args.target_name = "test_handler";
  args.target_name_size = strlen(args.target_name);
  args.platform_name = "Host";
  args.platform_name_size = strlen(args.platform_name);

  check_err(plugin->api.get(), ffi_extension->register_handler(&args));
}

}  // namespace rpjrt

// [[Rcpp::export]]
bool test_get_extension(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin) {
  if (rpjrt::get_pjrt_ffi_extension(plugin.get()) != nullptr) {
    register_ffi_handlers(plugin.get());
    return true;
  }
  
  return false;
}

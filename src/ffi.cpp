#include "plugin.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/ffi/api/c_api.h" 

namespace rpjrt {

PJRT_FFI_Extension* get_pjrt_ffi_extension(PJRTPlugin* plugin) {
    auto extension = plugin->api->extension_start;
    while (extension != nullptr) {
        if (extension->type == PJRT_Extension_Type_FFI) {
            return reinterpret_cast<PJRT_FFI_Extension*>(extension);
        }
        extension = extension->next;
    }
    throw std::runtime_error("PJRT FFI Extension not found");
}

void register_ffi_handlers(PJRTPlugin* plugin) {
    auto ffi_extension = get_pjrt_ffi_extension(plugin);
    
    PJRT_FFI_Register_Handler_Args args{};
    args.struct_size = sizeof(PJRT_FFI_Register_Handler_Args);
    //args.handler = 
    //ffi_extension->register_handler()
}

}

// [[Rcpp::export]]
bool test_get_extension(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin) {
    return rpjrt::get_pjrt_ffi_extension(plugin.get()) != nullptr;
}

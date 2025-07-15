#include "plugin.h"

#include <dlfcn.h>

#include "pjrt.h"
#include "utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

PJRTPlugin::PJRTPlugin(const std::string &path)
    :  // Use an explicit empty deleter cause there's no way to unload a plugin
       // yet
      api(load_pjrt_plugin(path), [](auto) {}) {
  this->initialize();
}

void PJRTPlugin::initialize() {
  // Initialize the Plugin
  {
    PJRT_Plugin_Initialize_Args args{};
    args.extension_start = nullptr;
    args.struct_size = sizeof(PJRT_Plugin_Initialize_Args);
    check_err(api.get(), api->PJRT_Plugin_Initialize(&args));
  }
}

std::unique_ptr<PJRTClient> PJRTPlugin::client_create() {
  PJRT_Client_Create_Args args{};
  args.num_options = 0;
  args.struct_size = sizeof(PJRT_Client_Create_Args);
  check_err(this->api.get(), this->api->PJRT_Client_Create(&args));
  return std::make_unique<PJRTClient>(args.client, this->api);
}

PJRT_Api *PJRTPlugin::load_pjrt_plugin(const std::string &path) {
  const auto handle =
      dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);

  if (!handle) {
    const char* error = dlerror();
    throw std::runtime_error("Failed to load plugin from path: " + path + "\nError: " + (error ? error : "Unknown error"));
  }

  GetPjrtApiFunc GetPjrtApi = nullptr;
  GetPjrtApi = (GetPjrtApiFunc)dlsym(handle, "GetPjrtApi");

  if (!GetPjrtApi) {
    throw std::runtime_error("Failed to load GetPjrtApi function");
  }

  return GetPjrtApi();
}

}  // namespace rpjrt

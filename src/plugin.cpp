#include "plugin.h"

#include <Rcpp.h>
#ifndef _WIN32
# include <dlfcn.h>
#else
# define WIN32_LEAN_AND_MEAN 1
# include <windows.h>
#endif

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
    check_err(api.get(), api->PJRT_Plugin_Initialize_(&args));
  }
}

std::unique_ptr<PJRTClient> PJRTPlugin::client_create() {
  PJRT_Client_Create_Args args{};
  args.num_options = 0;
  args.struct_size = sizeof(PJRT_Client_Create_Args);
  check_err(this->api.get(), this->api->PJRT_Client_Create_(&args));
  return std::make_unique<PJRTClient>(args.client, this->api);
}

std::vector<std::pair<std::string, SEXP>> PJRTPlugin::attributes() const {
  PJRT_Plugin_Attributes_Args args{};
  args.struct_size = sizeof(PJRT_Plugin_Attributes_Args);
  args.extension_start = nullptr;
  check_err(api.get(), api->PJRT_Plugin_Attributes_(&args));
  std::vector<std::pair<std::string, SEXP>> out;
  for (size_t i = 0; i < args.num_attributes; ++i) {
    const PJRT_NamedValue &nv = args.attributes[i];
    std::string key(nv.name, nv.name_size);
    SEXP value;
    switch (nv.type) {
      case PJRT_NamedValue_kString:
        value =
            Rf_mkString(std::string(nv.string_value, nv.value_size).c_str());
        break;
      case PJRT_NamedValue_kInt64:
        value = Rf_ScalarInteger(static_cast<int>(nv.int64_value));
        break;
      case PJRT_NamedValue_kInt64List: {
        Rcpp::IntegerVector v(nv.int64_array_value,
                              nv.int64_array_value + nv.value_size);
        value = v;
        break;
      }
      case PJRT_NamedValue_kFloat:
        value = Rf_ScalarReal(static_cast<double>(nv.float_value));
        break;
      case PJRT_NamedValue_kBool:
        value = Rf_ScalarLogical(nv.bool_value ? TRUE : FALSE);
        break;
      default:
        value = Rf_ScalarLogical(NA_LOGICAL);
    }
    out.emplace_back(key, value);
  }
  return out;
}

std::pair<int, int> PJRTPlugin::pjrt_api_version() const {
  return {api->pjrt_api_version.major_version,
          api->pjrt_api_version.minor_version};
}

PJRT_Api *PJRTPlugin::load_pjrt_plugin(const std::string &path) {
  void* handle = NULL;
#ifdef _WIN32
  std::cout << "LOADING PJRT DLL" << std::endl;
  handle = (void*)::LoadLibraryEx(path.c_str(), NULL, 0);
  std::cout << "THE DLL MAYBE BE LOADED?" << std::endl;
  if (handle == NULL) {
    std::string* pError;
    LPVOID lpMsgBuf;
    DWORD dw = ::GetLastError();

    DWORD length = ::FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );

    if (length != 0)
    {
      std::string msg((LPTSTR)lpMsgBuf);
      LocalFree(lpMsgBuf);
      pError->assign(msg);
    }
    else
    {
      pError->assign("(Unknown error)");
    }
    throw std::runtime_error(pError->c_str());
  }

  GetPjrtApiFunc GetPjrtApi = nullptr;
  GetPjrtApi = (GetPjrtApiFunc)::GetProcAddress((HINSTANCE)handle, "GetPjrtApi");

  if (!GetPjrtApi) {
    throw std::runtime_error("Failed to load GetPjrtApi function");
  }

  std::cout << "FOUND PJRT API" << std::endl; 

  return GetPjrtApi();
#else
  int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
  flags |= RTLD_NODELETE;
#endif
  handle = dlopen(path.c_str(), flags);

  if (handle == NULL) {
    const char *error = dlerror();
    throw std::runtime_error("Failed to load plugin from path: " + path +
                             "\nError: " + (error ? error : "Unknown error"));
  }

  GetPjrtApiFunc GetPjrtApi = nullptr;
  GetPjrtApi = (GetPjrtApiFunc)dlsym(handle, "GetPjrtApi");

  if (!GetPjrtApi) {
    throw std::runtime_error("Failed to load GetPjrtApi function");
  }

  return GetPjrtApi();
#endif
}
}  // namespace rpjrt

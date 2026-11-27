#include "plugin.h"

#include <Rcpp.h>

#include <algorithm>
#include <limits>
#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
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

std::unique_ptr<PJRTClient> PJRTPlugin::client_create(
    const std::vector<std::pair<std::string, int64_t>> &options) {
  PJRT_Client_Create_Args args{};
  args.struct_size = sizeof(PJRT_Client_Create_Args);

  // Convert options to PJRT_NamedValue array
  std::vector<PJRT_NamedValue> named_values;
  named_values.reserve(options.size());

  for (const auto &opt : options) {
    PJRT_NamedValue nv{};
    nv.struct_size = sizeof(PJRT_NamedValue);
    nv.name = opt.first.c_str();
    nv.name_size = opt.first.size();
    nv.type = PJRT_NamedValue_kInt64;
    nv.int64_value = opt.second;
    named_values.push_back(nv);
  }

  args.create_options = named_values.empty() ? nullptr : named_values.data();
  args.num_options = named_values.size();

  check_err(this->api.get(), this->api->PJRT_Client_Create_(&args));
  return std::make_unique<PJRTClient>(
      make_pjrt_handle(this->api.get(), args.client), this->api);
}

namespace {

Rcpp::RObject int64s_to_r(const int64_t *values, size_t n) {
  bool fits = std::all_of(values, values + n, [](int64_t v) {
    return v > std::numeric_limits<int>::min() &&
           v <= std::numeric_limits<int>::max();
  });
  if (fits) {
    Rcpp::IntegerVector out(n);
    std::transform(values, values + n, out.begin(),
                   [](int64_t v) { return static_cast<int>(v); });
    return out;
  }
  Rcpp::NumericVector out(n);
  std::transform(values, values + n, out.begin(),
                 [](int64_t v) { return static_cast<double>(v); });
  return out;
}

}  // namespace

Rcpp::List named_values_to_list(const PJRT_NamedValue *values, size_t n) {
  Rcpp::List out(n);
  Rcpp::CharacterVector names(n);
  for (size_t i = 0; i < n; ++i) {
    const PJRT_NamedValue &nv = values[i];
    names[i] = std::string(nv.name, nv.name_size);
    switch (nv.type) {
      case PJRT_NamedValue_kString:
        out[i] = std::string(nv.string_value, nv.value_size);
        break;
      case PJRT_NamedValue_kInt64:
        out[i] = int64s_to_r(&nv.int64_value, 1);
        break;
      case PJRT_NamedValue_kInt64List:
        out[i] = int64s_to_r(nv.int64_array_value, nv.value_size);
        break;
      case PJRT_NamedValue_kFloat:
        out[i] = static_cast<double>(nv.float_value);
        break;
      case PJRT_NamedValue_kBool:
        out[i] = static_cast<bool>(nv.bool_value);
        break;
      default:
        out[i] = Rcpp::LogicalVector::create(NA_LOGICAL);
    }
  }
  out.attr("names") = names;
  return out;
}

Rcpp::List PJRTPlugin::attributes() const {
  PJRT_Plugin_Attributes_Args args{};
  args.struct_size = sizeof(PJRT_Plugin_Attributes_Args);
  args.extension_start = nullptr;
  check_err(api.get(), api->PJRT_Plugin_Attributes_(&args));
  return named_values_to_list(args.attributes, args.num_attributes);
}

std::pair<int, int> PJRTPlugin::pjrt_api_version() const {
  return {api->pjrt_api_version.major_version,
          api->pjrt_api_version.minor_version};
}

void throw_last_error(const std::string &prefix) {
#ifdef _WIN32
  DWORD dw = ::GetLastError();
  if (dw == 0) {
    throw std::runtime_error(
        prefix + ": Failed to load library (no error code available)");
  }

  LPTSTR lpMsgBuf = NULL;
  DWORD length = ::FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0,
      NULL);

  if (length != 0) {
    std::string msg(lpMsgBuf);
    LocalFree(lpMsgBuf);
    throw std::runtime_error(prefix + ": " + msg);
  }
  throw std::runtime_error(prefix + ": (Unknown error)");
#else
  const char *error = dlerror();
  throw std::runtime_error(prefix + ": " + (error ? error : "Unknown error"));
#endif
}

PJRT_Api *PJRTPlugin::load_pjrt_plugin(const std::string &path) {
  void *handle = NULL;
#ifdef _WIN32
  handle = (void *)::LoadLibraryEx(path.c_str(), NULL, 0);
  if (handle == NULL) {
    throw_last_error("Failed to load plugin from path: " + path);
  }

  GetPjrtApiFunc GetPjrtApi = nullptr;
  GetPjrtApi =
      (GetPjrtApiFunc)::GetProcAddress((HINSTANCE)handle, "GetPjrtApi");

  if (!GetPjrtApi) {
    throw_last_error("Failed to load GetPjrtApi function");
  }

  return GetPjrtApi();
#else
  int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
  flags |= RTLD_NODELETE;
#endif
  handle = dlopen(path.c_str(), flags);

  if (handle == NULL) {
    throw_last_error("Failed to load plugin from path: " + path);
  }

  GetPjrtApiFunc GetPjrtApi = nullptr;
  GetPjrtApi = (GetPjrtApiFunc)dlsym(handle, "GetPjrtApi");

  if (!GetPjrtApi) {
    throw_last_error("Failed to load GetPjrtApi function");
  }

  return GetPjrtApi();
#endif
}
}  // namespace rpjrt

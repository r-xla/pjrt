// The `pjrt_cuda_kernel` custom call: launches a user-written CUDA kernel.
//
// A kernel lives in a *module*, registered from R by `pjrt_cuda_module()`
// under a content hash. A module is either CUDA C++ source, compiled with
// NVRTC the first time one of its kernels runs on a device (and cached on
// disk, keyed by source, options and GPU architecture), or a prebuilt image
// (PTX, cubin or fatbin) handed straight to the driver.
//
// The custom call names the module and kernel in its attributes, together
// with the launch configuration and the kernel's scalar arguments. Its
// operands and results are passed to the kernel as device pointers, in that
// order, followed by the scalars:
//
//   kernel(in_0, ..., in_n, out_0, ..., out_m, scalar_0, ..., scalar_k)
//
// This is the same model as jaxlib's `triton_kernel_call` and CuPy's
// `RawKernel`: one generic handler instead of one FFI handler per kernel.
//
// Like the rest of pjrt's CUDA code (see ffi_cuda.h), nothing here includes a
// CUDA header: the driver and NVRTC entry points are resolved with dlsym, so
// the package builds and loads without a CUDA toolkit.
#include <Rcpp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ffi_common.h"
#include "xla/ffi/api/ffi.h"

#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#endif

using namespace xla::ffi;

namespace rpjrt {
namespace {

// 64-bit FNV-1a, used for module ids and cache keys. Each field is followed
// by a zero byte so that ("ab", "c") and ("a", "bc") hash differently.
struct Fnv1a {
  std::uint64_t h = 14695981039346656037ULL;
  void add(std::string_view s) {
    for (unsigned char c : s) byte(c);
    byte(0);
  }
  void byte(unsigned char c) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::string hex() const {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(h));
    return buf;
  }
};

struct ModuleSpec {
  std::string source;
  std::string filename;
  std::vector<std::string> options;
  std::vector<std::string> kernels;  // NVRTC name expressions, e.g. "f<float>"
  std::string image;                 // prebuilt PTX / cubin / fatbin
  std::string cache_dir;
};

std::mutex registry_mu;
std::unordered_map<std::string, ModuleSpec> registry;  // guarded by registry_mu

}  // namespace
}  // namespace rpjrt

#ifndef _WIN32

namespace rpjrt {
namespace {

// CUDA driver API entry points (see cuda.h). Opaque handles are void *.
struct Driver {
  int (*get_error_string)(int, const char **);
  int (*stream_get_ctx)(void *, void **);
  int (*ctx_push)(void *);
  int (*ctx_pop)(void **);
  int (*ctx_get_device)(int *);
  int (*device_get_attribute)(int *, int, int);
  int (*module_load_data)(void **, const void *);
  int (*module_get_function)(void **, void *, const char *);
  // cuFuncGetParamInfo appeared in CUDA 12.4, well before the CUDA 13 driver
  // the plugin needs, so it is required: launches are always checked.
  int (*func_get_param_info)(void *, std::size_t, std::size_t *, std::size_t *);
  int (*func_set_attribute)(void *, int, int);
  int (*launch_kernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                       unsigned, unsigned, void *, void **, void **);
  bool loaded = false;
};

constexpr int kErrorInvalidValue = 1;        // CUDA_ERROR_INVALID_VALUE
constexpr int kComputeCapabilityMajor = 75;  // CU_DEVICE_ATTRIBUTE_...
constexpr int kComputeCapabilityMinor = 76;
constexpr int kMaxDynamicSharedSize = 8;  // CU_FUNC_ATTRIBUTE_...
constexpr int kDefaultSharedMemLimit = 48 * 1024;

// NVRTC entry points (see nvrtc.h).
struct Nvrtc {
  const char *(*get_error_string)(int);
  int (*version)(int *, int *);
  int (*create_program)(void **, const char *, const char *, int,
                        const char *const *, const char *const *);
  int (*destroy_program)(void **);
  int (*add_name_expression)(void *, const char *);
  int (*compile_program)(void *, int, const char *const *);
  int (*get_program_log_size)(void *, std::size_t *);
  int (*get_program_log)(void *, char *);
  int (*get_cubin_size)(void *, std::size_t *);
  int (*get_cubin)(void *, char *);
  int (*get_lowered_name)(void *, const char *, const char **);
  bool loaded = false;
};

template <typename T>
void load_sym(void *lib, const char *name, T &out) {
  out = reinterpret_cast<T>(dlsym(lib, name));
}

// Opening by SONAME finds the libraries pjrt already loaded from the CUDA R
// package before starting the plugin (see setup_cuda_env()).
void *dlopen_first(std::initializer_list<const char *> names) {
  for (const char *name : names) {
    if (void *lib = dlopen(name, RTLD_LAZY)) return lib;
  }
  return nullptr;
}

Driver &driver() {
  static Driver d;
  static std::once_flag once;
  std::call_once(once, [] {
    void *lib = dlopen_first({"libcuda.so.1", "libcuda.so"});
    if (!lib) return;
    load_sym(lib, "cuGetErrorString", d.get_error_string);
    load_sym(lib, "cuStreamGetCtx", d.stream_get_ctx);
    load_sym(lib, "cuCtxPushCurrent_v2", d.ctx_push);
    load_sym(lib, "cuCtxPopCurrent_v2", d.ctx_pop);
    load_sym(lib, "cuCtxGetDevice", d.ctx_get_device);
    load_sym(lib, "cuDeviceGetAttribute", d.device_get_attribute);
    load_sym(lib, "cuModuleLoadData", d.module_load_data);
    load_sym(lib, "cuModuleGetFunction", d.module_get_function);
    load_sym(lib, "cuFuncGetParamInfo", d.func_get_param_info);
    load_sym(lib, "cuFuncSetAttribute", d.func_set_attribute);
    load_sym(lib, "cuLaunchKernel", d.launch_kernel);
    d.loaded = d.get_error_string && d.stream_get_ctx && d.ctx_push &&
               d.ctx_pop && d.ctx_get_device && d.device_get_attribute &&
               d.module_load_data && d.module_get_function &&
               d.func_get_param_info && d.func_set_attribute && d.launch_kernel;
  });
  return d;
}

Nvrtc &nvrtc() {
  static Nvrtc n;
  static std::once_flag once;
  std::call_once(once, [] {
    // SONAMEs first: they match the copy pjrt preloaded from the CUDA R
    // package, rather than whatever toolkit happens to be on the search path.
    void *lib =
        dlopen_first({"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"});
    if (!lib) return;
    load_sym(lib, "nvrtcGetErrorString", n.get_error_string);
    load_sym(lib, "nvrtcVersion", n.version);
    load_sym(lib, "nvrtcCreateProgram", n.create_program);
    load_sym(lib, "nvrtcDestroyProgram", n.destroy_program);
    load_sym(lib, "nvrtcAddNameExpression", n.add_name_expression);
    load_sym(lib, "nvrtcCompileProgram", n.compile_program);
    load_sym(lib, "nvrtcGetProgramLogSize", n.get_program_log_size);
    load_sym(lib, "nvrtcGetProgramLog", n.get_program_log);
    load_sym(lib, "nvrtcGetCUBINSize", n.get_cubin_size);
    load_sym(lib, "nvrtcGetCUBIN", n.get_cubin);
    load_sym(lib, "nvrtcGetLoweredName", n.get_lowered_name);
    n.loaded = n.get_error_string && n.version && n.create_program &&
               n.destroy_program && n.add_name_expression &&
               n.compile_program && n.get_program_log_size &&
               n.get_program_log && n.get_cubin_size && n.get_cubin &&
               n.get_lowered_name;
  });
  return n;
}

Error driver_error(int status, const std::string &what) {
  const char *msg = nullptr;
  driver().get_error_string(status, &msg);
  return Error::Internal(what + " failed: " + (msg ? msg : "unknown error") +
                         " (CUresult " + std::to_string(status) + ")");
}

#define RETURN_IF_DRIVER_ERROR(expr, what)                  \
  do {                                                      \
    int _status = (expr);                                   \
    if (_status != 0) return driver_error(_status, (what)); \
  } while (0)

// A compiled module: the cubin plus the lowered (mangled) names of the
// requested name expressions.
struct Compiled {
  std::string cubin;
  std::map<std::string, std::string> lowered;
};

// Cache file layout: one "<expression>\t<lowered name>" line per kernel, an
// empty line, then the cubin.
bool read_cache(const std::string &path, Compiled &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line) && !line.empty()) {
    auto tab = line.find('\t');
    if (tab == std::string::npos) return false;
    out.lowered[line.substr(0, tab)] = line.substr(tab + 1);
  }
  out.cubin.assign(std::istreambuf_iterator<char>(in), {});
  return !out.cubin.empty();
}

void write_cache(const std::string &path, const Compiled &c) {
  // Written to a unique temporary file and renamed, so that concurrent
  // sessions -- possibly in containers sharing the cache, with equal PIDs --
  // never read a half-written entry.
  std::string tmp = path + ".XXXXXX";
  int fd = mkstemp(tmp.data());
  if (fd < 0) return;
  close(fd);
  bool ok = false;
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    for (const auto &[expr, name] : c.lowered)
      out << expr << '\t' << name << '\n';
    out << '\n';
    out.write(c.cubin.data(), static_cast<std::streamsize>(c.cubin.size()));
    ok = static_cast<bool>(out);
  }
  if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0)
    std::remove(tmp.c_str());
}

std::string cache_path(const ModuleSpec &spec, const std::string &arch,
                       int nv_major, int nv_minor) {
  if (spec.cache_dir.empty()) return "";
  Fnv1a key;
  key.add(spec.source);
  for (const auto &o : spec.options) key.add(o);
  for (const auto &k : spec.kernels) key.add(k);
  key.add(arch);
  key.add(std::to_string(nv_major) + "." + std::to_string(nv_minor));
  return spec.cache_dir + "/" + key.hex() + ".cubin";
}

// Compiles `spec` for sm_<major><minor>, or reads it from the disk cache;
// `cached` then names the cache file it came from.
Error compile(const ModuleSpec &spec, int major, int minor, Compiled &out,
              std::string &cached) {
  Nvrtc &n = nvrtc();
  if (!n.loaded) {
    return Error::Internal(
        "NVRTC (libnvrtc.so) not found; it is needed to compile CUDA kernels "
        "from source. It ships with the CUDA R package that the CUDA plugin "
        "uses.");
  }
  int nv_major = 0, nv_minor = 0;
  n.version(&nv_major, &nv_minor);
  std::string arch =
      "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);

  std::string path = cache_path(spec, arch, nv_major, nv_minor);
  if (cached.empty() && !path.empty() && read_cache(path, out)) {
    cached = path;
    return Error::Success();
  }
  cached.clear();
  out = Compiled();

  void *prog = nullptr;
  int status = n.create_program(&prog, spec.source.c_str(),
                                spec.filename.c_str(), 0, nullptr, nullptr);
  if (status != 0) {
    return Error::Internal(std::string("nvrtcCreateProgram failed: ") +
                           n.get_error_string(status));
  }
  std::unique_ptr<void *, void (*)(void **)> guard(
      &prog, [](void **p) { nvrtc().destroy_program(p); });

  for (const auto &k : spec.kernels) {
    status = n.add_name_expression(prog, k.c_str());
    if (status != 0) {
      return Error::InvalidArgument(
          "nvrtcAddNameExpression(\"" + k +
          "\") failed: " + n.get_error_string(status));
    }
  }

  std::vector<const char *> opts{arch.c_str()};
  for (const auto &o : spec.options) opts.push_back(o.c_str());
  status = n.compile_program(prog, static_cast<int>(opts.size()), opts.data());

  if (status != 0) {
    std::size_t log_size = 0;
    n.get_program_log_size(prog, &log_size);
    std::string log(log_size, '\0');
    if (log_size) n.get_program_log(prog, log.data());
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n'))
      log.pop_back();
    return Error::InvalidArgument("Compiling CUDA module '" + spec.filename +
                                  "' failed:\n" + log);
  }

  for (const auto &k : spec.kernels) {
    const char *lowered = nullptr;
    status = n.get_lowered_name(prog, k.c_str(), &lowered);
    if (status != 0 || !lowered) {
      return Error::InvalidArgument("No kernel matches the name expression '" +
                                    k + "'.");
    }
    out.lowered[k] = lowered;
  }

  std::size_t size = 0;
  n.get_cubin_size(prog, &size);
  out.cubin.resize(size);
  n.get_cubin(prog, out.cubin.data());

  if (!path.empty()) write_cache(path, out);
  return Error::Success();
}

struct Kernel {
  void *function = nullptr;
  std::vector<std::size_t> param_sizes;  // in bytes, from cuFuncGetParamInfo
};

// A module loaded into one CUDA context, and the kernels looked up in it.
struct Loaded {
  void *module = nullptr;
  std::map<std::string, std::string> lowered;
  std::map<std::string, Kernel, std::less<>> kernels;
};

// Keyed by context: a module is loaded once per context it runs in. Entries
// are never removed, which assumes a context outlives the process -- true for
// pjrt, whose clients are process-wide singletons.
std::mutex loaded_mu;
std::map<std::pair<std::string, void *>, Loaded>
    loaded;  // guarded by loaded_mu

// Compiles (if needed) and loads module `id` into the current context.
Error load_module(const std::string &id, Loaded &out) {
  Driver &d = driver();
  ModuleSpec spec;
  {
    std::lock_guard<std::mutex> lock(registry_mu);
    auto s = registry.find(id);
    if (s == registry.end()) {
      return Error::InvalidArgument(
          "CUDA module " + id +
          " is not registered in this session; create it with "
          "pjrt_cuda_module() before running the program.");
    }
    spec = s->second;
  }

  if (!spec.image.empty()) {
    RETURN_IF_DRIVER_ERROR(d.module_load_data(&out.module, spec.image.data()),
                           "cuModuleLoadData");
    return Error::Success();
  }

  int device = 0, major = 0, minor = 0;
  RETURN_IF_DRIVER_ERROR(d.ctx_get_device(&device), "cuCtxGetDevice");
  RETURN_IF_DRIVER_ERROR(
      d.device_get_attribute(&major, kComputeCapabilityMajor, device),
      "cuDeviceGetAttribute");
  RETURN_IF_DRIVER_ERROR(
      d.device_get_attribute(&minor, kComputeCapabilityMinor, device),
      "cuDeviceGetAttribute");

  Compiled compiled;
  std::string cached;
  PJRT_RETURN_IF_ERROR(compile(spec, major, minor, compiled, cached));
  int status = d.module_load_data(&out.module, compiled.cubin.data());
  if (status != 0 && !cached.empty()) {
    // a corrupt or stale cache entry: drop it and compile afresh
    std::remove(cached.c_str());
    PJRT_RETURN_IF_ERROR(compile(spec, major, minor, compiled, cached));
    status = d.module_load_data(&out.module, compiled.cubin.data());
  }
  if (status != 0) return driver_error(status, "cuModuleLoadData");
  out.lowered = std::move(compiled.lowered);
  return Error::Success();
}

// Returns the kernel `name` of module `id` in context `ctx`, compiling and
// loading the module on first use. Must be called with `ctx` current.
Error get_kernel(std::string_view id, std::string_view name, void *ctx,
                 Kernel *&out) {
  Driver &d = driver();
  auto key = std::make_pair(std::string(id), ctx);

  bool present;
  {
    std::lock_guard<std::mutex> lock(loaded_mu);
    present = loaded.count(key) > 0;
  }
  if (!present) {
    // Compiling can take a while; other launches go ahead meanwhile. Should
    // two threads race here, the first module stored wins and the other is
    // left loaded but unused.
    Loaded mod;
    PJRT_RETURN_IF_ERROR(load_module(key.first, mod));
    std::lock_guard<std::mutex> lock(loaded_mu);
    loaded.emplace(key, std::move(mod));
  }

  std::lock_guard<std::mutex> lock(loaded_mu);
  Loaded &mod = loaded.at(key);
  auto k = mod.kernels.find(name);
  if (k == mod.kernels.end()) {
    std::string expr(name);
    auto low = mod.lowered.find(expr);
    const std::string &symbol = low == mod.lowered.end() ? expr : low->second;
    Kernel kernel;
    int status =
        d.module_get_function(&kernel.function, mod.module, symbol.c_str());
    if (status != 0) {
      std::string hint =
          expr.find('<') != std::string::npos
              ? " A template instantiation must be listed in the `kernels` "
                "argument of pjrt_cuda_module()."
              : " Kernels not listed in `kernels` must be declared "
                "extern \"C\".";
      return Error::InvalidArgument("CUDA module " + key.first +
                                    " has no kernel '" + expr + "'." + hint);
    }
    // cuFuncGetParamInfo reports CUDA_ERROR_INVALID_VALUE past the last
    // parameter; anything else is a real failure.
    for (std::size_t i = 0;; ++i) {
      std::size_t offset = 0, size = 0;
      status = d.func_get_param_info(kernel.function, i, &offset, &size);
      if (status == kErrorInvalidValue) break;
      if (status != 0) return driver_error(status, "cuFuncGetParamInfo");
      kernel.param_sizes.push_back(size);
    }
    k = mod.kernels.emplace(expr, std::move(kernel)).first;
  }
  out = &k->second;
  return Error::Success();
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

// A scalar kernel argument: its bytes, zero-padded to 8, and its real size.
struct Scalar {
  std::array<unsigned char, 8> bytes{};
  std::size_t size = 0;
};

// `scalars` is a comma-separated list of little-endian hex byte strings, one
// per scalar argument, e.g. "0a000000,000000000000f03f" for (int 10, 1.0).
// pjrt_cuda_launch_attrs() writes it.
Error decode_scalars(std::string_view scalars, std::vector<Scalar> &out) {
  std::size_t start = 0;
  while (start < scalars.size()) {
    std::size_t end = scalars.find(',', start);
    if (end == std::string_view::npos) end = scalars.size();
    std::string_view hex = scalars.substr(start, end - start);
    if (hex.empty() || hex.size() % 2 != 0 || hex.size() > 16) {
      return Error::InvalidArgument("malformed `scalars` attribute");
    }
    Scalar s;
    s.size = hex.size() / 2;
    for (std::size_t i = 0; i < s.size; ++i) {
      int hi = hex_digit(hex[2 * i]), lo = hex_digit(hex[2 * i + 1]);
      if (hi < 0 || lo < 0) {
        return Error::InvalidArgument("malformed `scalars` attribute");
      }
      s.bytes[i] = static_cast<unsigned char>(hi * 16 + lo);
    }
    out.push_back(s);
    start = end + 1;
  }
  return Error::Success();
}

Error check_signature(const Kernel &kernel, std::string_view name,
                      std::size_t n_args, std::size_t n_rets,
                      const std::vector<Scalar> &scalars) {
  std::size_t n_ptrs = n_args + n_rets;
  std::size_t given = n_ptrs + scalars.size();
  const auto &sizes = kernel.param_sizes;
  if (sizes.size() != given) {
    return Error::InvalidArgument(
        "CUDA kernel '" + std::string(name) + "' takes " +
        std::to_string(sizes.size()) + " parameters, but was launched with " +
        std::to_string(given) + " (" + std::to_string(n_args) + " inputs, " +
        std::to_string(n_rets) + " outputs, " + std::to_string(scalars.size()) +
        " scalars).");
  }
  for (std::size_t i = 0; i < given; ++i) {
    bool is_ptr = i < n_ptrs;
    std::size_t expected = is_ptr ? sizeof(void *) : scalars[i - n_ptrs].size;
    if (sizes[i] != expected) {
      return Error::InvalidArgument(
          "Parameter " + std::to_string(i + 1) + " of CUDA kernel '" +
          std::string(name) + "' is " + std::to_string(sizes[i]) +
          " bytes, but " + (is_ptr ? "a device pointer" : "a scalar") + " of " +
          std::to_string(expected) + " bytes was passed.");
    }
  }
  return Error::Success();
}

}  // namespace

Error cuda_kernel_launch(void *stream, RemainingArgs args, RemainingRets rets,
                         std::string_view module, std::string_view kernel,
                         int32_t grid_x, int32_t grid_y, int32_t grid_z,
                         int32_t block_x, int32_t block_y, int32_t block_z,
                         int32_t shared_mem, std::string_view scalars) {
  // An empty grid has nothing to do, e.g. for zero-length operands.
  if (grid_x == 0 || grid_y == 0 || grid_z == 0) return Error::Success();

  Driver &d = driver();
  if (!d.loaded) return Error::Internal("CUDA driver (libcuda.so.1) not found");

  std::vector<Scalar> scalar_args;
  PJRT_RETURN_IF_ERROR(decode_scalars(scalars, scalar_args));

  // The context XLA runs this stream in; made current for loading and launch.
  void *ctx = nullptr;
  RETURN_IF_DRIVER_ERROR(d.stream_get_ctx(stream, &ctx), "cuStreamGetCtx");
  RETURN_IF_DRIVER_ERROR(d.ctx_push(ctx), "cuCtxPushCurrent");
  struct PopCtx {
    ~PopCtx() {
      void *c = nullptr;
      driver().ctx_pop(&c);
    }
  } pop;

  Kernel *k = nullptr;
  PJRT_RETURN_IF_ERROR(get_kernel(module, kernel, ctx, k));
  PJRT_RETURN_IF_ERROR(
      check_signature(*k, kernel, args.size(), rets.size(), scalar_args));

  std::vector<void *> ptrs;
  ptrs.reserve(args.size() + rets.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    auto buf = args.get<AnyBuffer>(i);
    if (buf.has_error()) return buf.error();
    ptrs.push_back(buf->untyped_data());
  }
  for (std::size_t i = 0; i < rets.size(); ++i) {
    auto buf = rets.get<AnyBuffer>(i);
    if (buf.has_error()) return buf.error();
    ptrs.push_back((*buf)->untyped_data());
  }

  // cuLaunchKernel reads each parameter through a pointer to it, taking as
  // many bytes as the kernel's signature says.
  std::vector<void *> params;
  params.reserve(ptrs.size() + scalar_args.size());
  for (auto &p : ptrs) params.push_back(&p);
  for (auto &s : scalar_args) params.push_back(s.bytes.data());

  if (shared_mem > kDefaultSharedMemLimit) {
    RETURN_IF_DRIVER_ERROR(
        d.func_set_attribute(k->function, kMaxDynamicSharedSize, shared_mem),
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
  }

  RETURN_IF_DRIVER_ERROR(
      d.launch_kernel(k->function, grid_x, grid_y, grid_z, block_x, block_y,
                      block_z, shared_mem, stream, params.data(), nullptr),
      "cuLaunchKernel('" + std::string(kernel) + "')");
  return Error::Success();
}

}  // namespace rpjrt

#else  // _WIN32

namespace rpjrt {
Error cuda_kernel_launch(void *, RemainingArgs, RemainingRets, std::string_view,
                         std::string_view, int32_t, int32_t, int32_t, int32_t,
                         int32_t, int32_t, int32_t, std::string_view) {
  return Error(ErrorCode::kUnimplemented,
               "CUDA kernels are not supported on Windows");
}
}  // namespace rpjrt

#endif  // _WIN32

namespace rpjrt {

// The host "implementation": registered so that running a CUDA kernel on the
// CPU fails with a clear message rather than "no handler registered".
Error cuda_kernel_host(RemainingArgs, RemainingRets, Dictionary) {
  return Error(ErrorCode::kUnimplemented,
               "pjrt_cuda_kernel only runs on CUDA devices; the program was "
               "compiled for the CPU.");
}

XLA_FFI_DEFINE_HANDLER(cuda_kernel_handler, cuda_kernel_launch,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<std::string_view>("module")
                           .Attr<std::string_view>("kernel")
                           .Attr<int32_t>("grid_x")
                           .Attr<int32_t>("grid_y")
                           .Attr<int32_t>("grid_z")
                           .Attr<int32_t>("block_x")
                           .Attr<int32_t>("block_y")
                           .Attr<int32_t>("block_z")
                           .Attr<int32_t>("shared_mem")
                           .Attr<std::string_view>("scalars"));

XLA_FFI_DEFINE_HANDLER(
    cuda_kernel_handler_host, cuda_kernel_host,
    Ffi::Bind().RemainingArgs().RemainingRets().Attrs<Dictionary>());

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_cuda_kernel_handler() {
  return R_MakeExternalPtr((void *)rpjrt::cuda_kernel_handler, R_NilValue,
                           R_NilValue);
}

// [[Rcpp::export]]
SEXP get_cuda_kernel_handler_host() {
  return R_MakeExternalPtr((void *)rpjrt::cuda_kernel_handler_host, R_NilValue,
                           R_NilValue);
}

// The id of a module: a hash of its contents, so the same module gets the
// same id in every session, which keeps the programs referring to it
// identical.
std::string module_id(const rpjrt::ModuleSpec &spec) {
  rpjrt::Fnv1a h;
  h.add(spec.source);
  h.add(spec.image);
  for (const auto &o : spec.options) h.add(o);
  for (const auto &k : spec.kernels) h.add(k);
  return h.hex();
}

// Registers a module, or updates the cache directory of one registered
// before, and returns its id.
// [[Rcpp::export]]
std::string impl_cuda_module_register(std::string source, std::string filename,
                                      std::vector<std::string> options,
                                      std::vector<std::string> kernels,
                                      Rcpp::RawVector image,
                                      std::string cache_dir) {
  rpjrt::ModuleSpec spec{source,
                         filename,
                         options,
                         kernels,
                         std::string(image.begin(), image.end()),
                         cache_dir};
  std::string id = module_id(spec);
  std::lock_guard<std::mutex> lock(rpjrt::registry_mu);
  auto [it, inserted] = rpjrt::registry.emplace(id, std::move(spec));
  if (!inserted) it->second.cache_dir = cache_dir;
  return id;
}

// Whether module `id` is registered; if so, its cache directory is updated.
// The cheap path for a module that is used over and over.
// [[Rcpp::export]]
bool impl_cuda_module_refresh(std::string id, std::string cache_dir) {
  std::lock_guard<std::mutex> lock(rpjrt::registry_mu);
  auto it = rpjrt::registry.find(id);
  if (it == rpjrt::registry.end()) return false;
  it->second.cache_dir = cache_dir;
  return true;
}

// Owning handles for PJRT C API objects: a PJRTHandle<T> is a std::unique_ptr
// whose deleter calls T's PJRT_*_Destroy. The deleter holds a raw PJRT_Api*, so
// the owner must keep the API alive at least as long as the handle -- for a
// class member, by declaring its shared_ptr<PJRT_Api> before the handle. The
// deleter must not throw (it runs during unwinding), so a destroy error is
// dropped. To own another object type, specialize PJRTDestroy for it.

#pragma once

#include <memory>

#include "utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

// How to free a T: `destroy(api, p)` calls T's PJRT_*_Destroy and returns its
// error, which the caller owns.
template <typename T>
struct PJRTDestroy;

template <>
struct PJRTDestroy<PJRT_LoadedExecutable> {
  static PJRT_Error *destroy(const PJRT_Api *api, PJRT_LoadedExecutable *p) {
    PJRT_LoadedExecutable_Destroy_Args args{};
    args.struct_size = sizeof(PJRT_LoadedExecutable_Destroy_Args);
    args.executable = p;
    return api->PJRT_LoadedExecutable_Destroy_(&args);
  }
};

template <>
struct PJRTDestroy<PJRT_Executable> {
  static PJRT_Error *destroy(const PJRT_Api *api, PJRT_Executable *p) {
    PJRT_Executable_Destroy_Args args{};
    args.struct_size = sizeof(PJRT_Executable_Destroy_Args);
    args.executable = p;
    return api->PJRT_Executable_Destroy_(&args);
  }
};

// A default-constructed deleter has no API to call through, so a handle must
// get its pointer from make_pjrt_handle() or by move, never by reset().
template <typename T>
class PJRTDeleter {
 public:
  PJRTDeleter() = default;
  explicit PJRTDeleter(const PJRT_Api *api) : api_(api) {}
  void operator()(T *p) const noexcept {
    destroy_error(api_, PJRTDestroy<T>::destroy(api_, p));
  }

 private:
  const PJRT_Api *api_ = nullptr;
};

template <typename T>
using PJRTHandle = std::unique_ptr<T, PJRTDeleter<T>>;

// Take ownership of `p`, freed through `api`.
template <typename T>
PJRTHandle<T> make_pjrt_handle(const PJRT_Api *api, T *p) {
  return PJRTHandle<T>(p, PJRTDeleter<T>(api));
}

}  // namespace rpjrt

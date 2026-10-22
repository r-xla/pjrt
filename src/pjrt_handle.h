// Owning handles for PJRT C API objects.
//
// A PjrtHandle<T> is a std::unique_ptr whose deleter calls T's PJRT_*_Destroy,
// so an object is freed on every path out of its scope -- including a C++
// exception thrown halfway through a constructor, which skips the destructor of
// the object being built but still destroys its fully-constructed members.
//
// The deleter holds the PJRT_Api as a raw pointer rather than a copy of the
// owner's std::shared_ptr<PJRT_Api>, so a handle costs no reference counting.
// Whoever owns a handle must therefore keep the PJRT_Api alive at least as long
// -- for a class member, by declaring its shared_ptr<PJRT_Api> *before* the
// handle, so that the handle is destroyed first.
//
// The deleter runs in destructors and during stack unwinding, so it must not
// throw: an error from the destroy call is dropped rather than raised.
//
// To own another object type, specialize PjrtDestroy for it.

#pragma once

#include <memory>

#include "utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

// How to free a T: `destroy(api, p)` calls T's PJRT_*_Destroy and returns its
// error, which the caller owns.
template <typename T>
struct PjrtDestroy;

template <>
struct PjrtDestroy<PJRT_LoadedExecutable> {
  static PJRT_Error *destroy(const PJRT_Api *api, PJRT_LoadedExecutable *p) {
    PJRT_LoadedExecutable_Destroy_Args args{};
    args.struct_size = sizeof(PJRT_LoadedExecutable_Destroy_Args);
    args.executable = p;
    return api->PJRT_LoadedExecutable_Destroy_(&args);
  }
};

template <>
struct PjrtDestroy<PJRT_Executable> {
  static PJRT_Error *destroy(const PJRT_Api *api, PJRT_Executable *p) {
    PJRT_Executable_Destroy_Args args{};
    args.struct_size = sizeof(PJRT_Executable_Destroy_Args);
    args.executable = p;
    return api->PJRT_Executable_Destroy_(&args);
  }
};

template <typename T>
class PjrtDeleter {
 public:
  PjrtDeleter() = default;
  explicit PjrtDeleter(const PJRT_Api *api) : api_(api) {}
  void operator()(T *p) const noexcept {
    destroy_error(api_, PjrtDestroy<T>::destroy(api_, p));
  }

 private:
  const PJRT_Api *api_ = nullptr;
};

template <typename T>
using PjrtHandle = std::unique_ptr<T, PjrtDeleter<T>>;

// Take ownership of `p`, freed through `api`.
template <typename T>
PjrtHandle<T> own(const PJRT_Api *api, T *p) {
  return PjrtHandle<T>(p, PjrtDeleter<T>(api));
}

}  // namespace rpjrt

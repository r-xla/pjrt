#pragma once

#include <memory>

#include "pjrt.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

class PJRTDevice {
 public:
  PJRT_Device *device;
  std::shared_ptr<PJRT_Api> api;
  PJRTDevice(PJRT_Device *device, std::shared_ptr<PJRT_Api> api);
};
}  // namespace rpjrt

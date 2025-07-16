#pragma once

#include "pjrt.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

class PJRTDevice {
 public:
  PJRT_Device *device;
  PJRTDevice(PJRT_Device *device);
};
}  // namespace rpjrt

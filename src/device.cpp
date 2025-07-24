#include "device.h"

namespace rpjrt {
PJRTDevice::PJRTDevice(PJRT_Device *device, std::shared_ptr<PJRT_Api> api)
    : device(device), api(api) {}
}  // namespace rpjrt

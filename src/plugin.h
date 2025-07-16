#pragma once

#include "client.h"
#include "pjrt.h"

namespace rpjrt {

class PJRTPlugin {
 public:
  PJRTPlugin(const std::string &path);
  void initialize();
  std::unique_ptr<PJRTClient> client_create();
  std::shared_ptr<PJRT_API> api;

 private:
  static PJRT_API *load_pjrt_plugin(const std::string &path);
};

}  // namespace rpjrt

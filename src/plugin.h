#pragma once

#include <Rcpp.h>

#include "client.h"

namespace rpjrt {

class PJRTPlugin {
 public:
  PJRTPlugin(const std::string &path);
  void initialize();
  std::unique_ptr<PJRTClient> client_create(
      const std::vector<std::pair<std::string, int64_t>> &options = {});
  std::pair<int, int> pjrt_api_version() const;
  std::vector<std::pair<std::string, SEXP>> attributes() const;
  std::shared_ptr<PJRT_Api> api;

 private:
  static PJRT_Api *load_pjrt_plugin(const std::string &path);
};

}  // namespace rpjrt

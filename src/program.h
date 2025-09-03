#pragma once

#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {
enum PJRTProgramFormat { HLO, MLIR };

class PJRTProgram {
 public:
  PJRTProgram(const std::string &fname, const PJRTProgramFormat &format);
  PJRTProgramFormat format() const;
  std::string repr(int n) const;
  std::string code;
  PJRT_Program program;

 private:
  static std::string load_program_from_file(const std::string &fname,
                                            const PJRTProgramFormat &format);
  static std::string parse_hlo_program(const std::vector<char> &buffer);
  static PJRT_Program create_program(std::string &code,
                                     const PJRTProgramFormat &format);
};
}  // namespace rpjrt

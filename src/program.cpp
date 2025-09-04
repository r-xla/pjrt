#include "program.h"

#include <fstream>

#include "xla/service/hlo.pb.h"

namespace rpjrt {

// Implementation of PJRTProgram methods
PJRTProgram::PJRTProgram(const std::string &fname,
                         const PJRTProgramFormat &format)
    : code(load_program_from_file(fname, format)),
      program(create_program(this->code, format)) {}

PJRTProgramFormat PJRTProgram::format() const {
  if (strcmp(program.format, "hlo") == 0) {
    return HLO;
  } else if (strcmp(program.format, "mlir") == 0) {
    return MLIR;
  } else {
    throw std::runtime_error("Unknown program format");
  }
}

std::string PJRTProgram::load_program_from_file(
    const std::string &fname, const PJRTProgramFormat &format) {
  std::ifstream input(fname, std::ios::binary | std::ios::ate);
  std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);  // rewind
  std::vector<char> buffer(size);
  input.read(buffer.data(), size);

  switch (format) {
    case HLO:
      return parse_hlo_program(buffer);
      break;
    case MLIR:
      return std::string(buffer.data(), buffer.size());
      break;
  };

  throw std::runtime_error("Unknown program format");
}

PJRT_Program PJRTProgram::create_program(std::string &code,
                                         const PJRTProgramFormat &format) {
  PJRT_Program program{};
  program.struct_size = sizeof(PJRT_Program);

  program.code = code.data();
  program.code_size = code.size();

  switch (format) {
    case HLO:
      program.format = "hlo";
      program.format_size = strlen("hlo");
      break;
    case MLIR:
      program.format = "mlir";
      program.format_size = strlen("mlir");
      break;
    default:
      throw std::runtime_error("Unknown program format");
  }

  return program;
}

std::string PJRTProgram::parse_hlo_program(const std::vector<char> &buffer) {
  xla::HloModuleProto hlo_proto{};
  hlo_proto.ParseFromArray(buffer.data(), buffer.size());

  return hlo_proto.SerializeAsString();
}

std::string PJRTProgram::repr(int n) const {
  auto repr = "PJRTProgram(format=" + std::string(program.format) +
              ", code_size=" + std::to_string(program.code_size) + ")";
  auto format = this->format();
  std::string debug("");
  switch (format) {
    case HLO: {
      xla::HloModuleProto hlo_proto{};
      hlo_proto.ParseFromArray(this->code.data(), this->code.size());
      debug = hlo_proto.DebugString();
    } break;
    case MLIR:
      debug = this->code;
      break;
  }

  // debug must not be larger than n lines
  // Find the nth newline position directly without counting all lines
  size_t pos = 0;
  int lines_found = 0;

  while (lines_found < n && pos != std::string::npos) {
    pos = debug.find('\n', pos);
    if (pos != std::string::npos) {
      lines_found++;
      pos++;  // Move past the newline
    }
  }

  // If we found n newlines, truncate there
  if (lines_found == n && pos != std::string::npos) {
    // Check if there's more content after the nth line
    if (pos < debug.length()) {
      debug = debug.substr(0, pos - 1) +
              "\n...";  // pos-1 to include the nth newline
    }
  }

  return repr + "\n" + debug;
}

}  // namespace rpjrt

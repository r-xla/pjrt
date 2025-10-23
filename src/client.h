#pragma once
#include <optional>
#include <vector>

#include "buffer.h"
#include "program.h"
#include "proto/xla/pjrt/proto/compile_options.pb.h"

namespace rpjrt {

class PJRTBuildOptions {
 public:
  std::unique_ptr<xla::ExecutableBuildOptionsProto> build_options;
  PJRTBuildOptions();
  PJRTBuildOptions(const int num_replicas, const int num_partitions,
                   const int device_ordinal);
  PJRTBuildOptions clone() const;
};

class PJRTCompileOptions {
 public:
  xla::CompileOptionsProto compile_options{};
  PJRTCompileOptions(PJRTBuildOptions build_options);
  std::string serialize();
};

class PJRTExecuteOptions {
 public:
  std::vector<int64_t> non_donatable_input_indices;
  int launch_id;

  PJRTExecuteOptions();
  PJRTExecuteOptions(const std::vector<int64_t> &non_donatable_indices,
                     int launch_id = 0);
};

class PJRTLoadedExecutable {
 public:
  PJRT_LoadedExecutable *executable;
  std::shared_ptr<PJRT_Api> api;
  PJRTLoadedExecutable(PJRT_LoadedExecutable *executable,
                       std::shared_ptr<PJRT_Api> api);
  std::vector<std::unique_ptr<PJRTBuffer>> execute(
      std::vector<PJRTBuffer *> input,
      const PJRTExecuteOptions &options = PJRTExecuteOptions{});
  ~PJRTLoadedExecutable();
};

class PJRTClient {
 public:
  PJRT_Client *client;
  std::shared_ptr<PJRT_Api> api;
  PJRTClient(PJRT_Client *client, std::shared_ptr<PJRT_Api> api);
  ~PJRTClient();
  std::vector<PJRT_Device *> devices();
  std::unique_ptr<PJRTLoadedExecutable> compile(
      const PJRTProgram &program, PJRTCompileOptions &compile_options);
  std::unique_ptr<PJRTBuffer> buffer_from_host(
      void *data, const std::optional<std::vector<int64_t>> &dims,
      const std::optional<std::vector<int64_t>> &strides,
      PJRT_Buffer_Type dtype, PJRT_Device *device = nullptr);
  std::string platform();
};

}  // namespace rpjrt

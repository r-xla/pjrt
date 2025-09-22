#include "client.h"

#include <memory>

#include "buffer.h"
#include "utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace rpjrt {

PJRTClient::PJRTClient(PJRT_Client *client, std::shared_ptr<PJRT_Api> api)
    : client(client), api(api) {}

PJRTClient::~PJRTClient() {
  PJRT_Client_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_Client_Destroy_Args);
  args.client = this->client;
  check_err(this->api.get(), this->api->PJRT_Client_Destroy_(&args));
}

std::vector<PJRT_Device *> PJRTClient::devices() {
  PJRT_Client_AddressableDevices_Args args{};
  args.client = this->client;
  args.struct_size = sizeof(PJRT_Client_AddressableDevices_Args);
  check_err(this->api.get(), this->api->PJRT_Client_AddressableDevices_(&args));
  return std::vector(args.addressable_devices,
                     args.addressable_devices + args.num_addressable_devices);
}

std::unique_ptr<PJRTLoadedExecutable> PJRTClient::compile(
    const PJRTProgram &program, PJRTCompileOptions &compile_options) {
  PJRT_Client_Compile_Args args{};
  args.struct_size = sizeof(PJRT_Client_Compile_Args);

  args.client = this->client;
  args.program = &program.program;

  auto opts = compile_options.serialize();
  args.compile_options = opts.data();
  args.compile_options_size = opts.size();

  check_err(this->api.get(), this->api->PJRT_Client_Compile_(&args));
  return std::make_unique<PJRTLoadedExecutable>(args.executable, this->api);
}

void BufferFromHostAndWait(const PJRT_Api *api,
                           PJRT_Client_BufferFromHostBuffer_Args *args) {
  check_err(api, api->PJRT_Client_BufferFromHostBuffer_(args));

  PJRT_Event_Await_Args event_args = {0};
  event_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  event_args.event = args->done_with_host_buffer;
  check_err(api, api->PJRT_Event_Await_(&event_args));

  PJRT_Event_Destroy_Args efree_args;
  efree_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  efree_args.event = args->done_with_host_buffer;
  check_err(api, api->PJRT_Event_Destroy_(&efree_args));
}

std::unique_ptr<PJRTBuffer> PJRTClient::buffer_from_host(
    void *data, const std::optional<std::vector<int64_t>> &dims,
    const std::optional<std::vector<int64_t>> &strides,
    PJRT_Buffer_Type dtype) {
  const auto devices = this->devices();

  // Initialize args to zero to ensure optional fields are null.
  PJRT_Client_BufferFromHostBuffer_Args args{};
  args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
  args.client = this->client;
  args.data = data;
  args.type = dtype;
  // Set dimensions for the buffer
  args.dims = dims.has_value() ? dims->data() : nullptr;
  args.num_dims = dims.has_value() ? dims->size() : 0;
  // No custom strides: assume dense layout
  args.byte_strides = strides.has_value() ? strides->data() : nullptr;
  args.num_byte_strides = strides.has_value() ? strides->size() : 0;
  args.host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  args.device = devices[0];
  // No preallocated memory
  args.memory = nullptr;

  BufferFromHostAndWait(this->api.get(), &args);
  return std::make_unique<PJRTBuffer>(args.buffer, this->api);
}

// PJRTBuildOptions implementations
PJRTBuildOptions::PJRTBuildOptions()
    : build_options(std::make_unique<xla::ExecutableBuildOptionsProto>()) {
  this->build_options->set_num_replicas(1);
  this->build_options->set_num_partitions(1);
  this->build_options->set_device_ordinal(-1);
}

PJRTBuildOptions::PJRTBuildOptions(const int num_replicas,
                                   const int num_partitions,
                                   const int device_ordinal)
    : build_options(std::make_unique<xla::ExecutableBuildOptionsProto>()) {
  this->build_options->set_num_replicas(num_replicas);
  this->build_options->set_num_partitions(num_partitions);
  this->build_options->set_device_ordinal(device_ordinal);
}

PJRTBuildOptions PJRTBuildOptions::clone() const {
  PJRTBuildOptions clone;
  clone.build_options =
      std::make_unique<xla::ExecutableBuildOptionsProto>(*this->build_options);
  return clone;
}

// PJRTCompileOptions implementations
PJRTCompileOptions::PJRTCompileOptions(PJRTBuildOptions build_options) {
  compile_options.set_compile_portable_executable(false);
  compile_options.set_profile_version(0);
  compile_options.set_parameter_is_tupled_arguments(false);
  compile_options.set_allocated_executable_build_options(
      build_options.build_options.release());
}

std::string PJRTCompileOptions::serialize() {
  return compile_options.SerializeAsString();
}

// PJRTExecuteOptions implementations
PJRTExecuteOptions::PJRTExecuteOptions() : launch_id(0) {}

PJRTExecuteOptions::PJRTExecuteOptions(
    const std::vector<int64_t> &non_donatable_indices, int launch_id)
    : non_donatable_input_indices(non_donatable_indices),
      launch_id(launch_id) {}

PJRTLoadedExecutable::PJRTLoadedExecutable(PJRT_LoadedExecutable *executable,
                                           std::shared_ptr<PJRT_Api> api)
    : executable(executable), api(api) {}

PJRTLoadedExecutable::~PJRTLoadedExecutable() {
  PJRT_LoadedExecutable_Destroy_Args args{};
  args.struct_size = sizeof(PJRT_LoadedExecutable_Destroy_Args);
  args.executable = this->executable;
  check_err(this->api.get(), this->api->PJRT_LoadedExecutable_Destroy_(&args));
}

std::vector<std::unique_ptr<PJRTBuffer>> PJRTLoadedExecutable::execute(
    std::vector<PJRTBuffer *> input, const PJRTExecuteOptions &options) {
  PJRT_ExecuteOptions exec_options{};
  exec_options.struct_size = sizeof(PJRT_ExecuteOptions);
  exec_options.launch_id = options.launch_id;
  exec_options.non_donatable_input_indices =
      options.non_donatable_input_indices.empty()
          ? nullptr
          : options.non_donatable_input_indices.data();
  exec_options.num_non_donatable_input_indices =
      options.non_donatable_input_indices.size();

  PJRT_LoadedExecutable_Execute_Args exec_args{};
  exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  exec_args.executable = this->executable;
  exec_args.options = &exec_options;

  // This is the actual parameters
  std::vector<PJRT_Buffer *> inner(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    inner[i] = input[i]->buffer;
  }
  // We need an outer list, because its one input per execution device.
  // Currently we only support one device, so we have a single element in the
  // outer list.
  std::vector<PJRT_Buffer *const *> outer;
  if (input.empty()) {
    // When there are no inputs, we still need to provide a valid pointer
    // but the inner vector will be empty
    outer = {nullptr};
  } else {
    outer = {inner.data()};
  }
  exec_args.argument_lists = outer.data();
  exec_args.num_args = input.size();
  exec_args.num_devices = 1;

  exec_args.execute_device = nullptr;

  // Get the number of outputs from the executable
  PJRT_LoadedExecutable_GetExecutable_Args get_exec_args{};
  get_exec_args.struct_size = sizeof(PJRT_LoadedExecutable_GetExecutable_Args);
  get_exec_args.loaded_executable = this->executable;
  check_err(this->api.get(),
            this->api->PJRT_LoadedExecutable_GetExecutable_(&get_exec_args));

  PJRT_Executable_NumOutputs_Args num_outputs_args{};
  num_outputs_args.struct_size = sizeof(PJRT_Executable_NumOutputs_Args);
  num_outputs_args.executable = get_exec_args.executable;
  check_err(this->api.get(),
            this->api->PJRT_Executable_NumOutputs_(&num_outputs_args));

  size_t num_outputs = num_outputs_args.num_outputs;

  // Prepare output buffer storage
  std::vector<PJRT_Buffer *> inner_out(num_outputs);
  std::vector<PJRT_Buffer **> outer_out = {inner_out.data()};

  exec_args.output_lists = outer_out.data();

  check_err(this->api.get(),
            this->api->PJRT_LoadedExecutable_Execute_(&exec_args));

  std::vector<std::unique_ptr<PJRTBuffer>> out;
  for (size_t i = 0; i < num_outputs; ++i) {
    out.push_back(std::make_unique<PJRTBuffer>(outer_out[0][i], this->api));
  }

  // Clean up the executable we got
  PJRT_Executable_Destroy_Args destroy_exec_args{};
  destroy_exec_args.struct_size = sizeof(PJRT_Executable_Destroy_Args);
  destroy_exec_args.executable = get_exec_args.executable;
  check_err(this->api.get(),
            this->api->PJRT_Executable_Destroy_(&destroy_exec_args));

  return out;
};

std::string PJRTClient::platform() {
  PJRT_Client_PlatformName_Args args{};
  args.struct_size = sizeof(PJRT_Client_PlatformName_Args);
  args.client = this->client;
  check_err(this->api.get(), this->api->PJRT_Client_PlatformName_(&args));
  return std::string(args.platform_name, args.platform_name_size);
}

}  // namespace rpjrt

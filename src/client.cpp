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
    const PJRTProgram &program, PJRTCompileOptions &compile_options,
    PJRTDevice &device) {
  // Get the device's local hardware ID and set it as the device_ordinal
  PJRT_Device_LocalHardwareId_Args hw_args{};
  hw_args.struct_size = sizeof(PJRT_Device_LocalHardwareId_Args);
  hw_args.device = device.device;
  check_err(this->api.get(), this->api->PJRT_Device_LocalHardwareId_(&hw_args));
  compile_options.compile_options.mutable_executable_build_options()
      ->set_device_ordinal(hw_args.local_hardware_id);

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

AsyncBufferFromHostResult PJRTClient::buffer_from_host_async(
    void *data, const std::optional<std::vector<int64_t>> &dims,
    const std::optional<std::vector<int64_t>> &strides, PJRT_Buffer_Type dtype,
    PJRT_Device *device) {
  // If no device is specified, use the first device
  if (device == nullptr) {
    const auto devices = this->devices();
    device = devices[0];
  }

  PJRT_Client_BufferFromHostBuffer_Args args{};
  args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
  args.client = this->client;
  args.data = data;
  args.type = dtype;
  args.dims = dims.has_value() ? dims->data() : nullptr;
  args.num_dims = dims.has_value() ? dims->size() : 0;
  args.byte_strides = strides.has_value() ? strides->data() : nullptr;
  args.num_byte_strides = strides.has_value() ? strides->size() : 0;
  args.host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  args.device = device;
  args.memory = nullptr;

  check_err(this->api.get(),
            this->api->PJRT_Client_BufferFromHostBuffer_(&args));

  AsyncBufferFromHostResult result;
  result.buffer = std::make_unique<PJRTBuffer>(args.buffer, this->api);

  // Return the event so caller can wait for transfer completion
  if (args.done_with_host_buffer != nullptr) {
    result.event =
        std::make_unique<PJRTEvent>(args.done_with_host_buffer, this->api);
  }

  return result;
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

AsyncExecuteResult PJRTLoadedExecutable::execute_async(
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

  // Allocate storage for device completion events (one per device)
  PJRT_Event *completion_event = nullptr;
  exec_args.device_complete_events = &completion_event;

  check_err(this->api.get(),
            this->api->PJRT_LoadedExecutable_Execute_(&exec_args));

  // Clean up the device completion event
  if (completion_event != nullptr) {
    PJRTEvent event(completion_event, this->api);
    // Event is destroyed when it goes out of scope
  }

  // Build result
  AsyncExecuteResult result;
  for (size_t i = 0; i < num_outputs; ++i) {
    auto buf = std::make_unique<PJRTBuffer>(outer_out[0][i], this->api);
    result.buffers.push_back(std::move(buf));
  }

  // Clean up the executable we got
  PJRT_Executable_Destroy_Args destroy_exec_args{};
  destroy_exec_args.struct_size = sizeof(PJRT_Executable_Destroy_Args);
  destroy_exec_args.executable = get_exec_args.executable;
  check_err(this->api.get(),
            this->api->PJRT_Executable_Destroy_(&destroy_exec_args));

  return result;
};

std::string PJRTClient::platform() {
  PJRT_Client_PlatformName_Args args{};
  args.struct_size = sizeof(PJRT_Client_PlatformName_Args);
  args.client = this->client;
  check_err(this->api.get(), this->api->PJRT_Client_PlatformName_(&args));
  return std::string(args.platform_name, args.platform_name_size);
}

}  // namespace rpjrt

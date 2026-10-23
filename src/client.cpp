#include "client.h"

#include <Rcpp.h>

#include <cctype>
#include <memory>
#include <string>
#include <utility>

#include "buffer.h"
#include "utils.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/service/hlo.pb.h"

namespace rpjrt {

PJRTClient::PJRTClient(PJRTHandle<PJRT_Client> client,
                       std::shared_ptr<PJRT_Api> api)
    : api(std::move(api)), client(std::move(client)) {}

std::vector<PJRT_Device *> PJRTClient::devices() {
  PJRT_Client_AddressableDevices_Args args{};
  args.client = this->client.get();
  args.struct_size = sizeof(PJRT_Client_AddressableDevices_Args);
  check_err(this->api.get(), this->api->PJRT_Client_AddressableDevices_(&args));
  return std::vector(args.addressable_devices,
                     args.addressable_devices + args.num_addressable_devices);
}

std::unique_ptr<PJRTLoadedExecutable> PJRTClient::compile(
    const PJRTProgram &program, PJRTCompileOptions &compile_options,
    PJRTDevice &device) {
  // device_ordinal is a local hardware (StreamExecutor) ordinal, while the
  // device_assignment uses global PJRT device IDs. These coincide on a single
  // host but diverge under distributed PJRT clients, so fetch both. Setting
  // only device_ordinal is insufficient: the CPU plugin exposes multiple
  // virtual devices sharing one StreamExecutor ordinal, so device_assignment
  // is needed to disambiguate them.
  PJRT_Device_GetDescription_Args desc_args{};
  desc_args.struct_size = sizeof(PJRT_Device_GetDescription_Args);
  desc_args.device = device.device;
  check_err(this->api.get(),
            this->api->PJRT_Device_GetDescription_(&desc_args));

  PJRT_DeviceDescription_Id_Args id_args{};
  id_args.struct_size = sizeof(PJRT_DeviceDescription_Id_Args);
  id_args.device_description = desc_args.device_description;
  check_err(this->api.get(), this->api->PJRT_DeviceDescription_Id_(&id_args));

  PJRT_Device_LocalHardwareId_Args hw_args{};
  hw_args.struct_size = sizeof(PJRT_Device_LocalHardwareId_Args);
  hw_args.device = device.device;
  check_err(this->api.get(), this->api->PJRT_Device_LocalHardwareId_(&hw_args));

  auto *build_opts =
      compile_options.compile_options.mutable_executable_build_options();
  build_opts->set_device_ordinal(hw_args.local_hardware_id);

  auto *da = build_opts->mutable_device_assignment();
  da->set_replica_count(build_opts->num_replicas());
  da->set_computation_count(build_opts->num_partitions());
  auto *cd = da->add_computation_devices();
  cd->add_replica_device_ids(id_args.id);

  PJRT_Client_Compile_Args args{};
  args.struct_size = sizeof(PJRT_Client_Compile_Args);

  args.client = this->client.get();
  args.program = &program.program;

  auto opts = compile_options.serialize();
  args.compile_options = opts.data();
  args.compile_options_size = opts.size();

  check_err(this->api.get(), this->api->PJRT_Client_Compile_(&args));
  return std::make_unique<PJRTLoadedExecutable>(
      make_pjrt_handle(this->api.get(), args.executable), this->api,
      program.code, program.format(), this->is_cpu());
}

AsyncBufferFromHostResult PJRTClient::buffer_from_host_async(
    void *data, const std::optional<std::vector<int64_t>> &dims,
    const std::optional<std::vector<int64_t>> &strides, PJRT_Buffer_Type dtype,
    PJRT_Device *device, PJRT_HostBufferSemantics semantics) {
  // If no device is specified, use the first device
  if (device == nullptr) {
    const auto devices = this->devices();
    device = devices[0];
  }

  PJRT_Client_BufferFromHostBuffer_Args args{};
  args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
  args.client = this->client.get();
  args.data = data;
  args.type = dtype;
  args.dims = dims.has_value() ? dims->data() : nullptr;
  args.num_dims = dims.has_value() ? dims->size() : 0;
  args.byte_strides = strides.has_value() ? strides->data() : nullptr;
  args.num_byte_strides = strides.has_value() ? strides->size() : 0;
  args.host_buffer_semantics = semantics;
  args.device = device;
  args.memory = nullptr;

  try_alloc(
      this->api.get(),
      [&] { return this->api->PJRT_Client_BufferFromHostBuffer_(&args); },
      /*suppress_logs=*/!this->is_cpu());

  AsyncBufferFromHostResult result;
  result.buffer = std::make_unique<PJRTBuffer>(
      make_pjrt_handle(this->api.get(), args.buffer), this->api);

  // Return the event so caller can wait for transfer completion
  if (args.done_with_host_buffer != nullptr) {
    result.event = std::make_unique<PJRTEvent>(
        make_pjrt_handle(this->api.get(), args.done_with_host_buffer),
        this->api);
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

PJRTLoadedExecutable::PJRTLoadedExecutable(
    PJRTHandle<PJRT_LoadedExecutable> executable_handle,
    std::shared_ptr<PJRT_Api> client_api, const std::string &program_code,
    PJRTProgramFormat program_format, bool is_cpu)
    : api(std::move(client_api)),
      executable(std::move(executable_handle)),
      is_cpu_(is_cpu) {
  // A throw below skips the destructor but still destroys `executable`.
  load_input_output_aliases_(program_code, program_format);
  load_num_outputs_();
}

// Query and cache the executable's output count. This is a static property of
// the compiled program, so resolving it per execute (GetExecutable + NumOutputs
// + Executable_Destroy) would be three plugin-boundary calls of pure overhead
// on the hot dispatch path.
void PJRTLoadedExecutable::load_num_outputs_() {
  PJRT_LoadedExecutable_GetExecutable_Args get_exec_args{};
  get_exec_args.struct_size = sizeof(PJRT_LoadedExecutable_GetExecutable_Args);
  get_exec_args.loaded_executable = this->executable.get();
  check_err(this->api.get(),
            this->api->PJRT_LoadedExecutable_GetExecutable_(&get_exec_args));
  // The executable handle is a new object that this call must free.
  PJRTHandle<PJRT_Executable> exec =
      make_pjrt_handle(this->api.get(), get_exec_args.executable);

  PJRT_Executable_NumOutputs_Args num_outputs_args{};
  num_outputs_args.struct_size = sizeof(PJRT_Executable_NumOutputs_Args);
  num_outputs_args.executable = exec.get();
  check_err(this->api.get(),
            this->api->PJRT_Executable_NumOutputs_(&num_outputs_args));
  this->num_outputs_ = num_outputs_args.num_outputs;
}

namespace {

// Parse input->output donation aliases out of the entry function's signature
// in an MLIR module. stablehlo emits donation as a per-argument attribute
//   %0: tensor<...> {tf.aliasing_output = M : i32}
// where the donated argument's positional index is the PJRT parameter number
// and M is the aliased output index.
std::vector<rpjrt::PJRTInputOutputAlias> parse_mlir_aliases(
    const std::string &code) {
  std::vector<rpjrt::PJRTInputOutputAlias> aliases;

  // Locate the `(` that opens @main's argument list. MLIR allows whitespace
  // between the symbol name and the list, so we can't assume `@main(` is
  // contiguous: find `@main`, skip any whitespace, and require a `(` next. The
  // whitespace check also rejects same-prefix symbols like `@main2(`, whose
  // next character is an identifier char rather than whitespace or `(`.
  static const std::string kMain = "@main";
  size_t paren_pos = std::string::npos;
  for (size_t m = code.find(kMain); m != std::string::npos;
       m = code.find(kMain, m + kMain.size())) {
    size_t p = m + kMain.size();
    while (p < code.size() &&
           std::isspace(static_cast<unsigned char>(code[p]))) {
      p++;
    }
    if (p < code.size() && code[p] == '(') {
      paren_pos = p;
      break;
    }
  }
  if (paren_pos == std::string::npos) return aliases;

  static const std::string kAttr = "tf.aliasing_output";
  int arg_index = 0, paren = 0, bracket = 0;
  for (size_t i = paren_pos /* at the '(' */; i < code.size(); ++i) {
    const char c = code[i];
    if (c == '(') {
      paren++;
    } else if (c == ')') {
      if (--paren == 0) break;
    } else if (c == '<' || c == '[' || c == '{') {
      bracket++;
    } else if (c == '>' || c == ']' || c == '}') {
      if (bracket > 0) bracket--;
    } else if (c == ',' && paren == 1 && bracket == 0) {
      arg_index++;
    } else if (code.compare(i, kAttr.size(), kAttr) == 0) {
      size_t j = code.find('=', i + kAttr.size());
      if (j == std::string::npos) continue;
      j++;
      while (j < code.size() &&
             std::isspace(static_cast<unsigned char>(code[j]))) {
        j++;
      }
      size_t end = j;
      while (end < code.size() &&
             std::isdigit(static_cast<unsigned char>(code[end]))) {
        end++;
      }
      if (end > j) {
        aliases.push_back({arg_index, std::stoi(code.substr(j, end - j))});
      }
    }
  }
  return aliases;
}

// Input/output aliasing for HLO-format programs is not supported; donation is
// only wired up for the MLIR (stablehlo) path. We don't silently ignore a
// declared HLO alias, though: PJRT would still donate the input at runtime
// while the R layer kept treating it as live, reading freed memory and
// double-freeing the keepalive. So inspect the module and reject the
// unsupported case loudly. HLO programs without aliasing are unaffected.
void reject_hlo_aliases(const std::string &code) {
  // `code` is the HloModuleProto that PJRTProgram already round-tripped through
  // protobuf and that compilation succeeded on, so ParseFromString cannot fail
  // here; it only loads the message.
  xla::HloModuleProto module;
  module.ParseFromString(code);
  if (module.has_input_output_alias()) {
    Rcpp::stop(
        "Input/output aliasing is not supported for HLO-format programs; "
        "use an MLIR (stablehlo) program instead.");
  }
}

}  // namespace

// Cache the input->output donation aliases declared in the program we just
// compiled. We read them from the program source we already hold rather than
// from the plugin's optimized executable: the aliasing is part of the program
// (XLA preserves caller-specified donation), so this needs no plugin support
// and works identically on CPU and CUDA. Only the MLIR path extracts
// aliases; the HLO path rejects them (see reject_hlo_aliases).
void PJRTLoadedExecutable::load_input_output_aliases_(
    const std::string &program_code, PJRTProgramFormat program_format) {
  switch (program_format) {
    case MLIR:
      aliases_ = parse_mlir_aliases(program_code);
      break;
    case HLO:
      reject_hlo_aliases(program_code);
      break;
  }
}

std::vector<PJRT_Device *> PJRTLoadedExecutable::addressable_devices() {
  PJRT_LoadedExecutable_AddressableDevices_Args args{};
  args.struct_size = sizeof(PJRT_LoadedExecutable_AddressableDevices_Args);
  args.executable = this->executable.get();
  check_err(this->api.get(),
            this->api->PJRT_LoadedExecutable_AddressableDevices_(&args));
  return std::vector(args.addressable_devices,
                     args.addressable_devices + args.num_addressable_devices);
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
  exec_args.executable = this->executable.get();
  exec_args.options = &exec_options;

  // This is the actual parameters
  std::vector<PJRT_Buffer *> inner(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    inner[i] = input[i]->buffer.get();
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

  // Output count is cached at construction (static per executable).
  size_t num_outputs = this->num_outputs_;

  // Prepare output buffer storage
  std::vector<PJRT_Buffer *> inner_out(num_outputs);
  std::vector<PJRT_Buffer **> outer_out = {inner_out.data()};

  exec_args.output_lists = outer_out.data();

  // Request a per-device completion event. It becomes ready when the execution
  // finishes reading its inputs, which is how the caller bounds the lifetime of
  // zero-copy input host keepalives (individual output buffers can become ready
  // at different times, so they are not a reliable "execution done" signal).
  // Length must equal num_devices (== 1 here). Per-buffer readiness still uses
  // PJRT_Buffer_ReadyEvent independently.
  std::vector<PJRT_Event *> complete_events(1, nullptr);
  exec_args.device_complete_events = complete_events.data();

  // Reserved up front: filling it after Execute must not allocate.
  std::vector<PJRTHandle<PJRT_Buffer>> owned_out;
  owned_out.reserve(num_outputs);

  try_alloc(
      this->api.get(),
      [&] { return this->api->PJRT_LoadedExecutable_Execute_(&exec_args); },
      /*suppress_logs=*/!this->is_cpu_);

  // Own every handle Execute returned before allocating anything, so that a
  // failed allocation below cannot leak the ones not yet wrapped.
  PJRTHandle<PJRT_Event> complete_event =
      make_pjrt_handle(this->api.get(), complete_events[0]);
  for (size_t i = 0; i < num_outputs; ++i) {
    owned_out.push_back(make_pjrt_handle(this->api.get(), inner_out[i]));
  }

  AsyncExecuteResult result;
  if (complete_event != nullptr) {
    result.complete_event =
        std::make_unique<PJRTEvent>(std::move(complete_event), this->api);
  }
  for (auto &out : owned_out) {
    result.buffers.push_back(
        std::make_unique<PJRTBuffer>(std::move(out), this->api));
  }

  return result;
};

std::string PJRTClient::platform() {
  PJRT_Client_PlatformName_Args args{};
  args.struct_size = sizeof(PJRT_Client_PlatformName_Args);
  args.client = this->client.get();
  check_err(this->api.get(), this->api->PJRT_Client_PlatformName_(&args));
  return std::string(args.platform_name, args.platform_name_size);
}

bool PJRTClient::is_cpu() {
  if (!is_cpu_.has_value()) {
    is_cpu_ = (platform() == "cpu");
  }
  return *is_cpu_;
}

}  // namespace rpjrt

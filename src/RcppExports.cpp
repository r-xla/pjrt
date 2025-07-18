// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "pjrt_types.h"
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// impl_plugin_load
Rcpp::XPtr<rpjrt::PJRTPlugin> impl_plugin_load(const std::string& path);
RcppExport SEXP _pjrt_impl_plugin_load(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type path(pathSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_plugin_load(path));
    return rcpp_result_gen;
END_RCPP
}
// impl_plugin_client_create
Rcpp::XPtr<rpjrt::PJRTClient> impl_plugin_client_create(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin);
RcppExport SEXP _pjrt_impl_plugin_client_create(SEXP pluginSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTPlugin> >::type plugin(pluginSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_plugin_client_create(plugin));
    return rcpp_result_gen;
END_RCPP
}
// impl_program_load
Rcpp::XPtr<rpjrt::PJRTProgram> impl_program_load(const std::string& fname, const std::string& format);
RcppExport SEXP _pjrt_impl_program_load(SEXP fnameSEXP, SEXP formatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type fname(fnameSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type format(formatSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_program_load(fname, format));
    return rcpp_result_gen;
END_RCPP
}
// impl_program_repr
std::string impl_program_repr(Rcpp::XPtr<rpjrt::PJRTProgram> program, int n);
RcppExport SEXP _pjrt_impl_program_repr(SEXP programSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTProgram> >::type program(programSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_program_repr(program, n));
    return rcpp_result_gen;
END_RCPP
}
// impl_build_options_create
Rcpp::XPtr<rpjrt::PJRTBuildOptions> impl_build_options_create(const int num_replicas, const int num_partitions, const int device_ordinal);
RcppExport SEXP _pjrt_impl_build_options_create(SEXP num_replicasSEXP, SEXP num_partitionsSEXP, SEXP device_ordinalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const int >::type num_replicas(num_replicasSEXP);
    Rcpp::traits::input_parameter< const int >::type num_partitions(num_partitionsSEXP);
    Rcpp::traits::input_parameter< const int >::type device_ordinal(device_ordinalSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_build_options_create(num_replicas, num_partitions, device_ordinal));
    return rcpp_result_gen;
END_RCPP
}
// impl_compile_options_create
Rcpp::XPtr<rpjrt::PJRTCompileOptions> impl_compile_options_create(Rcpp::XPtr<rpjrt::PJRTBuildOptions> build_options);
RcppExport SEXP _pjrt_impl_compile_options_create(SEXP build_optionsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTBuildOptions> >::type build_options(build_optionsSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_compile_options_create(build_options));
    return rcpp_result_gen;
END_RCPP
}
// impl_client_program_compile
Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> impl_client_program_compile(Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTProgram> program, Rcpp::XPtr<rpjrt::PJRTCompileOptions> compile_options);
RcppExport SEXP _pjrt_impl_client_program_compile(SEXP clientSEXP, SEXP programSEXP, SEXP compile_optionsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTClient> >::type client(clientSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTProgram> >::type program(programSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTCompileOptions> >::type compile_options(compile_optionsSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_client_program_compile(client, program, compile_options));
    return rcpp_result_gen;
END_RCPP
}
// impl_client_buffer_from_double
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_double(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims, std::string type);
RcppExport SEXP _pjrt_impl_client_buffer_from_double(SEXP clientSEXP, SEXP dataSEXP, SEXP dimsSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTClient> >::type client(clientSEXP);
    Rcpp::traits::input_parameter< SEXP >::type data(dataSEXP);
    Rcpp::traits::input_parameter< std::vector<int64_t> >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_client_buffer_from_double(client, data, dims, type));
    return rcpp_result_gen;
END_RCPP
}
// impl_client_buffer_from_integer
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_integer(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims, std::string type);
RcppExport SEXP _pjrt_impl_client_buffer_from_integer(SEXP clientSEXP, SEXP dataSEXP, SEXP dimsSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTClient> >::type client(clientSEXP);
    Rcpp::traits::input_parameter< SEXP >::type data(dataSEXP);
    Rcpp::traits::input_parameter< std::vector<int64_t> >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_client_buffer_from_integer(client, data, dims, type));
    return rcpp_result_gen;
END_RCPP
}
// impl_client_buffer_from_logical
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_client_buffer_from_logical(Rcpp::XPtr<rpjrt::PJRTClient> client, SEXP data, std::vector<int64_t> dims, std::string type);
RcppExport SEXP _pjrt_impl_client_buffer_from_logical(SEXP clientSEXP, SEXP dataSEXP, SEXP dimsSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTClient> >::type client(clientSEXP);
    Rcpp::traits::input_parameter< SEXP >::type data(dataSEXP);
    Rcpp::traits::input_parameter< std::vector<int64_t> >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_client_buffer_from_logical(client, data, dims, type));
    return rcpp_result_gen;
END_RCPP
}
// impl_client_buffer_to_host
SEXP impl_client_buffer_to_host(Rcpp::XPtr<rpjrt::PJRTClient> client, Rcpp::XPtr<rpjrt::PJRTBuffer> buffer);
RcppExport SEXP _pjrt_impl_client_buffer_to_host(SEXP clientSEXP, SEXP bufferSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTClient> >::type client(clientSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTBuffer> >::type buffer(bufferSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_client_buffer_to_host(client, buffer));
    return rcpp_result_gen;
END_RCPP
}
// impl_client_platform_name
std::string impl_client_platform_name(Rcpp::XPtr<rpjrt::PJRTClient> client);
RcppExport SEXP _pjrt_impl_client_platform_name(SEXP clientSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTClient> >::type client(clientSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_client_platform_name(client));
    return rcpp_result_gen;
END_RCPP
}
// impl_loaded_executable_execute
Rcpp::XPtr<rpjrt::PJRTBuffer> impl_loaded_executable_execute(Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> executable, Rcpp::List input, Rcpp::XPtr<rpjrt::PJRTExecuteOptions> execution_options);
RcppExport SEXP _pjrt_impl_loaded_executable_execute(SEXP executableSEXP, SEXP inputSEXP, SEXP execution_optionsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTLoadedExecutable> >::type executable(executableSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type input(inputSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTExecuteOptions> >::type execution_options(execution_optionsSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_loaded_executable_execute(executable, input, execution_options));
    return rcpp_result_gen;
END_RCPP
}
// impl_buffer_element_type
Rcpp::XPtr<rpjrt::PJRTElementType> impl_buffer_element_type(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer);
RcppExport SEXP _pjrt_impl_buffer_element_type(SEXP bufferSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTBuffer> >::type buffer(bufferSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_buffer_element_type(buffer));
    return rcpp_result_gen;
END_RCPP
}
// impl_buffer_memory
Rcpp::XPtr<rpjrt::PJRTMemory> impl_buffer_memory(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer);
RcppExport SEXP _pjrt_impl_buffer_memory(SEXP bufferSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTBuffer> >::type buffer(bufferSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_buffer_memory(buffer));
    return rcpp_result_gen;
END_RCPP
}
// impl_memory_debug_string
std::string impl_memory_debug_string(Rcpp::XPtr<rpjrt::PJRTMemory> memory);
RcppExport SEXP _pjrt_impl_memory_debug_string(SEXP memorySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTMemory> >::type memory(memorySEXP);
    rcpp_result_gen = Rcpp::wrap(impl_memory_debug_string(memory));
    return rcpp_result_gen;
END_RCPP
}
// impl_memory_id
int impl_memory_id(Rcpp::XPtr<rpjrt::PJRTMemory> memory);
RcppExport SEXP _pjrt_impl_memory_id(SEXP memorySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTMemory> >::type memory(memorySEXP);
    rcpp_result_gen = Rcpp::wrap(impl_memory_id(memory));
    return rcpp_result_gen;
END_RCPP
}
// impl_memory_kind
std::string impl_memory_kind(Rcpp::XPtr<rpjrt::PJRTMemory> memory);
RcppExport SEXP _pjrt_impl_memory_kind(SEXP memorySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTMemory> >::type memory(memorySEXP);
    rcpp_result_gen = Rcpp::wrap(impl_memory_kind(memory));
    return rcpp_result_gen;
END_RCPP
}
// impl_memory_to_string
std::string impl_memory_to_string(Rcpp::XPtr<rpjrt::PJRTMemory> memory);
RcppExport SEXP _pjrt_impl_memory_to_string(SEXP memorySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTMemory> >::type memory(memorySEXP);
    rcpp_result_gen = Rcpp::wrap(impl_memory_to_string(memory));
    return rcpp_result_gen;
END_RCPP
}
// impl_element_type_as_string
std::string impl_element_type_as_string(Rcpp::XPtr<rpjrt::PJRTElementType> element_type);
RcppExport SEXP _pjrt_impl_element_type_as_string(SEXP element_typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTElementType> >::type element_type(element_typeSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_element_type_as_string(element_type));
    return rcpp_result_gen;
END_RCPP
}
// impl_buffer_dimensions
std::vector<int64_t> impl_buffer_dimensions(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer);
RcppExport SEXP _pjrt_impl_buffer_dimensions(SEXP bufferSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTBuffer> >::type buffer(bufferSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_buffer_dimensions(buffer));
    return rcpp_result_gen;
END_RCPP
}
// impl_execution_options_create
Rcpp::XPtr<rpjrt::PJRTExecuteOptions> impl_execution_options_create(std::vector<int64_t> non_donatable_input_indices, int launch_id);
RcppExport SEXP _pjrt_impl_execution_options_create(SEXP non_donatable_input_indicesSEXP, SEXP launch_idSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<int64_t> >::type non_donatable_input_indices(non_donatable_input_indicesSEXP);
    Rcpp::traits::input_parameter< int >::type launch_id(launch_idSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_execution_options_create(non_donatable_input_indices, launch_id));
    return rcpp_result_gen;
END_RCPP
}
// impl_plugin_pjrt_api_version
Rcpp::IntegerVector impl_plugin_pjrt_api_version(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin);
RcppExport SEXP _pjrt_impl_plugin_pjrt_api_version(SEXP pluginSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTPlugin> >::type plugin(pluginSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_plugin_pjrt_api_version(plugin));
    return rcpp_result_gen;
END_RCPP
}
// impl_plugin_attributes
Rcpp::List impl_plugin_attributes(Rcpp::XPtr<rpjrt::PJRTPlugin> plugin);
RcppExport SEXP _pjrt_impl_plugin_attributes(SEXP pluginSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<rpjrt::PJRTPlugin> >::type plugin(pluginSEXP);
    rcpp_result_gen = Rcpp::wrap(impl_plugin_attributes(plugin));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_pjrt_impl_plugin_load", (DL_FUNC) &_pjrt_impl_plugin_load, 1},
    {"_pjrt_impl_plugin_client_create", (DL_FUNC) &_pjrt_impl_plugin_client_create, 1},
    {"_pjrt_impl_program_load", (DL_FUNC) &_pjrt_impl_program_load, 2},
    {"_pjrt_impl_program_repr", (DL_FUNC) &_pjrt_impl_program_repr, 2},
    {"_pjrt_impl_build_options_create", (DL_FUNC) &_pjrt_impl_build_options_create, 3},
    {"_pjrt_impl_compile_options_create", (DL_FUNC) &_pjrt_impl_compile_options_create, 1},
    {"_pjrt_impl_client_program_compile", (DL_FUNC) &_pjrt_impl_client_program_compile, 3},
    {"_pjrt_impl_client_buffer_from_double", (DL_FUNC) &_pjrt_impl_client_buffer_from_double, 4},
    {"_pjrt_impl_client_buffer_from_integer", (DL_FUNC) &_pjrt_impl_client_buffer_from_integer, 4},
    {"_pjrt_impl_client_buffer_from_logical", (DL_FUNC) &_pjrt_impl_client_buffer_from_logical, 4},
    {"_pjrt_impl_client_buffer_to_host", (DL_FUNC) &_pjrt_impl_client_buffer_to_host, 2},
    {"_pjrt_impl_client_platform_name", (DL_FUNC) &_pjrt_impl_client_platform_name, 1},
    {"_pjrt_impl_loaded_executable_execute", (DL_FUNC) &_pjrt_impl_loaded_executable_execute, 3},
    {"_pjrt_impl_buffer_element_type", (DL_FUNC) &_pjrt_impl_buffer_element_type, 1},
    {"_pjrt_impl_buffer_memory", (DL_FUNC) &_pjrt_impl_buffer_memory, 1},
    {"_pjrt_impl_memory_debug_string", (DL_FUNC) &_pjrt_impl_memory_debug_string, 1},
    {"_pjrt_impl_memory_id", (DL_FUNC) &_pjrt_impl_memory_id, 1},
    {"_pjrt_impl_memory_kind", (DL_FUNC) &_pjrt_impl_memory_kind, 1},
    {"_pjrt_impl_memory_to_string", (DL_FUNC) &_pjrt_impl_memory_to_string, 1},
    {"_pjrt_impl_element_type_as_string", (DL_FUNC) &_pjrt_impl_element_type_as_string, 1},
    {"_pjrt_impl_buffer_dimensions", (DL_FUNC) &_pjrt_impl_buffer_dimensions, 1},
    {"_pjrt_impl_execution_options_create", (DL_FUNC) &_pjrt_impl_execution_options_create, 2},
    {"_pjrt_impl_plugin_pjrt_api_version", (DL_FUNC) &_pjrt_impl_plugin_pjrt_api_version, 1},
    {"_pjrt_impl_plugin_attributes", (DL_FUNC) &_pjrt_impl_plugin_attributes, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_pjrt(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

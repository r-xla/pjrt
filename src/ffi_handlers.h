// pjrt's built-in XLA FFI handlers, each defined in the file of its op.
// `builtin_custom_calls()` in ffi.cpp lists them under their custom-call
// target names, and `.onLoad()` registers that list with
// `pjrt_register_custom_call()`.
#pragma once

#include "xla/ffi/api/c_api.h"

namespace rpjrt {

XLA_FFI_Handler *get_geqrf_handler();
XLA_FFI_Handler *get_geqrf_handler_cuda();
XLA_FFI_Handler *get_orgqr_handler();
XLA_FFI_Handler *get_orgqr_handler_cuda();
XLA_FFI_Handler *get_lu_handler();
XLA_FFI_Handler *get_lu_handler_cuda();
XLA_FFI_Handler *get_svd_handler();
XLA_FFI_Handler *get_svd_handler_cuda();
XLA_FFI_Handler *get_eigh_handler();
XLA_FFI_Handler *get_eigh_handler_cuda();
XLA_FFI_Handler *get_potrf_handler();
XLA_FFI_Handler *get_potrf_handler_cuda();
XLA_FFI_Handler *get_trsm_handler();

}  // namespace rpjrt

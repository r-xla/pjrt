// CUDA counterpart of src/lu_pivots.cpp, backed by the precompiled kernel in
// src/cuda/lu_pivots_to_permutation.cu.
//
// Unlike the cuSOLVER-backed kernels next door, this one runs device code pjrt
// compiled itself; cuda_kernels.h explains how that code gets into the package
// and out to the driver.
#include <Rcpp.h>

#include "ffi_common.h"

#include "cuda_kernels.h"

#ifndef _WIN32
#include <algorithm>
#include <cstdint>
#endif

using namespace xla::ffi;

namespace rpjrt {

#ifndef _WIN32
static Error lu_pivots_to_permutation_cuda_impl(
    void *stream, BufferR1<DataType::S32> pivots,
    Result<BufferR1<DataType::S32>> perm) {
  const auto k64 = static_cast<std::int64_t>(pivots.element_count());
  const auto n64 = static_cast<std::int64_t>(perm->element_count());
  int k, n;
  PJRT_RETURN_IF_ERROR(dim_to_int(k64, "pivots", k));
  PJRT_RETURN_IF_ERROR(dim_to_int(n64, "permutation", n));
  if (k > n) {
    return Error::InvalidArgument(
        "lu_pivots_to_permutation: got more pivots than permutation entries");
  }

  void *kernel = nullptr;
  PJRT_RETURN_IF_ERROR(cuda_kernel("pjrt_lu_pivots_to_permutation", &kernel));

  // Device pointers -- only the kernel may dereference these.
  void *pivots_ptr = pivots.untyped_data();
  void *perm_ptr = perm->untyped_data();
  void *args[] = {&pivots_ptr, &perm_ptr, &k, &n};

  // One block, as the kernel's __syncthreads() requires. The threads only
  // help with the identity fill, so there is nothing to gain past `n`.
  const unsigned int block = static_cast<unsigned int>(
      std::min<std::int64_t>(std::max<std::int64_t>(n, 1), 256));
  return cuda_launch(kernel, /*grid_dim=*/1, block, stream, args);
}
#endif  // _WIN32

static Error do_lu_pivots_to_permutation_cuda(
    void *stream, BufferR1<DataType::S32> pivots,
    Result<BufferR1<DataType::S32>> perm) {
#ifdef _WIN32
  return Error(ErrorCode::kUnimplemented,
               "CUDA lu_pivots_to_permutation is not supported on Windows");
#else
  return lu_pivots_to_permutation_cuda_impl(stream, pivots, perm);
#endif
}

XLA_FFI_DEFINE_HANDLER(lu_pivots_to_permutation_handler_cuda,
                       do_lu_pivots_to_permutation_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Arg<BufferR1<DataType::S32>>()   // pivots
                           .Ret<BufferR1<DataType::S32>>());  // permutation

}  // namespace rpjrt

// [[Rcpp::export]]
SEXP get_lu_pivots_to_permutation_handler_cuda() {
  return R_MakeExternalPtr((void *)rpjrt::lu_pivots_to_permutation_handler_cuda,
                           R_NilValue, R_NilValue);
}

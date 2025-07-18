#include "utils.h"

#include <memory>

void check_err(const PJRT_Api *api, PJRT_Error *err) {
  if (err) {
    PJRT_Error_Message_Args args;
    args.error = err;
    args.struct_size = sizeof(PJRT_Error_Message_Args);
    api->PJRT_Error_Message_(&args);
    throw std::runtime_error(args.message);
  }
}

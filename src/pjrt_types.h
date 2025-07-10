#pragma once

#include "buffer.h"
#include "client.h"
#include "pjrt.h"
#include "plugin.h"
#include "program.h"

namespace rpjrt {

class PJRTElementType {
public:
  explicit PJRTElementType(PJRT_Buffer_Type type) : element_type_(type) {}

  PJRT_Buffer_Type get_type() const { return element_type_; }

  int as_integer() const { return static_cast<int>(element_type_); }

  std::string as_string() const {
    switch (element_type_) {
    case PJRT_Buffer_Type_INVALID:
      return "INVALID";
    case PJRT_Buffer_Type_PRED:
      return "PRED";
    case PJRT_Buffer_Type_S8:
      return "S8";
    case PJRT_Buffer_Type_S16:
      return "S16";
    case PJRT_Buffer_Type_S32:
      return "S32";
    case PJRT_Buffer_Type_S64:
      return "S64";
    case PJRT_Buffer_Type_U8:
      return "U8";
    case PJRT_Buffer_Type_U16:
      return "U16";
    case PJRT_Buffer_Type_U32:
      return "U32";
    case PJRT_Buffer_Type_U64:
      return "U64";
    case PJRT_Buffer_Type_F16:
      return "F16";
    case PJRT_Buffer_Type_F32:
      return "F32";
    case PJRT_Buffer_Type_F64:
      return "F64";
    case PJRT_Buffer_Type_BF16:
      return "BF16";
    case PJRT_Buffer_Type_C64:
      return "C64";
    case PJRT_Buffer_Type_C128:
      return "C128";
    case PJRT_Buffer_Type_F8E5M2:
      return "F8E5M2";
    case PJRT_Buffer_Type_F8E4M3FN:
      return "F8E4M3FN";
    case PJRT_Buffer_Type_F8E4M3B11FNUZ:
      return "F8E4M3B11FNUZ";
    case PJRT_Buffer_Type_F8E5M2FNUZ:
      return "F8E5M2FNUZ";
    case PJRT_Buffer_Type_F8E4M3FNUZ:
      return "F8E4M3FNUZ";
    case PJRT_Buffer_Type_S4:
      return "S4";
    case PJRT_Buffer_Type_U4:
      return "U4";
    case PJRT_Buffer_Type_TOKEN:
      return "TOKEN";
    case PJRT_Buffer_Type_S2:
      return "S2";
    case PJRT_Buffer_Type_U2:
      return "U2";
    case PJRT_Buffer_Type_F8E4M3:
      return "F8E4M3";
    case PJRT_Buffer_Type_F8E3M4:
      return "F8E3M4";
    case PJRT_Buffer_Type_F8E8M0FNU:
      return "F8E8M0FNU";
    case PJRT_Buffer_Type_F4E2M1FN:
      return "F4E2M1FN";
    default:
      return "UNKNOWN(" + std::to_string(as_integer()) + ")";
    }
  }

private:
  PJRT_Buffer_Type element_type_;
};

} // namespace rpjrt

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
        return "pred";
      case PJRT_Buffer_Type_S8:
        return "i8";
      case PJRT_Buffer_Type_S16:
        return "i16";
      case PJRT_Buffer_Type_S32:
        return "i32";
      case PJRT_Buffer_Type_S64:
        return "i64";
      case PJRT_Buffer_Type_U8:
        return "ui8";
      case PJRT_Buffer_Type_U16:
        return "ui16";
      case PJRT_Buffer_Type_U32:
        return "ui32";
      case PJRT_Buffer_Type_U64:
        return "ui64";
      case PJRT_Buffer_Type_F32:
        return "f32";
      case PJRT_Buffer_Type_F64:
        return "f64";
      default:
        Rcpp::stop("Unknown element type: %d", as_integer());
    }
  }

 private:
  PJRT_Buffer_Type element_type_;
};

}  // namespace rpjrt

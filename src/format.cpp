#include <Rcpp.h>

#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

#include "bfloat16.h"

using namespace Rcpp;

// The stablehlo hex spellings of NaN and infinity, plus the number of decimal
// digits that round-trips the format, for one floating-point dtype.
//
// Keyed by dtype rather than by bit width: bf16 and f16 are both 16 bits, so
// width no longer identifies a floating-point format.
struct FloatFormat {
  const char *nan;
  const char *pos_inf;
  const char *neg_inf;
  int digits;  // significant decimal digits needed to round-trip
};

// digits = ceil(mantissa_bits * log10(2)) + 1: 4 for bf16 (8 bits), 9 for f32
// (24), 17 for f64 (53). std::setprecision under std::scientific prints one
// digit before the point and `n` after, so it is passed digits - 1.
constexpr FloatFormat kBF16Format = {"0x7FC0", "0x7F80", "0xFF80", 4};
constexpr FloatFormat kF32Format = {"0x7FC00000", "0x7F800000", "0xFF800000",
                                    9};
constexpr FloatFormat kF64Format = {"0x7FF8000000000000", "0x7FF0000000000000",
                                    "0xFFF0000000000000", 17};

std::string format_float_value(double value, const FloatFormat &fmt) {
  if (R_IsNaN(value)) {
    return fmt.nan;
  } else if (!R_finite(value)) {
    return (value > 0) ? fmt.pos_inf : fmt.neg_inf;
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(fmt.digits - 1);
  oss << value;
  return oss.str();
}

std::string format_element(const unsigned char *ptr, std::string dtype) {
  std::ostringstream oss;
  if (dtype == "bf16") {
    uint16_t bits;
    std::memcpy(&bits, ptr, 2);
    return format_float_value(
        static_cast<double>(rpjrt::bfloat16::from_bits(bits).to_float()),
        kBF16Format);
  } else if (dtype == "f32") {
    float val;
    std::memcpy(&val, ptr, 4);
    return format_float_value((double)val, kF32Format);
  } else if (dtype == "f64") {
    double val;
    std::memcpy(&val, ptr, 8);
    return format_float_value(val, kF64Format);
  } else if (dtype == "i64") {
    int64_t val;
    std::memcpy(&val, ptr, 8);
    oss << val;
  } else if (dtype == "ui64") {
    uint64_t val;
    std::memcpy(&val, ptr, 8);
    oss << val;
  } else if (dtype == "i32") {
    int32_t val;
    std::memcpy(&val, ptr, 4);
    oss << val;
  } else if (dtype == "ui32") {
    uint32_t val;
    std::memcpy(&val, ptr, 4);
    oss << val;
  } else if (dtype == "i16") {
    int16_t val;
    std::memcpy(&val, ptr, 2);
    oss << val;
  } else if (dtype == "ui16") {
    uint16_t val;
    std::memcpy(&val, ptr, 2);
    oss << val;
  } else if (dtype == "i8") {
    // default formatting uses char, so need to convert to int
    oss << (int)(*reinterpret_cast<const int8_t *>(ptr));
  } else if (dtype == "ui8") {
    // default formatting uses char, so need to convert to unsigned int
    oss << (unsigned int)(*ptr);
  } else if (dtype == "pred" || dtype == "i1") {
    return (*ptr) ? "true" : "false";
  } else {
    stop("Unsupported dtype: " + dtype);
  }
  return oss.str();
}

int get_element_size(std::string dtype) {
  if (dtype == "f64" || dtype == "i64" || dtype == "ui64") return 8;
  if (dtype == "f32" || dtype == "i32" || dtype == "ui32") return 4;
  if (dtype == "bf16" || dtype == "i16" || dtype == "ui16") return 2;
  return 1;
}

// [[Rcpp::export]]
CharacterVector format_raw_buffer_cpp(RawVector data, std::string dtype,
                                      IntegerVector shape) {
  int element_size = get_element_size(dtype);

  int rank = shape.length();

  int expected_length = 1;
  for (int i = 0; i < rank; ++i) {
    expected_length *= shape[i];
  }

  if (expected_length == 0) {
    return CharacterVector(0);
  }

  int num_elements = expected_length;
  expected_length *= element_size;
  if (data.length() != expected_length) {
    stop("Data size mismatch");
  }

  CharacterVector result(num_elements);

  for (int i = 0; i < num_elements; ++i) {
    result[i] = format_element(data.begin() + i * element_size, dtype);
  }

  // Add dimensions attribute if rank > 0
  if (rank > 0) {
    result.attr("dim") = shape;
  }

  return result;
}

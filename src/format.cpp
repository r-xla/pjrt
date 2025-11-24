#include <Rcpp.h>

#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace Rcpp;

std::string format_float_value(double value, int precision) {
  // stablehlo format for NaN and infinity
  if (R_IsNaN(value)) {
    return "0x7FC00000";
  } else if (!R_finite(value)) {
    return (value > 0) ? "0x7F800000" : "0xFF800000";
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(precision == 32 ? 8 : 16);
  oss << value;
  return oss.str();
}

std::string format_element(const unsigned char *ptr, std::string dtype) {
  std::ostringstream oss;
  if (dtype == "f32") {
    float val;
    std::memcpy(&val, ptr, 4);
    return format_float_value((double)val, 32);
  } else if (dtype == "f64") {
    double val;
    std::memcpy(&val, ptr, 8);
    return format_float_value(val, 64);
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
  if (dtype == "i16" || dtype == "ui16") return 2;
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

  return result;
}

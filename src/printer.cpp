#include "printer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <span>
#include <sstream>
#include <type_traits>
#include <vector>

#include "buffer.h"
#include "utils.h"

// For floats, formatting is defined globally per slice
// For integers, each value determines whether to use scientific notation
enum class FloatPrintMode { Fixed, Scientific, Scaled };

// Decide a global float formatting mode for a slice:
// - Fixed: standard fixed-point formatting
// - Scientific: scientific notation for all values in the slice
// - Scaled: emit a single scale prefix (1e±NN *) and print scaled fixed-point
// The returned pair is (mode, scale_exponent). The scale exponent is only
// meaningful when mode == FloatPrintMode::Scaled.
// returns the print mode and the scaling factor
// the latter matters only for FloatPrintMode::Scaled
template <typename T>
  requires std::is_floating_point_v<T>
static std::pair<FloatPrintMode, int> choose_float_print_mode(
    const std::vector<T> &values) {

  // Find smallest and largest absolute magnitudes across finite, non-zero
  // values
  double min_abs = std::numeric_limits<double>::infinity();
  double max_abs = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    double av = std::abs(static_cast<double>(values[i]));
    if (std::isfinite(av) && av > 0.0) {
      if (av < min_abs) min_abs = av;
      if (av > max_abs) max_abs = av;
    }
  }
  // If there are no non-zero finite values, default to fixed formatting
  if (!(max_abs > 0.0 && std::isfinite(min_abs))) {
    return {FloatPrintMode::Fixed, 1};
  }

  // Values are in a reasonable range for fixed formatting
  if (min_abs > 1e-4 && max_abs < 1e6) return {FloatPrintMode::Fixed, 1};

  // nicer printer when there is only one value
  if (values.size() == 1) {
    return {FloatPrintMode::Scientific, 1};
  }

  // Now we check whether all values have the same exponent,
  // if so, we use it for scaling

  // Helper to get an integer exponent close to log10(x)
  auto nearest_exp = [](double x) {
    if (x == 0.0 || !std::isfinite(x)) return 0;
    std::ostringstream oss;
    oss.setf(std::ios::scientific, std::ios::floatfield);
    oss.precision(0);
    oss << x;
    std::string s = oss.str();  // e.g., "1e+03"
    auto p = s.find('e');
    if (p == std::string::npos) return 0;
    try {
      return std::stoi(s.substr(p + 1));
    } catch (...) {
      return 0;
    }
  };
  int min_e = nearest_exp(min_abs);
  int max_e = nearest_exp(max_abs);

  // If all magnitudes share the same exponent, prefer scaled printing
  if (min_e == max_e) return {FloatPrintMode::Scaled, max_e};
  // Otherwise prefer uniform scientific notation for the slice
  return {FloatPrintMode::Scientific, 1};
}

// Move device buffer to host into a typed vector
template <typename CopyT>
static std::vector<CopyT> buffer_to_host_copy(rpjrt::PJRTBuffer *buf,
                                              int64_t numel) {
  std::vector<CopyT> temp_vec(static_cast<size_t>(numel));
  std::span<uint8_t> host_buffer(reinterpret_cast<uint8_t *>(temp_vec.data()),
                                 static_cast<size_t>(numel * sizeof(CopyT)));
  buf->buffer_to_host(host_buffer);
  return temp_vec;
}

// Build aligned lines for a subset of columns [c0, c1] and first rows_to_print
// rows values: the slice of the data to be printed rows: the number of rows in
// the array cols: the number of columns in the array c0: the starting column
// index c1: the ending column index rows_to_print: the (maximum) number of rows
// to print lines: the output lines. This is modified in-place format_value: a
// function to format the values
template <typename T, typename Formatter>
static std::pair<int64_t, size_t> col_info_for_printing(
    std::span<const T> values, int64_t cols, int64_t c_start,
    int64_t rows_to_print, int max_width, Formatter format_value) {
  int64_t c_end = c_start - 1;
  size_t uniform_width_running = 0;
  size_t uniform_width_best = 0;
  for (int64_t cand_end = c_start; cand_end < cols; ++cand_end) {
    // Compute max token width for this new column
    size_t column_width = 0;
    for (int64_t r = 0; r < rows_to_print; ++r) {
      int64_t base = r * cols;
      std::string tok =
          format_value(values[static_cast<size_t>(base + cand_end)]);
      if (tok.size() > column_width) column_width = tok.size();
    }
    // Update running uniform width across the current window
    if (column_width > uniform_width_running)
      uniform_width_running = column_width;

    size_t num_cols = static_cast<size_t>(cand_end - c_start + 1);
    size_t required =
        num_cols * uniform_width_running + (num_cols > 1 ? num_cols - 1 : 0);

    // If no width restriction, or still fits, accept and continue
    if (max_width <= 0 || required <= static_cast<size_t>(max_width)) {
      c_end = cand_end;
      uniform_width_best = uniform_width_running;
      continue;
    }
    // Exceeded width; stop expanding
    break;
  }

  // If nothing fit within max_width, fall back to single column at c_start
  if (c_end < c_start) {
    // this should never happen, but just to be sure
    c_end = c_start;
  }

  return {c_end, uniform_width_best};
}

template <typename T, typename Formatter>
static std::pair<int64_t, int64_t> build_buffer_lines_subset(std::span<const T> values,
                                         int64_t cols,
                                         int64_t c_start, int64_t rows_to_print,
                                         int max_width,
                                         std::vector<std::string> &lines,
                                         Formatter format_value) {
  auto [c_end, uniform_width] = col_info_for_printing<T, Formatter>(
      values, cols, c_start, rows_to_print, max_width, format_value);

  // Emit subset header if we are not showing all columns
  bool subset = !(c_start == 0 && c_end == cols - 1);
  if (subset) {
    std::ostringstream hdr2;
    hdr2 << "Columns " << (c_start + 1) << " to " << (c_end + 1);
    lines.push_back(hdr2.str());
  }

  // uniform_width already computed together with c_end

  int64_t r = 0;
  for (r = 0; r < rows_to_print; ++r) {
    std::ostringstream line;
    // Prefix one leading space for each printed row line
    line << ' ';
    int64_t base = r * cols;
    for (int64_t c = c_start; c <= c_end; ++c) {
      std::string tok = format_value(values[static_cast<size_t>(base + c)]);
      if (c > c_start) line << ' ';
      for (size_t pad = tok.size(); pad < uniform_width; ++pad) line << ' ';
      line << tok;
    }
    lines.push_back(line.str());
  }

  return {r, c_end};
}

// Create a contiguous span over the last two dimensions for a given leading
// index
template <typename T>
static std::span<const T> make_last2_contiguous_span(
    const std::vector<T> &flat, const std::vector<int64_t> &pseudo_dims,
    const std::vector<int64_t> &lead_index) {
  const int nprint = static_cast<int>(pseudo_dims.size());
  std::vector<int64_t> stride(nprint, 1);
  for (int i = nprint - 2; i >= 0; --i)
    stride[i] = stride[i + 1] * pseudo_dims[i + 1];
  int64_t base = 0;
  for (size_t k = 0; k < lead_index.size(); ++k)
    base += lead_index[k] * stride[k];
  const int64_t rows = pseudo_dims[nprint - 2];
  const int64_t cols = pseudo_dims[nprint - 1];
  return std::span<const T>(flat.data() + static_cast<size_t>(base),
                            static_cast<size_t>(rows * cols));
}

// Core printer for a typed host vector using a value formatter and an optional
// scale-prefix emitter. Prints last two dims as matrix, chunks columns to fit
// max_width, and limits rows to max_rows_slice.
template <typename CopyT, typename Formatter, typename PrefixFn>
static void print_with_formatter_fn(
    const std::vector<int64_t> &dimensions, int max_width, int max_rows_slice,
    int &rows_left, std::vector<std::string> &cont, bool &truncated,
    const std::vector<CopyT> &temp_vec, Formatter formatter,
    PrefixFn maybe_insert_scale_prefix = []() {}) {
  const int ndim = dimensions.size();

  // pseudo_dims are used so we don't have to treat the 0d and 1d cases special
  std::vector<int64_t> pseudo_dims;
  // for the 0d case we treat it as 1x1 matrix
  if (ndim == 0) pseudo_dims = {1, 1};
  // for the 1d case we treat it as nx1 matrix
  else if (ndim == 1)
    pseudo_dims = {dimensions[0], 1};
  else
    pseudo_dims = dimensions;

  const int nprint = pseudo_dims.size();
  const int64_t rows = pseudo_dims[nprint - 2];
  const int64_t cols = pseudo_dims[nprint - 1];

  std::vector<int64_t> lead_dims;
  if (nprint > 2) lead_dims.assign(pseudo_dims.begin(), pseudo_dims.end() - 2);
  int64_t lead_count = number_of_elements(lead_dims);

  std::vector<int64_t> lead_strides =  dims2strides(lead_dims, true);

  std::vector<int64_t> lead_index = {};
  int64_t lid = 0;
  // iterate over leading dimensions to print slices


  for (int64_t lid = 0; lid < std::max<int64_t>(lead_count, 1); ++lid) {
    lead_index = id2indices(lid, lead_strides);
    // information on slice
    if (!lead_index.empty()) {
      std::ostringstream hdr;
      hdr << "(";
      for (size_t k = 0; k < lead_index.size(); ++k) {
        hdr << (lead_index[k] + 1);
        if (k + 1 < lead_index.size()) hdr << ",";
      }
      hdr << ",.,.) =";
      cont.push_back(hdr.str());
    }

    // one global scale prefix
    if (lid == 0) maybe_insert_scale_prefix();

    // Extract this slice as a contiguous span (because of row-major ordering, this data is contigous)
    std::span<const CopyT> slice =
        make_last2_contiguous_span<CopyT>(temp_vec, pseudo_dims, lead_index);

    // now do the actual data printing
    int64_t rows_to_print = rows;
    if (max_rows_slice > 0)
      rows_to_print = std::min<int64_t>(rows_to_print, max_rows_slice);
    if (rows_left >= 0)
      rows_to_print = std::min<int64_t>(rows_to_print, rows_left);

    int64_t c_start = 0;
    // iterate over the rows and print line by line.
    while (c_start < cols) {
      auto [r_end, c_end] = build_buffer_lines_subset<CopyT>(
          slice, cols, c_start, rows_to_print, max_width, cont,
          formatter);
      if (r_end != rows) {
        truncated = true;
      }
      c_start = c_end + 1;

      if (rows_left >= 0) {
        rows_left -= rows_to_print;
        if (rows_left <= 0) {
          if (c_start != cols) {
            truncated = true;
          }
          return;
        }
      }
    }
    if (c_start != cols) {
      truncated = true;
    }

    if (lid + 1 < std::max<int64_t>(lead_count, 1)) cont.push_back("");
  }
  // We definitely truncated if we didn't exhaust all the leading dims,
  // (but we might have also truncated individual slices)
  if (lid != (lead_count - 1)) {
    truncated = false;
  }
}

// [[Rcpp::export()]]
void impl_buffer_print(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, int max_rows,
                       int max_width, int max_rows_slice) {
  const auto dimensions = buffer->dimensions();
  const auto element_type = buffer->element_type();

  int64_t numel = dimensions.empty() ? 1 : number_of_elements(dimensions);

  std::vector<std::string> cont;
  bool truncated = false;
  int rows_left = (max_rows == -1 ? -1 : max_rows);

  // Generic printer that takes a value vector and a formatter
  // No-op; replaced by print_with_formatter_fn

  // Float handler
  auto handle_float = [buffer, numel, &cont, &truncated, &rows_left, dimensions, max_width,
                       max_rows_slice](auto fp_tag) {
    using FP = decltype(fp_tag);
    std::vector<FP> temp_vec = buffer_to_host_copy<FP>(buffer.get(), numel);
    auto [mode, scale_exp] = choose_float_print_mode<FP>(temp_vec);
    bool use_scientific = (mode == FloatPrintMode::Scientific);
    double denom = (mode == FloatPrintMode::Scaled)
                       ? std::pow(10.0, static_cast<double>(scale_exp))
                       : 1.0;
    bool inserted_scale_prefix = false;
    auto maybe_prefix = [&cont, mode, scale_exp, &inserted_scale_prefix]() {
      if (!inserted_scale_prefix && mode == FloatPrintMode::Scaled) {
        std::ostringstream p;
        int e = scale_exp;
        char sign = (e >= 0) ? '+' : '-';
        int abse = std::abs(e);
        std::ostringstream exp_ss;
        exp_ss << sign << std::setfill('0') << std::setw(2) << abse;
        p << "1e" << exp_ss.str() << " *";
        cont.push_back(p.str());
        inserted_scale_prefix = true;
      }
    };
    auto fmt = [use_scientific, denom](const FP &v) {
      std::ostringstream s;
      if (use_scientific)
        s.setf(std::ios::scientific, std::ios::floatfield);
      else
        s.setf(std::ios::fixed, std::ios::floatfield);
      s << std::setprecision(4)
        << (denom != 1.0 ? static_cast<double>(v) / denom
                         : static_cast<double>(v));
      return s.str();
    };
    print_with_formatter_fn<FP>(dimensions, max_width, max_rows_slice, rows_left, cont,
                                truncated, temp_vec, fmt, maybe_prefix);
  };

  // Integer handler (scientific if >6 digits)
  auto handle_integer = [buffer, numel, &cont, &truncated, &rows_left, dimensions,
                         max_width, max_rows_slice](auto int_tag) {
    using IT = decltype(int_tag);
    std::vector<IT> temp_vec = buffer_to_host_copy<IT>(buffer.get(), numel);
    auto noop = []() {};
    auto fmt = [](const IT &v) {
      long double abs_val;
      if constexpr (std::is_signed_v<IT>) {
        long long vv = static_cast<long long>(v);
        abs_val = static_cast<long double>(vv < 0 ? -vv : vv);
      } else {
        unsigned long long vv = static_cast<unsigned long long>(v);
        abs_val = static_cast<long double>(vv);
      }
      int digits = (abs_val == 0)
                       ? 1
                       : static_cast<int>(std::floor(std::log10(abs_val))) + 1;
      if (digits > 6) {
        std::ostringstream s;
        s.setf(std::ios::scientific, std::ios::floatfield);
        s << std::setprecision(4) << static_cast<long double>(v);
        return s.str();
      }
      if constexpr (std::is_signed_v<IT>)
        return std::to_string(static_cast<long long>(v));
      else
        return std::to_string(static_cast<unsigned long long>(v));
    };
    print_with_formatter_fn<IT>(dimensions, max_width, max_rows_slice, rows_left, cont,
                                truncated, temp_vec, fmt, noop);
  };

  // Logical handler (predicates)
  auto handle_logical = [buffer, numel, &cont, &truncated, &rows_left, dimensions,
                         max_width, max_rows_slice]() {
    using BT = uint8_t;
    std::vector<BT> temp_vec = buffer_to_host_copy<BT>(buffer.get(), numel);
    auto noop = []() {};
    auto fmt = [](const BT &v) { return std::string(v ? "true" : "false"); };
    print_with_formatter_fn<BT>(dimensions, max_width, max_rows_slice, rows_left, cont,
                                truncated, temp_vec, fmt, noop);
  };

  switch (element_type) {
    case PJRT_Buffer_Type_F32:
      handle_float(float{});
      break;
    case PJRT_Buffer_Type_F64:
      handle_float(double{});
      break;
    case PJRT_Buffer_Type_S8:
      handle_integer(int8_t{});
      break;
    case PJRT_Buffer_Type_S16:
      handle_integer(int16_t{});
      break;
    case PJRT_Buffer_Type_S32:
      handle_integer(int32_t{});
      break;
    case PJRT_Buffer_Type_S64:
      handle_integer(int64_t{});
      break;
    case PJRT_Buffer_Type_U8:
      handle_integer(uint8_t{});
      break;
    case PJRT_Buffer_Type_U16:
      handle_integer(uint16_t{});
      break;
    case PJRT_Buffer_Type_U32:
      handle_integer(uint32_t{});
      break;
    case PJRT_Buffer_Type_U64:
      handle_integer(uint64_t{});
      break;
    case PJRT_Buffer_Type_PRED:
      handle_logical();
      break;
    default:
      Rcpp::stop("Unsupported buffer element type for printing.");
  }

  if (truncated) {
    cont.push_back(" ... [output was truncated, set max_rows = -1 to see all]");
  }

  for (int i = 0; i < static_cast<int>(cont.size()); ++i) {
    Rcpp::Rcout << cont[i] << '\n';
  }
}

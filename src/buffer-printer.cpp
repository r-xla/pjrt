#include "buffer-printer.h"

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
enum class FloatPrintMode { Fixed, Scientific, Scaled };

// builds the buffer lines for a specific slice
template <typename T, typename F>
std::pair<bool, int64_t> build_buffer_lines(
    int64_t ncols, int64_t nrows, int64_t rows_to_print, int64_t max_width,
    std::vector<std::string> &cont, F fmt, std::span<T> slice,
    int64_t rows_left, std::string prefix) {
  bool truncated = false;
  int64_t c_start = 0;
  // iterate over the rows and print line by line.
  if (prefix.size() != 0) {
    cont.push_back(prefix);
  };
  while (c_start < ncols) {
    auto [r_end, c_end] = build_buffer_lines_subset<T>(
        slice, ncols, c_start, rows_to_print, max_width, cont, fmt);
    if (r_end != nrows) {
      truncated = true;
    }
    c_start = c_end + 1;

    if (rows_left >= 0) {
      rows_left -= rows_to_print;
      if (rows_left <= 0) {
        break;
      }
    }
  }
  if (c_start != ncols) {
    truncated = true;
  }

  return {truncated, rows_left};
}

// Returns: pair<print mode, scale exponent>
template <typename T>
  requires std::is_floating_point_v<T> || std::is_integral_v<T> ||
           std::is_same_v<bool, T>
static std::pair<FloatPrintMode, int> choose_float_print_mode(
    std::span<const T> &values) {
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
  if (!(max_abs > 0.0 && std::isfinite(min_abs))) {
    return {FloatPrintMode::Fixed, 1};
  }

  if (min_abs > 1e-4 && max_abs < 1e6) return {FloatPrintMode::Fixed, 1};

  if (values.size() == 1) {
    return {FloatPrintMode::Scientific, 1};
  }

  auto get_exp = [](double x) {
    if (!std::isfinite(x) || x == 0.0) return 0;
    const double ax = std::fabs(x);
    int e = static_cast<int>(std::floor(std::log10(ax)));
    return e;
  };

  int min_e = get_exp(min_abs);
  int max_e = get_exp(max_abs);

  if (min_e == max_e) return {FloatPrintMode::Scaled, max_e};

  return {FloatPrintMode::Scientific, 1};
}

template <typename CopyT>
static std::vector<CopyT> buffer_to_host_copy(rpjrt::PJRTBuffer *buf,
                                              int64_t numel) {
  std::vector<CopyT> temp_vec(static_cast<size_t>(numel));
  std::span<uint8_t> host_buffer(reinterpret_cast<uint8_t *>(temp_vec.data()),
                                 static_cast<size_t>(numel * sizeof(CopyT)));
  buf->buffer_to_host(host_buffer);
  return temp_vec;
}

// Returns a pair:
// * c_end: until which column to print from c_start
// * uniform_width: column width
template <typename T, typename Formatter>
static std::pair<int64_t, size_t> col_info_for_printing(
    std::span<const T> values, int64_t ncols, int64_t c_start,
    int64_t rows_to_print, int max_width, Formatter format_value) {
  int64_t c_end = c_start - 1;
  size_t uniform_width_running = 0;
  size_t uniform_width_best = 0;
  for (int64_t cand_end = c_start; cand_end < ncols; ++cand_end) {
    // Compute max token width for this new column
    size_t column_width = 0;
    for (int64_t r = 0; r < rows_to_print; ++r) {
      int64_t base = r * ncols;
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
  // this happens if the user e.g. sets max_width = 2
  if (c_end < c_start) {
    c_end = c_start;
  }

  return {c_end, uniform_width_best};
}

// Build aligned lines for a subset of columns [c0, c1] and first rows_to_print
// rows values: the slice of the data to be printed rows: the number of rows in
// the array cols: the number of columns in the array c0: the starting column
// index c1: the ending column index rows_to_print: the (maximum) number of rows
// to print lines: the output lines. This is modified in-place format_value: a
// function to format the values
template <typename T, typename F>
  requires std::is_floating_point_v<T> || std::is_integral_v<T> ||
           std::is_same_v<T, bool>
static std::pair<int64_t, int64_t> build_buffer_lines_subset(
    const std::span<T> values, int64_t ncols, int64_t c_start,
    int64_t rows_to_print, int max_width, std::vector<std::string> &lines,
    F fmt) {
  // it's possible that c_end = s_start and uniform_width = 0
  // in this case we print exactly one column
  auto [c_end, uniform_width] = col_info_for_printing<T, F>(
      values, ncols, c_start, rows_to_print, max_width, fmt);

  // Emit subset header if we are not showing all columns
  bool subset = !(c_start == 0 && c_end == ncols - 1);
  if (subset) {
    std::ostringstream hdr2;
    hdr2 << "Columns " << (c_start + 1) << " to " << (c_end + 1);
    lines.push_back(hdr2.str());
  }

  int64_t r = 0;
  for (r = 0; r < rows_to_print; ++r) {
    std::ostringstream line;
    // Prefix one leading space for each printed row line
    line << ' ';
    int64_t base = r * ncols;
    for (int64_t c = c_start; c <= c_end; ++c) {
      std::string tok = fmt(values[static_cast<size_t>(base + c)]);
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
  const int64_t nrows = pseudo_dims[nprint - 2];
  const int64_t ncols = pseudo_dims[nprint - 1];
  return std::span<const T>(flat.data() + static_cast<size_t>(base),
                            static_cast<size_t>(nrows * ncols));
}

// Core printer for a typed host vector using a value formatter and an optional
// scale-prefix emitter. Prints last two dims as matrix, chunks columns to fit
// max_width, and limits rows to max_rows_slice.
template <typename CopyT>
  requires std::is_floating_point_v<CopyT> || std::is_integral_v<CopyT> ||
           std::is_same_v<CopyT, bool>
static void print_with_formatter_fn(const std::vector<int64_t> &dimensions,
                                    int max_width, int max_rows_slice,
                                    int &rows_left,
                                    std::vector<std::string> &cont,
                                    const std::vector<CopyT> &temp_vec) {
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
  const int64_t nrows = pseudo_dims[nprint - 2];
  const int64_t ncols = pseudo_dims[nprint - 1];

  std::vector<int64_t> lead_dims;
  if (nprint > 2) lead_dims.assign(pseudo_dims.begin(), pseudo_dims.end() - 2);
  int64_t lead_count = number_of_elements(lead_dims);

  std::vector<int64_t> lead_strides = dims2strides(lead_dims, true);

  std::vector<int64_t> lead_index = {};
  int64_t lid = 0;

  bool truncated = false;

  // iterate over leading dimensions to print slices
  for (lid = 0; lid < std::max<int64_t>(lead_count, 1); ++lid) {
    std::pair<bool, int64_t> result;

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

    // Extract this slice as a span (because of row-major ordering,
    // this data is contigous)
    std::span<const CopyT> slice =
        make_last2_contiguous_span<CopyT>(temp_vec, pseudo_dims, lead_index);

    // now do the actual data printing
    int64_t rows_to_print = nrows;
    if (max_rows_slice > 0)
      rows_to_print = std::min<int64_t>(rows_to_print, max_rows_slice);
    if (rows_left >= 0)
      rows_to_print = std::min<int64_t>(rows_to_print, rows_left);

    if (std::is_floating_point_v<CopyT>) {
      auto mode_scale = choose_float_print_mode<CopyT>(slice);
      FloatPrintMode mode = mode_scale.first;
      int scale_exp = mode_scale.second;
      bool use_scientific = (mode == FloatPrintMode::Scientific);
      double denom = (mode == FloatPrintMode::Scaled)
                         ? std::pow(10.0, static_cast<double>(scale_exp))
                         : 1.0;

      std::string prefix;

      if (mode == FloatPrintMode::Scaled && denom != 1.0) {
        std::ostringstream p;
        int abse = std::abs(scale_exp);
        std::ostringstream exp_ss;
        exp_ss << '+' << std::setfill('0') << std::setw(2) << abse;
        p << "1e" << exp_ss.str() << " *";
        prefix = p.str();
      }

      auto fmt = [use_scientific, denom](const CopyT &v) {
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
      result = build_buffer_lines(ncols, nrows, rows_to_print, max_width, cont,
                                  fmt, slice, rows_left, prefix);
    } else if (std::is_same_v<bool, CopyT>) {  // bool
      auto fmt = [](const CopyT &v) {
        return std::string(v ? "true" : "false");
      };

      result = build_buffer_lines(ncols, nrows, rows_to_print, max_width, cont,
                                  fmt, slice, rows_left, "");
    } else {
      auto fmt = [](const CopyT &v) {
        long double abs_val;
        if constexpr (std::is_signed_v<CopyT>) {
          long long vv = static_cast<long long>(v);
          abs_val = static_cast<long double>(vv < 0 ? -vv : vv);
        } else {
          unsigned long long vv = static_cast<unsigned long long>(v);
          abs_val = static_cast<long double>(vv);
        }
        int digits =
            (abs_val == 0)
                ? 1
                : static_cast<int>(std::floor(std::log10(abs_val))) + 1;
        if (digits > 6) {
          std::ostringstream s;
          s.setf(std::ios::scientific, std::ios::floatfield);
          s << std::setprecision(4) << static_cast<long double>(v);
          return s.str();
        }
        if constexpr (std::is_signed_v<CopyT>)
          return std::to_string(static_cast<long long>(v));
        else
          return std::to_string(static_cast<unsigned long long>(v));
      };
      result = build_buffer_lines(ncols, nrows, rows_to_print, max_width, cont,
                                  fmt, slice, rows_left, "");
    };

    truncated = result.first;
    rows_left = result.second;

    if (lid + 1 < std::max<int64_t>(lead_count, 1)) cont.push_back("");

    if (rows_left < 0) {
      break;
    }
  }

  // We definitely truncated if we didn't exhaust all the leading dims,
  // (but we might have also truncated individual slices)
  if (truncated || (lid != lead_count)) {
    cont.push_back(" ... [output was truncated, set max_rows = -1 to see all]");
  }
}

// [[Rcpp::export()]]
void impl_buffer_print(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, int max_rows,
                       int max_width, int max_rows_slice) {
  const auto dimensions = buffer->dimensions();
  const auto element_type = buffer->element_type();

  // because every line starts with ' '
  max_width -= 1;

  int64_t numel = dimensions.empty() ? 1 : number_of_elements(dimensions);

  std::vector<std::string> cont;
  int rows_left = (max_rows == -1 ? -1 : max_rows);

  auto handle_float = [buffer, numel, &cont, &rows_left, dimensions, max_width,
                       max_rows_slice](auto fp_tag) {
    using FP = decltype(fp_tag);
    std::vector<FP> temp_vec = buffer_to_host_copy<FP>(buffer.get(), numel);
    print_with_formatter_fn<FP>(dimensions, max_width, max_rows_slice,
                                rows_left, cont, temp_vec);
  };

  auto handle_integer = [buffer, numel, &cont, &rows_left, dimensions,
                         max_width, max_rows_slice](auto int_tag) {
    using IT = decltype(int_tag);
    std::vector<IT> temp_vec = buffer_to_host_copy<IT>(buffer.get(), numel);
    print_with_formatter_fn<IT>(dimensions, max_width, max_rows_slice,
                                rows_left, cont, temp_vec);
  };

  auto handle_logical = [buffer, numel, &cont, &rows_left, dimensions,
                         max_width, max_rows_slice]() {
    using BT = uint8_t;
    std::vector<BT> temp_vec = buffer_to_host_copy<BT>(buffer.get(), numel);
    print_with_formatter_fn<BT>(dimensions, max_width, max_rows_slice,
                                rows_left, cont, temp_vec);
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

  for (int i = 0; i < static_cast<int>(cont.size()); ++i) {
    Rcpp::Rcout << cont[i] << '\n';
  }
}

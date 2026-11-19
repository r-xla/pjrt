#pragma once

#include <Rcpp.h>

#include <string>
#include <vector>

#include "buffer.h"

// Prints the buffer's data. `reversed` prints it as the R array of the
// reversed shape, whose row-major data are the buffer's column-major data.
void buffer_print(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, int n = 30,
                  int max_width = 85, int max_rows_slice = 30,
                  bool reversed = false);

std::vector<std::string> buffer_to_string_lines(
    const void* data, const std::vector<int64_t>& dimensions,
    PJRT_Buffer_Type element_type, int max_rows = 30, int max_width = 85,
    int max_rows_slice = 30);

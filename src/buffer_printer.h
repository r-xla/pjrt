#pragma once

#include <Rcpp.h>

#include "buffer.h"

void buffer_print(Rcpp::XPtr<rpjrt::PJRTBuffer> buffer, int n = 30,
                  int max_width = 85, int max_rows_slice = 30);

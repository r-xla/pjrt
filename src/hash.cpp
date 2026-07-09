// The R-value folds declared in hash.h. Their contract -- never split what
// identical() joins -- is documented there; this file mirrors the rules R's own
// value hashes use (src/main/unique.c: rhash, shash).

#include "hash.h"

#include <Rcpp.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

namespace rpjrt {

std::uint64_t hash_double(std::uint64_t h, double x) {
  std::uint64_t bits;
  if (ISNA(x)) {
    bits = 0x7ff00000000007a2ULL;  // NA_real_
  } else if (ISNAN(x)) {
    bits = 0x7ff8000000000000ULL;  // every other NaN compares equal
  } else if (x == 0.0) {
    bits = 0;  // +0.0 == -0.0
  } else {
    std::memcpy(&bits, &x, sizeof(bits));
  }
  // Both sentinels are NaN payloads, which no finite double can collide with.
  return hash_combine(h, bits);
}

std::uint64_t hash_atomic(std::uint64_t h, SEXP v) {
  const R_xlen_t n = Rf_xlength(v);
  switch (TYPEOF(v)) {
    case LGLSXP: {
      const int* p = LOGICAL(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_combine(h, static_cast<std::uint32_t>(p[i]));
      }
      break;
    }
    case INTSXP: {
      const int* p = INTEGER(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_combine(h, static_cast<std::uint32_t>(p[i]));
      }
      break;
    }
    case REALSXP: {
      const double* p = REAL(v);
      for (R_xlen_t i = 0; i < n; ++i) h = hash_double(h, p[i]);
      break;
    }
    case CPLXSXP: {
      const Rcomplex* p = COMPLEX(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_double(h, p[i].r);
        h = hash_double(h, p[i].i);
      }
      break;
    }
    case RAWSXP: {
      const Rbyte* p = RAW(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_combine(h, static_cast<std::uint64_t>(p[i]));
      }
      break;
    }
    case STRSXP: {
      for (R_xlen_t i = 0; i < n; ++i) {
        SEXP s = STRING_ELT(v, i);
        if (s == NA_STRING) {
          h = hash_combine(h, 0x4E41ULL);  // "NA"
          continue;
        }
        const char* c = CHAR(s);
        bool ascii = true;
        for (const char* q = c; *q; ++q) {
          if (static_cast<unsigned char>(*q) >= 0x80) {
            ascii = false;
            break;
          }
        }
        if (!ascii) {
          h = hash_combine(h, 0x8081ULL);  // non-ASCII: identical() decides
          continue;
        }
        h = hash_combine(h, std::hash<std::string>{}(std::string(c)));
      }
      break;
    }
    default:
      break;
  }
  return h;
}

}  // namespace rpjrt

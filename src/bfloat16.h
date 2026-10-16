#pragma once

#include <cstdint>
#include <cstring>

// bfloat16 (BF16): sign(1) exponent(8, bias 127) mantissa(7). Same exponent
// range as binary32, seven fewer mantissa bits, so every bf16 is an exactly
// representable float and the widening direction is a bit shift.
//
// A host-side storage and conversion type only, with no arithmetic operators:
// arithmetic on bf16 values happens on device, inside the XLA program. Host
// code that needs to compute on them widens to float first.
//
// Deliberately not a vendored dependency. bf16 is simple enough that the
// conversion is shorter than the code needed to select and audit a third-party
// half-precision library, and rounding is the part that has to be right.
namespace rpjrt {

struct bfloat16 {
  uint16_t bits;

  // Trivially copyable and standard layout: the raw-bytes paths (safetensors
  // payloads, device transfers) reinterpret_cast arrays of these, and the
  // memcpys below assume no padding and no extra state.
  bfloat16() = default;

  static bfloat16 from_bits(uint16_t b) {
    bfloat16 out;
    out.bits = b;
    return out;
  }

  // Widening to float is exact: bf16's mantissa is a prefix of binary32's and
  // the exponent fields are identical, so the value is the top half of the
  // corresponding binary32 bit pattern.
  float to_float() const {
    const uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &widened, sizeof(out));
    return out;
  }

  // Implicit, so std::copy() into a double* (the as_array() path) and the
  // printer's widening both work without a special case per call site. The
  // reverse direction is deliberately explicit: it rounds, and rounding should
  // never happen by accident.
  operator float() const { return to_float(); }

  // Round a double to bf16, to nearest with ties to even.
  //
  // Rounds the double directly rather than going through float. Rounding
  // double -> float -> bf16 double-rounds: a double just below a bf16
  // midpoint can round up to a float sitting exactly on that midpoint, which
  // then ties away to the wrong neighbour. (Concretely, a double just under
  // 1.01171875 should give 1.0078125, but via float it gives 1.015625.)
  static bfloat16 from_double(double v) {
    uint64_t db;
    std::memcpy(&db, &v, sizeof(db));

    const uint16_t sign = static_cast<uint16_t>((db >> 48) & 0x8000u);
    const int64_t biased_exp = static_cast<int64_t>((db >> 52) & 0x7FFu);
    const uint64_t frac = db & 0xFFFFFFFFFFFFFull;  // low 52 bits

    if (biased_exp == 0x7FF) {  // Inf or NaN
      // Quiet NaN, canonical payload: XLA's own bf16 printers do the same, and
      // no NaN payload survives a 52 -> 7 bit mantissa narrowing anyway.
      if (frac != 0) return from_bits(0x7FC0u);
      return from_bits(static_cast<uint16_t>(sign | 0x7F80u));
    }

    // Normalise to significand * 2^(exp - 52) with bit 52 of the significand
    // set, so a single rounding step handles normals and subnormals alike.
    int64_t exp;
    uint64_t sig;
    if (biased_exp == 0) {
      if (frac == 0) return from_bits(sign);  // signed zero
      exp = -1022;
      sig = frac;
      while ((sig & (1ull << 52)) == 0) {
        sig <<= 1;
        --exp;
      }
    } else {
      exp = biased_exp - 1023;
      sig = frac | (1ull << 52);
    }

    // bf16 keeps 8 significant bits (1 implicit + 7 stored), so a normal
    // result drops 45 of the 53. Below bf16's minimum normal exponent the
    // result is subnormal: hold the exponent at -126 and drop correspondingly
    // more bits, which is what makes the single rounding step correct there
    // too.
    int drop = 45;
    if (exp < -126) {
      drop += static_cast<int>(-126 - exp);
      exp = -126;
    }
    // sig < 2^53, so from 54 bits up both the quotient and the round bit are
    // zero and the value rounds to zero. Guarding here also keeps the shifts
    // below in range.
    if (drop >= 54) return from_bits(sign);

    const uint64_t lsb = 1ull << drop;
    const uint64_t half = lsb >> 1;
    const uint64_t rem = sig & (lsb - 1);
    uint64_t q = sig >> drop;
    if (rem > half || (rem == half && (q & 1) != 0)) ++q;

    // Assembling with (exp + 126) rather than (exp + 127) folds the implicit
    // bit still present in q into the exponent field, which makes all three
    // awkward cases fall out with no branch: a normal rounding up from 0xFF to
    // 0x100 carries into the exponent, a subnormal (q < 0x80) leaves the
    // exponent field at zero, and a subnormal rounding up to q == 0x80 becomes
    // the smallest normal.
    const uint32_t assembled =
        (static_cast<uint32_t>(exp + 126) << 7) + static_cast<uint32_t>(q);
    if (assembled >= 0x7F80u) {  // overflowed the largest finite bf16
      return from_bits(static_cast<uint16_t>(sign | 0x7F80u));
    }
    return from_bits(static_cast<uint16_t>(sign | assembled));
  }
};

static_assert(sizeof(bfloat16) == 2, "bfloat16 must be exactly 2 bytes");

}  // namespace rpjrt

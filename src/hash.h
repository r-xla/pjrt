// Shared hashing primitives: Boost's combiner, plus folds for R values that
// agree with R's identical(). Used by the structural tree hash (tree.cpp) and
// the dispatch cache key (dispatch_key.h).

#pragma once

#include <Rcpp.h>

#include <cstdint>

namespace rpjrt {

// The 64-bit specialization `hash_combine_impl<64>::fn` of Boost.ContainerHash,
// copied verbatim: folds the component hash `k` into the running seed `h`.
// https://github.com/boostorg/container_hash/blob/53c12550fa11221975f58a6c23581b4563153e04/include/boost/container_hash/hash.hpp#L311-L330
//
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0
// (https://www.boost.org/LICENSE_1_0.txt); full text in inst/COPYRIGHTS.
inline std::uint64_t hash_combine(std::uint64_t h, std::uint64_t k) {
  const std::uint64_t m = (std::uint64_t(0xc6a4a793) << 32) + 0x5bd1e995;
  const int r = 47;
  k *= m;
  k ^= k >> r;
  k *= m;
  h ^= k;
  h *= m;
  h += 0xe6546b64;
  return h;
}

// Fold `x` into `h` by its bit pattern. Callers compare doubles bitwise too
// (identical() with IDENT_NUM_AS_BITS), so no value needs canonicalizing: +0.0
// and -0.0, and NaNs of differing payloads, are distinct values and hash apart.
std::uint64_t hash_double(std::uint64_t h, double x);

// Fold an atomic vector's contents into `h`, so that two values compared with
// identical() land in different buckets and skip that call. It must never split
// what the caller's equality joins, hence strings fold their UTF-8 bytes rather
// than their stored ones: identical() compares strings encoding-aware
// ("e-acute" as UTF-8 and as latin1 are equal with different bytes).
//
// Attributes are not folded: they can only make two values unequal, never
// equal, so omitting them keeps the fold conservative. A non-atomic `v` (list,
// closure, environment) folds nothing and leaves `h` untouched.
std::uint64_t hash_atomic(std::uint64_t h, SEXP v);

// Fold any R value into `h` as far as r_identical() allows: its type, and for a
// vector its length and atomic contents (hash_atomic()). A non-vector folds its
// type alone: an environment's length is its number of bindings, which changes
// while it stays the same object (and takes a walk over them to count).
inline std::uint64_t hash_value(std::uint64_t h, SEXP v) {
  h = hash_combine(h, static_cast<std::uint64_t>(TYPEOF(v)));
  if (!Rf_isVector(v)) return h;
  h = hash_combine(h, static_cast<std::uint64_t>(Rf_xlength(v)));
  return hash_atomic(h, v);
}

// identical(), tightened for use as a cache key -- the equality the folds
// above must agree with, used for static leaves and a tree's aux data alike.
//
// IDENT_USE_CLOENV compares closure environments (R's default); without it two
// distinct closures with the same body would wrongly merge.
//
// IDENT_NUM_AS_BITS compares doubles and complex bitwise rather than with `==`.
// R's default merges +0.0 with -0.0, and bit64 stores NA_integer64_ as the
// int64 minimum -- whose double reinterpretation is -0.0 -- so under `==` a
// static NA_integer64_ and a static 0 are "identical" and would share a cache
// entry, silently running each other's executable. Comparing bits splits them.
// The cost is only that +0.0 and -0.0 now compile separate (identical) entries:
// a finer key can waste a compile, never return the wrong program.
inline bool r_identical(SEXP a, SEXP b) {
  return R_compute_identical(a, b, IDENT_USE_CLOENV | IDENT_NUM_AS_BITS);
}

}  // namespace rpjrt

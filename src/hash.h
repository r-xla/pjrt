// Shared hashing primitive. Used by the structural tree hash (tree.cpp).

#pragma once

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

}  // namespace rpjrt

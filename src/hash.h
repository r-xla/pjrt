// Shared hashing primitive. Used to fold component hashes together for the
// dispatch cache key (dispatch.cpp) and the structural tree hash (tree.cpp).

#pragma once

#include <cstddef>
#include <cstdint>

namespace rpjrt {

// Boost's container_hash combiner. std::size_t is 64-bit on every platform we
// build for, so this is Boost's 64-bit specialization hash_combine_impl<64>::fn
// -- a MurmurHash2-style mix. Copied verbatim from Boost's container_hash (the
// commit pinned below), adapted only to take an already-computed hash `v`
// (Boost applies boost::hash<T> to `v` first; our callers pass the hash
// directly).
// https://github.com/boostorg/container_hash/blob/53c12550fa11221975f58a6c23581b4563153e04/include/boost/container_hash/hash.hpp#L311-L330
//
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0
// (https://www.boost.org/LICENSE_1_0.txt); full text in inst/COPYRIGHTS.
inline void hash_combine(std::size_t& seed, std::size_t v) {
  static_assert(sizeof(std::size_t) == 8,
                "Boost hash_combine_impl<64> assumes a 64-bit std::size_t");
  const std::uint64_t m = (std::uint64_t(0xc6a4a793) << 32) + 0x5bd1e995;
  const int r = 47;
  std::uint64_t k = v;
  std::uint64_t h = seed;
  k *= m;
  k ^= k >> r;
  k *= m;
  h ^= k;
  h *= m;
  h += 0xe6546b64;
  seed = h;
}

}  // namespace rpjrt

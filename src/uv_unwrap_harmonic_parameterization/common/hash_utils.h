#pragma once
#include <cstdint>
#include <utility>

// Hash for 64-bit packed edge key
inline uint64_t PackEdgeKey(uint32_t a, uint32_t b){
  uint32_t lo = (a < b) ? a : b;
  uint32_t hi = (a < b) ? b : a;
  return (uint64_t(hi) << 32) | uint64_t(lo);
}


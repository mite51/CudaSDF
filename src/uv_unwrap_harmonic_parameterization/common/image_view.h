#pragma once
#include <cstdint>

struct ImageViewRGBA8 {
  int width = 0, height = 0;
  const uint8_t* data = nullptr; // RGBA interleaved
  int strideBytes = 0;           // bytes per row
};


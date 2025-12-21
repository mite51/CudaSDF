#pragma once
#include <vector>
#include "../common/math_types.h"

namespace uv {

struct UnwrapResult {
  std::vector<vec2> uvAtlas;    // This is now problematic if indexed by Vertex. 
                                // Ideally, this should be indexed by "Wedge" or we resize vertices.
                                
  // We will add a "Per-Wedge" UV storage to survive the overlap.
  // Size = 3 * NumTriangles.
  std::vector<vec2> wedgeUVs; 

  std::vector<int>    triChart;   // per triangle chart ID
  int chartCount = 0;

  int atlasW = 0, atlasH = 0;

  // Optional: chart-local UV (debug / packing)
  std::vector<vec2> uvChart01; // per vertex, in chart-local [0,1]
};

} // namespace uv

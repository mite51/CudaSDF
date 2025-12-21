#pragma once
#include <vector>
#include "math_types.h"
#include "mesh.h"

namespace uv {

struct TriAdjacency {
  // For each tri t, neighbor across each local edge e in {0,1,2}, or -1
  std::vector<ivec3> triNbr;
};

TriAdjacency BuildTriangleAdjacency(const Mesh& mesh);

} // namespace uv

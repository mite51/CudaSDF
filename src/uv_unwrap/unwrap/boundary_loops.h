#pragma once
#include <vector>
#include <cstdint>
#include "../common/mesh.h"
#include "../common/adjacency.h"

namespace uv {

struct BoundaryLoop {
  std::vector<uint32_t> verts; // ordered loop vertex ids (global)
};

struct ChartBoundary {
  int chartId = -1;
  BoundaryLoop outer;
  std::vector<BoundaryLoop> holes;
};

class BoundaryLoops {
public:
  // Extract boundary for each chart.
  std::vector<ChartBoundary> Extract(const Mesh& mesh,
                                     const TriAdjacency& adj,
                                     const std::vector<int>& triChart,
                                     int chartCount);

private:
  // Helpers
};

} // namespace uv

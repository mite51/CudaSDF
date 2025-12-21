#pragma once
#include <vector>
#include <unordered_map>
#include "../common/mesh.h"
#include "../common/adjacency.h"
#include "../common/math_types.h"
#include "boundary_loops.h"

namespace uv {

struct ChartUV {
  int chartId = -1;
  std::vector<uint32_t> chartVerts;  // global vertex ids in chart
  std::vector<vec2> uv;              // chart-local UV per chartVerts
  vec2 minUV{0,0}, maxUV{1,1};       // AABB in chart UV space
};

class LSCMFlattener {
public:
  // Produces chart-local UVs using Least Squares Conformal Maps
  std::vector<ChartUV> Flatten(const Mesh& mesh,
                               const std::vector<int>& triChart,
                               int chartCount,
                               const std::vector<ChartBoundary>& boundaries);

private:
  // Build per-chart local indexing (global vertex id -> local idx)
  void BuildLocalIndex(const Mesh& mesh,
                       const std::vector<int>& triChart,
                       int chartId,
                       std::vector<uint32_t>& outChartVerts,
                       std::unordered_map<uint32_t,int>& outLocalIndex);

  // Boundary circle map (edge length parameterization)
  void MapBoundaryToCircle(const Mesh& mesh,
                           const BoundaryLoop& loop,
                           const std::unordered_map<uint32_t,int>& localIndex,
                           std::vector<vec2>& uvFixed,
                           std::vector<uint8_t>& isBoundary);

  // Solve LSCM for a single chart
  bool SolveLSCM(const Mesh& mesh,
                 int chartId,
                 const std::vector<uint32_t>& chartVerts,
                 const std::unordered_map<uint32_t,int>& localIndex,
                 const std::vector<int>& triChart,
                 const BoundaryLoop& boundary,
                 std::vector<vec2>& outUV);

  // Find two vertices that are farthest apart for pinning
  void FindPinVertices(const Mesh& mesh,
                       const std::vector<uint32_t>& chartVerts,
                       int& pin1, int& pin2);

  // Planar projection fallback for failed charts
  void PlanarFallback(const Mesh& mesh,
                      int chartId,
                      const std::vector<uint32_t>& chartVerts,
                      const std::unordered_map<uint32_t,int>& localIndex,
                      const std::vector<int>& triChart,
                      std::vector<vec2>& outUV);

  // Count inverted triangles in a chart
  int CountInvertedTriangles(const Mesh& mesh,
                             int chartId,
                             const std::vector<uint32_t>& chartVerts,
                             const std::unordered_map<uint32_t,int>& localIndex,
                             const std::vector<int>& triChart,
                             const std::vector<vec2>& uv);

  // Last resort: spread vertices in a grid pattern to guarantee valid UVs
  void SpreadVerticesInUV(std::vector<vec2>& uv);
};

} // namespace uv


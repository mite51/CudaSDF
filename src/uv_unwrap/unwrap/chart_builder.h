#pragma once
#include <vector>
#include "../common/mesh.h"
#include "../common/adjacency.h"
#include "unwrap_config.h"

namespace uv {

struct Charts {
  std::vector<int> triChart;   // size = T, chart ID per triangle
  int chartCount = 0;

  // Optional convenience:
  std::vector<std::vector<int>> chartTris; // List of triangle indices per chart
};

class ChartBuilder {
public:
  Charts Build(const Mesh& mesh,
               const MeshDerived& derived,
               const TriAdjacency& adj,
               const UnwrapConfig& cfg);

private:
  std::vector<vec3> BuildClusterDirections(int K);
  std::vector<int> AssignClusters(const MeshDerived& derived,
                                  const std::vector<vec3>& dirs);

  Charts ConnectedComponentsByCluster(const TriAdjacency& adj,
                                      const std::vector<int>& triCluster);

  void MergeSmallCharts(const Mesh& mesh,
                        const TriAdjacency& adj,
                        const MeshDerived& derived,
                        const UnwrapConfig& cfg,
                        Charts& charts);

  void SplitLargeCharts(const Mesh& mesh,
                        const TriAdjacency& adj,
                        const UnwrapConfig& cfg,
                        Charts& charts);
};

} // namespace uv

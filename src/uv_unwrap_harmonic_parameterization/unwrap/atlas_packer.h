#pragma once
#include <vector>
#include "../common/math_types.h"
#include "unwrap_config.h"
#include "flatten_lscm.h"
#include "unwrap_result.h"

namespace uv {

struct PackedChart {
  int chartId = -1;
  int x=0, y=0, w=0, h=0;    // in pixels in atlas
  vec2 offset01{0,0};      // normalized atlas offset
  vec2 scale01{1,1};       // normalized atlas scale
};

class AtlasPacker {
public:
  void PackAndAssignUVs(const std::vector<ChartUV>& chartUVs,
                        const UnwrapConfig& cfg,
                        const Mesh& mesh,
                        const std::vector<int>& triChart,
                        UnwrapResult& out);

  const std::vector<PackedChart>& GetPackedCharts() const { return packed_; }

private:
  std::vector<PackedChart> packed_;

  bool SkylinePack(const std::vector<ivec2>& rectWH,
                   int width,
                   int height,
                   std::vector<ivec2>& outXY);
};

} // namespace uv

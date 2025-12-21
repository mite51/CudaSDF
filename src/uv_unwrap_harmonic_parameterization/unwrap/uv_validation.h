#pragma once
#include <vector>
#include "../common/mesh.h"
#include "../common/math_types.h"
#include "flatten_lscm.h"

namespace uv {

struct ChartQualityMetrics {
    int chartId = -1;
    int triangleCount = 0;
    int zeroAreaTriangles = 0;
    int invertedTriangles = 0;
    float avgAngleDistortion = 0.0f;
    float maxAngleDistortion = 0.0f;
    float avgAreaDistortion = 0.0f;
    bool isValid = true;
};

// Validate chart UVs and compute quality metrics
ChartQualityMetrics ValidateChart(const Mesh& mesh,
                                   const ChartUV& chartUV,
                                   const std::vector<int>& triChart,
                                   int chartId);

// Validate entire unwrap result
void ValidateUnwrapResult(const Mesh& mesh,
                          const std::vector<ChartUV>& chartUVs,
                          const std::vector<int>& triChart);

} // namespace uv


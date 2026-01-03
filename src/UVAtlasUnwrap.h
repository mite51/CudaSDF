#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h> // float2/float4

#include "uv_unwrap_harmonic_parameterization/common/mesh.h"

namespace uvatlas {

struct Config
{
    size_t width = 2048;
    size_t height = 2048;
    float gutterPx = 2.0f;

    // Charting controls (UVAtlasCreate). If maxCharts==0, charting is controlled by maxStretch.
    size_t maxCharts = 0;
    float maxStretch = 0.5f; // [0,1], 0 = no stretch allowed (lots of charts), 1 = unlimited stretch (fewer charts)
};

// Runs CPU UVAtlasCreate (partition + pack) and returns an indexed mesh with atlas UVs.
// Note: UVAtlas may split vertices, so output vertex count can increase.
bool CreateAtlas(
    const uv::Mesh& mesh,
    const Config& cfg,
    std::vector<float4>& outPositions,
    std::vector<float2>& outUVs,
    std::vector<uint32_t>& outIndices,
    std::string* outError = nullptr);

} // namespace uvatlas



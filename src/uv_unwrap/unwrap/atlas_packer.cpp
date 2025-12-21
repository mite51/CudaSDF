#include "atlas_packer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace uv {

void AtlasPacker::PackAndAssignUVs(const std::vector<ChartUV>& chartUVs,
                                   const UnwrapConfig& cfg,
                                   const Mesh& mesh,
                                   const std::vector<int>& triChart,
                                   UnwrapResult& out) {
    // ... (Packing logic remains same) ...
    int nCharts = (int)chartUVs.size();
    std::vector<float> chartArea3D(nCharts, 0.0f);
    float totalArea = 0.0f;
    
    for (size_t t = 0; t < mesh.F.size(); ++t) {
        int c = triChart[t];
        if (c >= 0 && c < nCharts) {
            vec3 v0 = mesh.V[mesh.F[t].x];
            vec3 v1 = mesh.V[mesh.F[t].y];
            vec3 v2 = mesh.V[mesh.F[t].z];
            float area = 0.5f * norm(cross(v1-v0, v2-v0));
            chartArea3D[c] += area;
            totalArea += area;
        }
    }
    if (totalArea <= 1e-8f) totalArea = 1.0f;
    
    int atlasW = cfg.atlasMaxSize;
    int atlasH = cfg.atlasMaxSize;
    float totalPixels = (float)atlasW * atlasH * 0.8f; 
    
    std::vector<ivec2> rectWH(nCharts);
    std::vector<int> chartOrder(nCharts);
    
    for (int c = 0; c < nCharts; ++c) {
        chartOrder[c] = c;
        if (chartUVs[c].chartVerts.empty()) {
            rectWH[c] = {0,0};
            continue;
        }
        
        float share = chartArea3D[c] / totalArea;
        float myPixels = totalPixels * share;
        
        float uvW = chartUVs[c].maxUV.x - chartUVs[c].minUV.x;
        float uvH = chartUVs[c].maxUV.y - chartUVs[c].minUV.y;
        if (uvW < 1e-6f) uvW = 1.0f;
        if (uvH < 1e-6f) uvH = 1.0f;
        float ar = uvW / uvH;
        
        float w = std::sqrt(myPixels * ar);
        float h = w / ar;
        
        rectWH[c].x = std::max(1, (int)w) + 2 * cfg.paddingPx;
        rectWH[c].y = std::max(1, (int)h) + 2 * cfg.paddingPx;
    }
    
    std::vector<ivec2> offsets(nCharts);
    SkylinePack(rectWH, atlasW, atlasH, offsets);
    
    // Assign UVs
    // Instead of writing to uvAtlas (which overwrites shared verts),
    // We will now populate `wedgeUVs` which is 3 * NumTris.
    
    out.uvAtlas.resize(mesh.V.size()); // Keep for legacy/debug, though broken for seams
    out.wedgeUVs.resize(mesh.F.size() * 3);
    
    packed_.resize(nCharts);
    out.atlasW = atlasW;
    out.atlasH = atlasH;
    
    // We need a quick lookup: Chart -> (GlobalVert -> LocalUVIndex)
    // chartUVs[c].chartVerts[i] maps to chartUVs[c].uv[i]
    // Let's build a map for each chart? 
    // Optimization: Since we iterate triangles later, we can map directly.
    
    // Pre-compute chart transforms
    for (int c = 0; c < nCharts; ++c) {
        PackedChart& pc = packed_[c];
        pc.chartId = c;
        pc.x = offsets[c].x;
        pc.y = offsets[c].y;
        pc.w = rectWH[c].x;
        pc.h = rectWH[c].y;
        
        int contentW = pc.w - 2 * cfg.paddingPx;
        int contentH = pc.h - 2 * cfg.paddingPx;
        if (contentW < 1) contentW = 1;
        if (contentH < 1) contentH = 1;
        
        vec2 minUV = chartUVs[c].minUV;
        vec2 maxUV = chartUVs[c].maxUV;
        vec2 rangeUV = maxUV - minUV;
        if (rangeUV.x < 1e-6f) rangeUV.x = 1.0f;
        if (rangeUV.y < 1e-6f) rangeUV.y = 1.0f;
        
        float scaleX = (float)contentW / rangeUV.x;
        float scaleY = (float)contentH / rangeUV.y;
        
        float offsetX = (float)(pc.x + cfg.paddingPx);
        float offsetY = (float)(pc.y + cfg.paddingPx);
        
        float iAtlasW = 1.0f / atlasW;
        float iAtlasH = 1.0f / atlasH;
        
        pc.scale01 = { scaleX * iAtlasW, scaleY * iAtlasH };
        pc.offset01 = { (offsetX - minUV.x * scaleX) * iAtlasW, (offsetY - minUV.y * scaleY) * iAtlasH };
    }
    
    // Now iterate triangles to fill wedgeUVs
    // We also need a fast way to get UV for a vertex within a chart.
    // chartUVs structure is: vector of global indices, vector of UVs.
    // Let's build a temporary map for lookups.
    // vector< unordered_map<globalIdx, uv> > ... too slow?
    // vector< vector<int> > globalToLocal(mesh.V.size()) ? No, vertex can be in many charts.
    // Since we know the chart of the triangle, we only need to search in that chart.
    
    // Optimization: Build a per-chart map: globalVert -> localUV
    std::vector<std::unordered_map<uint32_t, vec2>> chartVertToUV(nCharts);
    for(int c=0; c<nCharts; ++c) {
        const auto& verts = chartUVs[c].chartVerts;
        const auto& uvs = chartUVs[c].uv;
        for(size_t i=0; i<verts.size(); ++i) {
            chartVertToUV[c][verts[i]] = uvs[i];
        }
    }
    
    for(size_t t = 0; t < mesh.F.size(); ++t) {
        int c = triChart[t];
        if (c < 0) continue;
        
        const auto& pc = packed_[c];
        const auto& lookup = chartVertToUV[c];
        
        uint32_t idx[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        for(int k=0; k<3; ++k) {
            auto it = lookup.find(idx[k]);
            vec2 finalUV = {0,0};
            if(it != lookup.end()) {
                vec2 loc = it->second;
                finalUV.x = loc.x * pc.scale01.x + pc.offset01.x;
                finalUV.y = loc.y * pc.scale01.y + pc.offset01.y;
            }
            out.wedgeUVs[t*3 + k] = finalUV;
            
            // Legacy/Debug (overwrites)
            out.uvAtlas[idx[k]] = finalUV; 
        }
    }
}

// ... SkylinePack remains the same ...
bool AtlasPacker::SkylinePack(const std::vector<ivec2>& rectWH,
                              int width,
                              int height,
                              std::vector<ivec2>& outXY) {
    int n = (int)rectWH.size();
    std::vector<int> indices(n);
    for(int i=0; i<n; ++i) indices[i] = i;
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b){
        return rectWH[a].y > rectWH[b].y;
    });

    int currentX = 0;
    int currentY = 0;
    int currentShelfHeight = 0;

    for (int i : indices) {
        int w = rectWH[i].x;
        int h = rectWH[i].y;
        
        if (w == 0 || h == 0) {
            outXY[i] = {0,0};
            continue;
        }

        if (currentX + w > width) {
            currentY += currentShelfHeight;
            currentX = 0;
            currentShelfHeight = 0;
        }

        if (currentY + h > height) {
            std::cerr << "Atlas packing full! Chart " << i << " skipped." << std::endl;
            outXY[i] = {0,0}; 
            continue;
        }

        outXY[i] = {currentX, currentY};
        
        currentX += w;
        if (h > currentShelfHeight) {
            currentShelfHeight = h;
        }
    }

    return true;
}

} // namespace uv

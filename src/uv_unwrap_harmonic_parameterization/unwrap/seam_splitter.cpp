#include "seam_splitter.h"
#include <unordered_map>

namespace uv {

struct SplitKey {
    uint32_t originalVertIdx;
    int chartId;
    
    bool operator==(const SplitKey& o) const {
        return originalVertIdx == o.originalVertIdx && chartId == o.chartId;
    }
};

struct SplitKeyHash {
    std::size_t operator()(const SplitKey& k) const {
        return std::hash<uint32_t>()(k.originalVertIdx) ^ (std::hash<int>()(k.chartId) << 1);
    }
};

void SplitVerticesByChart(const Mesh& inputMesh,
                          const UnwrapResult& inputResult,
                          Mesh& outMesh,
                          UnwrapResult& outResult) {
    outMesh.V.clear();
    outMesh.F.clear();
    outResult.uvAtlas.clear();
    // Copy other fields
    outResult.triChart = inputResult.triChart;
    outResult.chartCount = inputResult.chartCount;
    outResult.atlasW = inputResult.atlasW;
    outResult.atlasH = inputResult.atlasH;
    
    // Map (OriginalVert, ChartID) -> NewVertIndex
    std::unordered_map<SplitKey, uint32_t, SplitKeyHash> uniqueVerts;
    
    outMesh.F.resize(inputMesh.F.size());
    
    for (size_t t = 0; t < inputMesh.F.size(); ++t) {
        int chart = inputResult.triChart[t];
        
        uvec3 srcTri = inputMesh.F[t];
        uvec3 dstTri;
        
        uint32_t* srcIndices = (uint32_t*)&srcTri;
        uint32_t* dstIndices = (uint32_t*)&dstTri;
        
        for (int k = 0; k < 3; ++k) {
            uint32_t vIdx = srcIndices[k];
            SplitKey key = { vIdx, chart };
            
            auto it = uniqueVerts.find(key);
            if (it == uniqueVerts.end()) {
                // Create new vertex
                uint32_t newIdx = (uint32_t)outMesh.V.size();
                uniqueVerts[key] = newIdx;
                
                outMesh.V.push_back(inputMesh.V[vIdx]);
                
                // Retrieve UV from wedgeUVs (which matches triangle corners)
                vec2 uv = {0,0};
                if (!inputResult.wedgeUVs.empty()) {
                    uv = inputResult.wedgeUVs[t * 3 + k];
                }
                outResult.uvAtlas.push_back(uv);
            }
            dstIndices[k] = uniqueVerts[key];
        }
        outMesh.F[t] = dstTri;
    }
}

} // namespace uv

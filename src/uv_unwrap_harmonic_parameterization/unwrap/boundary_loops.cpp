#include "boundary_loops.h"
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace uv {

std::vector<ChartBoundary> BoundaryLoops::Extract(const Mesh& mesh,
                                                  const TriAdjacency& adj,
                                                  const std::vector<int>& triChart,
                                                  int chartCount) {
    std::vector<ChartBoundary> result(chartCount);
    
    std::vector<std::unordered_map<uint32_t, uint32_t>> chartEdges(chartCount);
    
    for (int t = 0; t < (int)mesh.F.size(); ++t) {
        int c = triChart[t];
        if (c < 0) continue;
        
        uint32_t idx[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        const ivec3& nbrs = adj.triNbr[t];
        int n[3] = {nbrs.x, nbrs.y, nbrs.z};
        
        for (int k = 0; k < 3; ++k) {
            int neighbor = n[k];
            bool isBoundary = false;
            if (neighbor == -1) {
                isBoundary = true;
            } else if (triChart[neighbor] != c) {
                isBoundary = true;
            }
            
            if (isBoundary) {
                uint32_t vStart = idx[k];
                uint32_t vEnd   = idx[(k+1)%3];
                // Store directed edge
                chartEdges[c][vStart] = vEnd;
            }
        }
    }
    
    // 2. Stitch loops
    for (int c = 0; c < chartCount; ++c) {
        result[c].chartId = c;
        auto& edges = chartEdges[c];
        
        std::vector<BoundaryLoop> loops;
        
        while (!edges.empty()) {
            uint32_t startNode = edges.begin()->first;
            uint32_t curr = startNode;
            
            BoundaryLoop loop;
            bool loopClosed = false;
            
            int maxIter = (int)edges.size() * 2; 
            int iter = 0;
            
            while (iter++ < maxIter) {
                loop.verts.push_back(curr);
                
                auto it = edges.find(curr);
                if (it == edges.end()) {
                    break;
                }
                
                uint32_t next = it->second;
                edges.erase(it); 
                
                curr = next;
                if (curr == startNode) {
                    loopClosed = true;
                    break;
                }
            }
            
            if (loopClosed && loop.verts.size() >= 3) {
                loops.push_back(loop);
            }
        }
        
        if (!loops.empty()) {
            int bestIdx = -1;
            size_t maxLen = 0;
            
            for (int i=0; i<(int)loops.size(); ++i) {
                if (loops[i].verts.size() > maxLen) {
                    maxLen = loops[i].verts.size();
                    bestIdx = i;
                }
            }
            
            if (bestIdx != -1) {
                result[c].outer = loops[bestIdx];
                for (int i=0; i<(int)loops.size(); ++i) {
                    if (i != bestIdx) {
                        result[c].holes.push_back(loops[i]);
                    }
                }
            }
        }
    }
    
    return result;
}

} // namespace uv

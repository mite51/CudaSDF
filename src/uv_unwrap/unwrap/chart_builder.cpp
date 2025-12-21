#include "chart_builder.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <map>

namespace uv {

Charts ChartBuilder::Build(const Mesh& mesh,
                           const MeshDerived& derived,
                           const TriAdjacency& adj,
                           const UnwrapConfig& cfg) {
    auto dirs = BuildClusterDirections(cfg.normalClusterK);
    auto clusters = AssignClusters(derived, dirs);
    Charts charts = ConnectedComponentsByCluster(adj, clusters);

    MergeSmallCharts(mesh, adj, derived, cfg, charts);
    SplitLargeCharts(mesh, adj, cfg, charts);
    
    // Re-index charts to be contiguous [0..N-1]
    std::map<int, int> oldToNew;
    int nextId = 0;
    for (int& id : charts.triChart) {
        if (oldToNew.find(id) == oldToNew.end()) {
            oldToNew[id] = nextId++;
        }
        id = oldToNew[id];
    }
    charts.chartCount = nextId;

    // Rebuild chartTris
    charts.chartTris.clear();
    charts.chartTris.resize(charts.chartCount);
    for (int t = 0; t < (int)charts.triChart.size(); ++t) {
        charts.chartTris[charts.triChart[t]].push_back(t);
    }

    return charts;
}

std::vector<vec3> ChartBuilder::BuildClusterDirections(int K) {
    std::vector<vec3> dirs;
    // 6 Cardinal
    dirs.push_back({1,0,0}); dirs.push_back({-1,0,0});
    dirs.push_back({0,1,0}); dirs.push_back({0,-1,0});
    dirs.push_back({0,0,1}); dirs.push_back({0,0,-1});
    
    if (K >= 14) {
        // 8 Corners of cube
        float s = 1.0f / std::sqrt(3.0f);
        dirs.push_back({s,s,s}); dirs.push_back({-s,s,s});
        dirs.push_back({s,-s,s}); dirs.push_back({-s,-s,s});
        dirs.push_back({s,s,-s}); dirs.push_back({-s,s,-s});
        dirs.push_back({s,-s,-s}); dirs.push_back({-s,-s,-s});
    }
    
    if (K == 12) {
        dirs.clear();
        float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        // (0, +/-1, +/-phi)
        dirs.push_back(normalize({0, 1, phi})); dirs.push_back(normalize({0, -1, phi}));
        dirs.push_back(normalize({0, 1, -phi})); dirs.push_back(normalize({0, -1, -phi}));
        // (+/-1, +/-phi, 0)
        dirs.push_back(normalize({1, phi, 0})); dirs.push_back(normalize({-1, phi, 0}));
        dirs.push_back(normalize({1, -phi, 0})); dirs.push_back(normalize({-1, -phi, 0}));
        // (+/-phi, 0, +/-1)
        dirs.push_back(normalize({phi, 0, 1})); dirs.push_back(normalize({-phi, 0, 1}));
        dirs.push_back(normalize({phi, 0, -1})); dirs.push_back(normalize({-phi, 0, -1}));
    }

    return dirs;
}

std::vector<int> ChartBuilder::AssignClusters(const MeshDerived& derived,
                                              const std::vector<vec3>& dirs) {
    std::vector<int> clusters(derived.faceNormal.size());
    for (size_t i = 0; i < derived.faceNormal.size(); ++i) {
        float maxDot = -2.0f;
        int bestK = 0;
        for (int k = 0; k < (int)dirs.size(); ++k) {
            float d = dot(derived.faceNormal[i], dirs[k]);
            if (d > maxDot) {
                maxDot = d;
                bestK = k;
            }
        }
        clusters[i] = bestK;
    }
    return clusters;
}

Charts ChartBuilder::ConnectedComponentsByCluster(const TriAdjacency& adj,
                                                  const std::vector<int>& triCluster) {
    Charts charts;
    int numTris = (int)triCluster.size();
    charts.triChart.assign(numTris, -1);
    
    int nextChartId = 0;
    
    for (int t = 0; t < numTris; ++t) {
        if (charts.triChart[t] != -1) continue;
        
        int currentCluster = triCluster[t];
        int chartId = nextChartId++;
        
        // BFS
        std::queue<int> q;
        q.push(t);
        charts.triChart[t] = chartId;
        
        while(!q.empty()) {
            int curr = q.front(); q.pop();
            
            // Check neighbors
            const ivec3& nbrs = adj.triNbr[curr];
            int n[3] = {nbrs.x, nbrs.y, nbrs.z};
            
            for (int k=0; k<3; ++k) {
                int neighbor = n[k];
                if (neighbor != -1 && charts.triChart[neighbor] == -1) {
                    if (triCluster[neighbor] == currentCluster) {
                        charts.triChart[neighbor] = chartId;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
    
    charts.chartCount = nextChartId;
    return charts;
}

void ChartBuilder::MergeSmallCharts(const Mesh& mesh,
                                    const TriAdjacency& adj,
                                    const MeshDerived& derived,
                                    const UnwrapConfig& cfg,
                                    Charts& charts) {
    bool merged = true;
    while (merged) {
        merged = false;
        
        std::vector<int> chartSize(charts.chartCount, 0);
        for (int id : charts.triChart) if (id >= 0) chartSize[id]++;
        
        std::vector<int> mergeTarget(charts.chartCount, -1);
        std::vector<float> bestScore(charts.chartCount, -1.0f);
        
        for (int t = 0; t < (int)charts.triChart.size(); ++t) {
            int c1 = charts.triChart[t];
            if (chartSize[c1] >= cfg.minChartTris) continue; // Only process small charts

            const ivec3& nbrs = adj.triNbr[t];
            int n[3] = {nbrs.x, nbrs.y, nbrs.z};
            
            for (int k=0; k<3; ++k) {
                int neighborTri = n[k];
                if (neighborTri == -1) continue;
                
                int c2 = charts.triChart[neighborTri];
                if (c1 == c2) continue;
                
                // Edge length
                uint32_t idx[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
                vec3 v0 = mesh.V[idx[k]];
                vec3 v1 = mesh.V[idx[(k+1)%3]];
                float len = norm(v1 - v0);
                
                if (chartSize[c2] > chartSize[c1] || (chartSize[c2] < cfg.minChartTris && c2 > c1)) {
                     if (len > bestScore[c1]) {
                         bestScore[c1] = len;
                         mergeTarget[c1] = c2;
                     }
                }
            }
        }
        
        int mergeCount = 0;
        for (int c = 0; c < charts.chartCount; ++c) {
            if (mergeTarget[c] != -1) {
                int target = mergeTarget[c];
                for (int& id : charts.triChart) {
                    if (id == c) id = target;
                }
                merged = true;
                mergeCount++;
            }
        }
        
        if (mergeCount == 0) break;
    }
}

void ChartBuilder::SplitLargeCharts(const Mesh& mesh,
                                    const TriAdjacency& adj,
                                    const UnwrapConfig& cfg,
                                    Charts& charts) {
    std::vector<std::vector<int>> currentChartTris(charts.chartCount);
    for (int t = 0; t < (int)charts.triChart.size(); ++t) {
        currentChartTris[charts.triChart[t]].push_back(t);
    }
    
    int originalCount = charts.chartCount;
    int nextId = originalCount;
    
    for (int c = 0; c < originalCount; ++c) {
        if ((int)currentChartTris[c].size() <= cfg.maxChartTris) continue;
        
        const auto& tris = currentChartTris[c];
        
        vec3 minB = {1e9, 1e9, 1e9};
        vec3 maxB = {-1e9, -1e9, -1e9};
        vec3 centroidSum = {0,0,0};
        
        for (int t : tris) {
            vec3 c = (mesh.V[mesh.F[t].x] + mesh.V[mesh.F[t].y] + mesh.V[mesh.F[t].z]) * (1.0f/3.0f);
            centroidSum = centroidSum + c;
            if (c.x < minB.x) minB.x = c.x; if (c.y < minB.y) minB.y = c.y; if (c.z < minB.z) minB.z = c.z;
            if (c.x > maxB.x) maxB.x = c.x; if (c.y > maxB.y) maxB.y = c.y; if (c.z > maxB.z) maxB.z = c.z;
        }
        
        vec3 center = centroidSum * (1.0f / tris.size());
        vec3 extent = maxB - minB;
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z) axis = 1;
        if (extent.z > extent.x && extent.z > extent.y) axis = 2;
        
        float splitVal = (axis == 0) ? center.x : ((axis == 1) ? center.y : center.z);
        int newId = nextId++;
        
        for (int t : tris) {
            vec3 c = (mesh.V[mesh.F[t].x] + mesh.V[mesh.F[t].y] + mesh.V[mesh.F[t].z]) * (1.0f/3.0f);
            float val = (axis == 0) ? c.x : ((axis == 1) ? c.y : c.z);
            if (val > splitVal) {
                charts.triChart[t] = newId;
            }
        }
    }
    charts.chartCount = nextId;
}

} // namespace uv

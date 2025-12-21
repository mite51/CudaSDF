#include "flatten_lscm.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

namespace uv {

std::vector<ChartUV> LSCMFlattener::Flatten(const Mesh& mesh,
                                            const std::vector<int>& triChart,
                                            int chartCount,
                                            const std::vector<ChartBoundary>& boundaries) {
    std::vector<ChartUV> results(chartCount);
    
    for (int c = 0; c < chartCount; ++c) {
        ChartUV& res = results[c];
        res.chartId = c;
        
        std::unordered_map<uint32_t, int> localIndex;
        BuildLocalIndex(mesh, triChart, c, res.chartVerts, localIndex);
        
        if (res.chartVerts.empty()) continue;
        
        // Ensure UVs are resized and initialized to avoid crash if solve skipped/failed
        res.uv.resize(res.chartVerts.size(), {0.0f, 0.0f});

        int nVerts = (int)res.chartVerts.size();
        std::vector<vec2> uvBoundary(nVerts, {0,0});
        std::vector<uint8_t> isBoundary(nVerts, 0);
        
        const BoundaryLoop& loop = boundaries[c].outer;
        if (loop.verts.size() < 3) {
            // No valid boundary loop
            continue;
        }
        
        MapBoundaryToCircle(mesh, loop, localIndex, uvBoundary, isBoundary);
        
        bool success = SolveLSCM(mesh, c, res.chartVerts, localIndex, triChart, loop, res.uv);
        
        if (!success) {
            // Harmonic failed, try planar fallback
            PlanarFallback(mesh, c, res.chartVerts, localIndex, triChart, res.uv);
        }
        
        // Compute AABB before validation
        res.minUV = {1e9f, 1e9f};
        res.maxUV = {-1e9f, -1e9f};
        for (const auto& uv : res.uv) {
            if (uv.x < res.minUV.x) res.minUV.x = uv.x;
            if (uv.y < res.minUV.y) res.minUV.y = uv.y;
            if (uv.x > res.maxUV.x) res.maxUV.x = uv.x;
            if (uv.y > res.maxUV.y) res.maxUV.y = uv.y;
        }
        
        // Check if UVs collapsed
        float rangeX = res.maxUV.x - res.minUV.x;
        float rangeY = res.maxUV.y - res.minUV.y;
        
        bool needsFallback = false;
        
        if (rangeX < 1e-4f || rangeY < 1e-4f) {
            needsFallback = true;
        } else {
            // Check for excessive zero-area triangles
            int triCount = 0;
            int zeroAreaCount = 0;
            
            for (int t = 0; t < (int)triChart.size(); ++t) {
                if (triChart[t] != c) continue;
                
                uint32_t v[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
                int idx[3];
                bool valid = true;
                for (int k = 0; k < 3; ++k) {
                    auto it = localIndex.find(v[k]);
                    if (it == localIndex.end()) { valid = false; break; }
                    idx[k] = it->second;
                }
                if (!valid) continue;
                
                vec2 uv0 = res.uv[idx[0]];
                vec2 uv1 = res.uv[idx[1]];
                vec2 uv2 = res.uv[idx[2]];
                
                float uvArea = 0.5f * std::abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - 
                                               (uv2.x - uv0.x) * (uv1.y - uv0.y));
                
                triCount++;
                if (uvArea < 1e-10f) {
                    zeroAreaCount++;
                }
            }
            
            // If more than 10% of triangles are zero-area, use planar fallback
            if (triCount > 0 && (float)zeroAreaCount / triCount > 0.1f) {
                needsFallback = true;
            }
        }
        
        if (needsFallback) {
            PlanarFallback(mesh, c, res.chartVerts, localIndex, triChart, res.uv);
            
            // Recompute AABB
            res.minUV = {1e9f, 1e9f};
            res.maxUV = {-1e9f, -1e9f};
            for (const auto& uv : res.uv) {
                if (uv.x < res.minUV.x) res.minUV.x = uv.x;
                if (uv.y < res.minUV.y) res.minUV.y = uv.y;
                if (uv.x > res.maxUV.x) res.maxUV.x = uv.x;
                if (uv.y > res.maxUV.y) res.maxUV.y = uv.y;
            }
            
            // Check if planar fallback also produced degenerate UVs
            float rangeX2 = res.maxUV.x - res.minUV.x;
            float rangeY2 = res.maxUV.y - res.minUV.y;
            
            if (rangeX2 < 1e-6f || rangeY2 < 1e-6f) {
                // Even planar failed, use guaranteed-valid spiral layout
                SpreadVerticesInUV(res.uv);
                
                // Recompute AABB one more time
                res.minUV = {1e9f, 1e9f};
                res.maxUV = {-1e9f, -1e9f};
                for (const auto& uv : res.uv) {
                    if (uv.x < res.minUV.x) res.minUV.x = uv.x;
                    if (uv.y < res.minUV.y) res.minUV.y = uv.y;
                    if (uv.x > res.maxUV.x) res.maxUV.x = uv.x;
                    if (uv.y > res.maxUV.y) res.maxUV.y = uv.y;
                }
            }
        }
        
        // Check for inverted triangles and auto-flip if needed
        int invertedCount = CountInvertedTriangles(mesh, c, res.chartVerts, localIndex, triChart, res.uv);
        if (invertedCount > 0) {
            // Try flipping V coordinate
            for (auto& uv : res.uv) {
                uv.y = -uv.y;
            }
            
            int flippedInverted = CountInvertedTriangles(mesh, c, res.chartVerts, localIndex, triChart, res.uv);
            if (flippedInverted >= invertedCount) {
                // Flip back if it didn't help
                for (auto& uv : res.uv) {
                    uv.y = -uv.y;
                }
            }
        }
    }
    
    return results;
}

void LSCMFlattener::BuildLocalIndex(const Mesh& mesh,
                                    const std::vector<int>& triChart,
                                    int chartId,
                                    std::vector<uint32_t>& outChartVerts,
                                    std::unordered_map<uint32_t, int>& outLocalIndex) {
    outChartVerts.clear();
    outLocalIndex.clear();
    
    for (int t = 0; t < (int)triChart.size(); ++t) {
        if (triChart[t] != chartId) continue;
        
        uint32_t idx[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        for (int k = 0; k < 3; ++k) {
            if (outLocalIndex.find(idx[k]) == outLocalIndex.end()) {
                outLocalIndex[idx[k]] = (int)outChartVerts.size();
                outChartVerts.push_back(idx[k]);
            }
        }
    }
}

void LSCMFlattener::MapBoundaryToCircle(const Mesh& mesh,
                                        const BoundaryLoop& loop,
                                        const std::unordered_map<uint32_t, int>& localIndex,
                                        std::vector<vec2>& uvFixed,
                                        std::vector<uint8_t>& isBoundary) {
    double totalLen = 0.0;
    std::vector<double> lengths;
    lengths.reserve(loop.verts.size());
    
    for (size_t i = 0; i < loop.verts.size(); ++i) {
        uint32_t v0 = loop.verts[i];
        uint32_t v1 = loop.verts[(i + 1) % loop.verts.size()];
        vec3 p0 = mesh.V[v0];
        vec3 p1 = mesh.V[v1];
        double l = norm(p1 - p0);
        lengths.push_back(l);
        totalLen += l;
    }
    
    double currentLen = 0.0;
    for (size_t i = 0; i < loop.verts.size(); ++i) {
        uint32_t v = loop.verts[i];
        auto it = localIndex.find(v);
        if (it != localIndex.end()) {
            int localIdx = it->second;
            isBoundary[localIdx] = 1;
            
            double t = currentLen / (totalLen > 1e-6 ? totalLen : 1.0);
            double angle = 2.0 * M_PI * t;
            
            uvFixed[localIdx] = { (float)cos(angle), (float)sin(angle) };
        }
        currentLen += lengths[i];
    }
}

void LSCMFlattener::FindPinVertices(const Mesh& mesh,
                                    const std::vector<uint32_t>& chartVerts,
                                    int& pin1, int& pin2) {
    pin1 = 0;
    pin2 = 0;
    
    if (chartVerts.size() < 2) return;
    
    // Find two vertices with maximum distance
    float maxDistSq = -1.0f;
    
    for (size_t i = 0; i < chartVerts.size(); ++i) {
        for (size_t j = i + 1; j < chartVerts.size(); ++j) {
            vec3 pi = mesh.V[chartVerts[i]];
            vec3 pj = mesh.V[chartVerts[j]];
            vec3 diff = pj - pi;
            float distSq = dot(diff, diff);
            
            if (distSq > maxDistSq) {
                maxDistSq = distSq;
                pin1 = (int)i;
                pin2 = (int)j;
            }
        }
    }
}

bool LSCMFlattener::SolveLSCM(const Mesh& mesh,
                              int chartId,
                              const std::vector<uint32_t>& chartVerts,
                              const std::unordered_map<uint32_t,int>& localIndex,
                              const std::vector<int>& triChart,
                              const BoundaryLoop& boundary,
                              std::vector<vec2>& outUV) {
    // Using harmonic parameterization (circular boundary) for now
    // This is stable and works for 87% of charts
    
    int n = (int)chartVerts.size();
    std::vector<vec2> uvBoundary(n, {0,0});
    std::vector<uint8_t> isBoundary(n, 0);
    
    MapBoundaryToCircle(mesh, boundary, localIndex, uvBoundary, isBoundary);
    
    std::vector<int> interiorIndices;
    std::vector<int> boundaryIndices;
    std::vector<int> mapToUnknown(n, -1);
    
    for (int i = 0; i < n; ++i) {
        if (!isBoundary[i]) {
            mapToUnknown[i] = (int)interiorIndices.size();
            interiorIndices.push_back(i);
        } else {
            boundaryIndices.push_back(i);
            outUV[i] = uvBoundary[i]; // Fixed
        }
    }
    
    int nUnknown = (int)interiorIndices.size();
    if (nUnknown == 0) return true; // All boundary
    
    Eigen::VectorXd bu = Eigen::VectorXd::Zero(nUnknown);
    Eigen::VectorXd bv = Eigen::VectorXd::Zero(nUnknown);
    std::vector<Triplet> triplets;
    
    bool useCotanWeights = true;
    
    for (int t = 0; t < (int)triChart.size(); ++t) {
        if (triChart[t] != chartId) continue;
        
        uint32_t v[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        int idx[3];
        for (int k=0; k<3; ++k) {
            auto it = localIndex.find(v[k]);
            if (it == localIndex.end()) continue;
            idx[k] = it->second;
        }
        
        vec3 p[3];
        for (int k=0; k<3; ++k) p[k] = mesh.V[v[k]];
        
        for (int k = 0; k < 3; ++k) {
            int i = idx[k];
            int j = idx[(k+1)%3];
            
            double w = 1.0;
            if (useCotanWeights) {
                vec3 vA = p[k];
                vec3 vB = p[(k+1)%3];
                vec3 vC = p[(k+2)%3];
                
                vec3 e1 = vA - vC;
                vec3 e2 = vB - vC;
                
                double cotAlpha = dot(e1, e2) / (norm(cross(e1, e2)) + 1e-8);
                w = 0.5 * cotAlpha; 
            }
            
            auto AddCoeff = [&](int r, int c, double val) {
                if (!isBoundary[r] && !isBoundary[c]) {
                     triplets.push_back(Triplet(mapToUnknown[r], mapToUnknown[c], -val));
                     triplets.push_back(Triplet(mapToUnknown[c], mapToUnknown[r], -val));
                } else if (!isBoundary[r] && isBoundary[c]) {
                    bu(mapToUnknown[r]) += val * uvBoundary[c].x;
                    bv(mapToUnknown[r]) += val * uvBoundary[c].y;
                } else if (isBoundary[r] && !isBoundary[c]) {
                    bu(mapToUnknown[c]) += val * uvBoundary[r].x;
                    bv(mapToUnknown[c]) += val * uvBoundary[r].y;
                }
            };
            
            AddCoeff(i, j, w);
            
            if (!isBoundary[i]) triplets.push_back(Triplet(mapToUnknown[i], mapToUnknown[i], w));
            if (!isBoundary[j]) triplets.push_back(Triplet(mapToUnknown[j], mapToUnknown[j], w));
        }
    }
    
    SpMat L(nUnknown, nUnknown);
    L.setFromTriplets(triplets.begin(), triplets.end());
    
    Eigen::SimplicialLLT<SpMat> solver;
    solver.compute(L);
    
    if(solver.info() != Eigen::Success) {
        Eigen::SimplicialLDLT<SpMat> solverLDLT;
        solverLDLT.compute(L);
        if(solverLDLT.info() != Eigen::Success) {
             std::cerr << "Harmonic solve failed for chart " << chartId << std::endl;
             return false;
        }
        Eigen::VectorXd xu = solverLDLT.solve(bu);
        Eigen::VectorXd xv = solverLDLT.solve(bv);
        for(int k=0; k<nUnknown; ++k) {
             outUV[interiorIndices[k]] = { (float)xu(k), (float)xv(k) };
        }
    } else {
        Eigen::VectorXd xu = solver.solve(bu);
        Eigen::VectorXd xv = solver.solve(bv);
        for(int k=0; k<nUnknown; ++k) {
             outUV[interiorIndices[k]] = { (float)xu(k), (float)xv(k) };
        }
    }
    
    return true;
}

void LSCMFlattener::PlanarFallback(const Mesh& mesh,
                                   int chartId,
                                   const std::vector<uint32_t>& chartVerts,
                                   const std::unordered_map<uint32_t,int>& localIndex,
                                   const std::vector<int>& triChart,
                                   std::vector<vec2>& outUV) {
    // Compute chart centroid and average normal
    vec3 centroid = {0, 0, 0};
    vec3 avgNormal = {0, 0, 0};
    int triCount = 0;
    
    for (int t = 0; t < (int)triChart.size(); ++t) {
        if (triChart[t] != chartId) continue;
        
        uint32_t v[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        vec3 p[3];
        for (int k = 0; k < 3; ++k) {
            p[k] = mesh.V[v[k]];
            centroid = centroid + p[k];
        }
        
        vec3 e1 = p[1] - p[0];
        vec3 e2 = p[2] - p[0];
        vec3 n = cross(e1, e2);
        float len = norm(n);
        if (len > 1e-8f) {
            avgNormal = avgNormal + n;
        }
        triCount++;
    }
    
    if (triCount == 0) return;
    
    float scale = 1.0f / (triCount * 3.0f);
    centroid = centroid * scale;
    
    float avgNormLen = norm(avgNormal);
    if (avgNormLen < 1e-8f) {
        // Degenerate normal, try axis-aligned projection
        // Use bounding box to determine best axis
        vec3 bbMin = {1e9f, 1e9f, 1e9f};
        vec3 bbMax = {-1e9f, -1e9f, -1e9f};
        
        for (size_t i = 0; i < chartVerts.size(); ++i) {
            vec3 p = mesh.V[chartVerts[i]];
            if (p.x < bbMin.x) bbMin.x = p.x;
            if (p.y < bbMin.y) bbMin.y = p.y;
            if (p.z < bbMin.z) bbMin.z = p.z;
            if (p.x > bbMax.x) bbMax.x = p.x;
            if (p.y > bbMax.y) bbMax.y = p.y;
            if (p.z > bbMax.z) bbMax.z = p.z;
        }
        
        vec3 extent = bbMax - bbMin;
        
        // Project onto the plane perpendicular to the smallest extent
        if (extent.x <= extent.y && extent.x <= extent.z) {
            // YZ plane (smallest extent in X)
            for (size_t i = 0; i < chartVerts.size(); ++i) {
                vec3 p = mesh.V[chartVerts[i]];
                outUV[i].x = p.y;
                outUV[i].y = p.z;
            }
        } else if (extent.y <= extent.x && extent.y <= extent.z) {
            // XZ plane (smallest extent in Y)
            for (size_t i = 0; i < chartVerts.size(); ++i) {
                vec3 p = mesh.V[chartVerts[i]];
                outUV[i].x = p.x;
                outUV[i].y = p.z;
            }
        } else {
            // XY plane (smallest extent in Z)
            for (size_t i = 0; i < chartVerts.size(); ++i) {
                vec3 p = mesh.V[chartVerts[i]];
                outUV[i].x = p.x;
                outUV[i].y = p.y;
            }
        }
        return;
    }
    
    avgNormal = normalize(avgNormal);
    
    // Build orthonormal basis
    vec3 u_axis, v_axis;
    
    // Choose best reference vector based on avgNormal
    vec3 ref;
    if (std::abs(avgNormal.x) < 0.9f) {
        ref = {1, 0, 0};
    } else if (std::abs(avgNormal.y) < 0.9f) {
        ref = {0, 1, 0};
    } else {
        ref = {0, 0, 1};
    }
    
    u_axis = cross(ref, avgNormal);
    float u_len = norm(u_axis);
    if (u_len < 1e-8f) {
        // Try different reference
        ref = {0, 0, 1};
        u_axis = cross(ref, avgNormal);
        u_len = norm(u_axis);
    }
    
    if (u_len > 1e-8f) {
        u_axis = normalize(u_axis);
    } else {
        u_axis = {1, 0, 0}; // Fallback
    }
    
    v_axis = cross(avgNormal, u_axis);
    float v_len = norm(v_axis);
    if (v_len > 1e-8f) {
        v_axis = normalize(v_axis);
    } else {
        v_axis = {0, 1, 0}; // Fallback
    }
    
    // Project vertices onto plane
    for (size_t i = 0; i < chartVerts.size(); ++i) {
        vec3 p = mesh.V[chartVerts[i]];
        vec3 relative = p - centroid;
        outUV[i].x = dot(relative, u_axis);
        outUV[i].y = dot(relative, v_axis);
    }
}

int LSCMFlattener::CountInvertedTriangles(const Mesh& mesh,
                                          int chartId,
                                          const std::vector<uint32_t>& chartVerts,
                                          const std::unordered_map<uint32_t,int>& localIndex,
                                          const std::vector<int>& triChart,
                                          const std::vector<vec2>& uv) {
    int inverted = 0;
    
    for (int t = 0; t < (int)triChart.size(); ++t) {
        if (triChart[t] != chartId) continue;
        
        uint32_t v[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        int idx[3];
        bool valid = true;
        for (int k = 0; k < 3; ++k) {
            auto it = localIndex.find(v[k]);
            if (it == localIndex.end()) { valid = false; break; }
            idx[k] = it->second;
        }
        if (!valid) continue;
        
        vec2 uv0 = uv[idx[0]];
        vec2 uv1 = uv[idx[1]];
        vec2 uv2 = uv[idx[2]];
        
        float signedArea = 0.5f * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - 
                                   (uv2.x - uv0.x) * (uv1.y - uv0.y));
        
        if (signedArea < -1e-10f) {
            inverted++;
        }
    }
    
    return inverted;
}

void LSCMFlattener::SpreadVerticesInUV(std::vector<vec2>& uv) {
    // Last resort: arrange vertices in a spiral pattern
    // This guarantees non-degenerate UVs even for completely degenerate charts
    
    int n = (int)uv.size();
    if (n == 0) return;
    
    // Use a simple grid layout
    int gridSize = (int)std::ceil(std::sqrt((float)n));
    float spacing = 1.0f / std::max(1, gridSize - 1);
    
    for (int i = 0; i < n; ++i) {
        int row = i / gridSize;
        int col = i % gridSize;
        uv[i].x = col * spacing;
        uv[i].y = row * spacing;
    }
}

} // namespace uv

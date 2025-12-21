#include "flatten_harmonic.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

namespace uv {

std::vector<ChartUV> HarmonicFlattener::Flatten(const Mesh& mesh,
                                                const std::vector<int>& triChart,
                                                int chartCount,
                                                const std::vector<ChartBoundary>& boundaries,
                                                bool useCotanWeights) {
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
            // No valid boundary loop? Can't do harmonic parameterization without boundary conditions.
            // UVs remain 0.0f
            continue;
        }
        
        MapBoundaryToCircle(mesh, loop, localIndex, uvBoundary, isBoundary);
        
        SolveHarmonic(mesh, c, res.chartVerts, localIndex, triChart, 
                      isBoundary, uvBoundary, useCotanWeights, res.uv);
        
        res.minUV = {1e9, 1e9};
        res.maxUV = {-1e9, -1e9};
        for (const auto& uv : res.uv) {
            if (uv.x < res.minUV.x) res.minUV.x = uv.x;
            if (uv.y < res.minUV.y) res.minUV.y = uv.y;
            if (uv.x > res.maxUV.x) res.maxUV.x = uv.x;
            if (uv.y > res.maxUV.y) res.maxUV.y = uv.y;
        }
    }
    
    return results;
}

void HarmonicFlattener::BuildLocalIndex(const Mesh& mesh,
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

void HarmonicFlattener::MapBoundaryToCircle(const Mesh& mesh,
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

void HarmonicFlattener::SolveHarmonic(const Mesh& mesh,
                                      int chartId,
                                      const std::vector<uint32_t>& chartVerts,
                                      const std::unordered_map<uint32_t, int>& localIndex,
                                      const std::vector<int>& triChart,
                                      const std::vector<uint8_t>& isBoundary,
                                      const std::vector<vec2>& uvBoundary,
                                      bool useCotanWeights,
                                      std::vector<vec2>& outUV) {
    int n = (int)chartVerts.size();
    // outUV is already resized by caller
    
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
    if (nUnknown == 0) return; // All boundary?
    
    Eigen::VectorXd bu = Eigen::VectorXd::Zero(nUnknown);
    Eigen::VectorXd bv = Eigen::VectorXd::Zero(nUnknown);
    std::vector<Triplet> triplets;
    
    for (int t = 0; t < (int)triChart.size(); ++t) {
        if (triChart[t] != chartId) continue;
        
        uint32_t v[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };
        int idx[3];
        for (int k=0; k<3; ++k) idx[k] = localIndex.at(v[k]);
        
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
             return;
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
}

} // namespace uv

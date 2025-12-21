#include "uv_validation.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace uv {

ChartQualityMetrics ValidateChart(const Mesh& mesh,
                                   const ChartUV& chartUV,
                                   const std::vector<int>& triChart,
                                   int chartId) {
    ChartQualityMetrics metrics;
    metrics.chartId = chartId;
    
    // Build local vertex index map
    std::unordered_map<uint32_t, int> vertexToLocal;
    for (size_t i = 0; i < chartUV.chartVerts.size(); ++i) {
        vertexToLocal[chartUV.chartVerts[i]] = (int)i;
    }
    
    int triangleCount = 0;
    int zeroAreaCount = 0;
    int invertedCount = 0;
    
    // Check each triangle in this chart
    for (size_t t = 0; t < triChart.size(); ++t) {
        if (triChart[t] != chartId) continue;
        
        triangleCount++;
        
        // Get triangle vertices
        uint32_t v0 = mesh.F[t].x;
        uint32_t v1 = mesh.F[t].y;
        uint32_t v2 = mesh.F[t].z;
        
        // Get local indices
        auto it0 = vertexToLocal.find(v0);
        auto it1 = vertexToLocal.find(v1);
        auto it2 = vertexToLocal.find(v2);
        
        if (it0 == vertexToLocal.end() || it1 == vertexToLocal.end() || it2 == vertexToLocal.end()) {
            continue; // Vertex not in chart (shouldn't happen)
        }
        
        int idx0 = it0->second;
        int idx1 = it1->second;
        int idx2 = it2->second;
        
        // Get UVs
        vec2 uv0 = chartUV.uv[idx0];
        vec2 uv1 = chartUV.uv[idx1];
        vec2 uv2 = chartUV.uv[idx2];
        
        // Compute UV triangle area
        float uvArea = 0.5f * std::abs(
            (uv1.x - uv0.x) * (uv2.y - uv0.y) - 
            (uv2.x - uv0.x) * (uv1.y - uv0.y)
        );
        
        if (uvArea < 1e-10f) {
            zeroAreaCount++;
        }
        
        // Check if inverted (negative area)
        float signedArea = 0.5f * (
            (uv1.x - uv0.x) * (uv2.y - uv0.y) - 
            (uv2.x - uv0.x) * (uv1.y - uv0.y)
        );
        
        if (signedArea < -1e-10f) {
            invertedCount++;
        }
    }
    
    metrics.triangleCount = triangleCount;
    metrics.zeroAreaTriangles = zeroAreaCount;
    metrics.invertedTriangles = invertedCount;
    
    // Chart is invalid if it has significant problems
    if (triangleCount == 0 || 
        (float)zeroAreaCount / triangleCount > 0.1f || // More than 10% zero-area
        invertedCount > 0) {
        metrics.isValid = false;
    }
    
    return metrics;
}

void ValidateUnwrapResult(const Mesh& mesh,
                          const std::vector<ChartUV>& chartUVs,
                          const std::vector<int>& triChart) {
    std::cout << "\n=== UV Unwrap Validation ===" << std::endl;
    
    int totalZeroArea = 0;
    int totalInverted = 0;
    int totalTriangles = 0;
    int invalidCharts = 0;
    
    std::vector<ChartQualityMetrics> worstCharts;
    
    for (const auto& chartUV : chartUVs) {
        ChartQualityMetrics metrics = ValidateChart(mesh, chartUV, triChart, chartUV.chartId);
        
        totalZeroArea += metrics.zeroAreaTriangles;
        totalInverted += metrics.invertedTriangles;
        totalTriangles += metrics.triangleCount;
        
        if (!metrics.isValid) {
            invalidCharts++;
        }
        
        // Track worst charts for reporting
        if (metrics.zeroAreaTriangles > 10 || metrics.invertedTriangles > 0) {
            worstCharts.push_back(metrics);
        }
    }
    
    std::cout << "Total triangles: " << totalTriangles << std::endl;
    std::cout << "Zero-area triangles: " << totalZeroArea 
              << " (" << (100.0f * totalZeroArea / std::max(1, totalTriangles)) << "%)" << std::endl;
    std::cout << "Inverted triangles: " << totalInverted << std::endl;
    std::cout << "Invalid charts: " << invalidCharts << " / " << chartUVs.size() << std::endl;
    
    if (!worstCharts.empty()) {
        std::cout << "\nCharts with most issues:" << std::endl;
        
        // Sort by zero-area count
        std::sort(worstCharts.begin(), worstCharts.end(), 
                  [](const ChartQualityMetrics& a, const ChartQualityMetrics& b) {
                      return a.zeroAreaTriangles > b.zeroAreaTriangles;
                  });
        
        // Show top 10 worst
        int showCount = std::min(10, (int)worstCharts.size());
        for (int i = 0; i < showCount; ++i) {
            const auto& m = worstCharts[i];
            std::cout << "  Chart " << m.chartId << ": " 
                      << m.zeroAreaTriangles << " zero-area, "
                      << m.invertedTriangles << " inverted (of " 
                      << m.triangleCount << " triangles)" << std::endl;
        }
    }
    
    std::cout << "============================\n" << std::endl;
}

} // namespace uv


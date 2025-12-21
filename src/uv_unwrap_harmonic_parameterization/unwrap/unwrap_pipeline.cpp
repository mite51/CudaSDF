#include "unwrap_pipeline.h"
#include "chart_builder.h"
#include "boundary_loops.h"
#include "flatten_lscm.h"
#include "atlas_packer.h"
#include "uv_validation.h"
#include <iostream>

namespace uv {

UnwrapResult UnwrapPipeline::Run(const Mesh& mesh, const UnwrapConfig& cfg) {
    std::cout << "Starting UV Unwrap..." << std::endl;
    
    MeshDerived derived = ComputeMeshDerived(mesh);
    std::cout << "Computed derived data." << std::endl;
    
    TriAdjacency adj = BuildTriangleAdjacency(mesh);
    std::cout << "Built adjacency." << std::endl;
    
    ChartBuilder chartBuilder;
    Charts charts = chartBuilder.Build(mesh, derived, adj, cfg);
    std::cout << "Built " << charts.chartCount << " charts." << std::endl;
    
    BoundaryLoops boundaryLoops;
    std::vector<ChartBoundary> boundaries = boundaryLoops.Extract(mesh, adj, charts.triChart, charts.chartCount);
    std::cout << "Extracted boundaries." << std::endl;
    
    LSCMFlattener flattener;
    std::vector<ChartUV> chartUVs = flattener.Flatten(mesh, charts.triChart, charts.chartCount, boundaries);
    std::cout << "Flattened charts (LSCM)." << std::endl;
    
    // Validate UV quality
    ValidateUnwrapResult(mesh, chartUVs, charts.triChart);
    
    UnwrapResult result;
    result.triChart = charts.triChart;
    result.chartCount = charts.chartCount;
    
    AtlasPacker packer;
    packer.PackAndAssignUVs(chartUVs, cfg, mesh, charts.triChart, result);
    std::cout << "Packed atlas (" << result.atlasW << "x" << result.atlasH << ")." << std::endl;
    
    return result;
}

} // namespace uv

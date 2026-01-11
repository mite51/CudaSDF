#pragma once
#include "Commons.cuh"
#include "CudaSDFUtil.h"
#include <vector>

enum class MeshExtractionTechnique : int {
    MarchingCubes = 0,
    DualContouring = 1
};

class CudaSDFMesh {
public:
    CudaSDFMesh();
    ~CudaSDFMesh();

    void Initialize(float cellSize, float3 boundsMin, float3 boundsMax);
    
    // Helper to add a primitive to the list managed by this mesh
    void AddPrimitive(const SDFPrimitive& prim);
    void ClearPrimitives();
    
    // Direct access to modify primitives (for animation)
    std::vector<SDFPrimitive>& GetPrimitives() { return m_primitives; }
    
    // Update internal structures (BVH) and generate mesh on GPU
    // d_outVertices, d_outColors, d_outIndices must be valid device pointers
    // d_outUVs, d_outPrimitiveIDs, and d_outNormals are optional (can be nullptr)
    void Update(float time, float4* d_outVertices, float4* d_outColors, unsigned int* d_outIndices,
                float2* d_outUVs = nullptr, int* d_outPrimitiveIDs = nullptr, float4* d_outNormals = nullptr,
                MeshExtractionTechnique technique = MeshExtractionTechnique::MarchingCubes,
                float dcNormalSmoothAngleDeg = 30.0f,
                float dcQefBlend = 1.0f);

    // Getters for Renderer to update its buffers
    const std::vector<BVHNode>& GetBVHNodes() const { return m_bvhNodes; }
    const std::vector<GPUPrimitive>& GetGPUPrimitives() const { return m_gpuPrimitives; }
    
    // Get number of indices generated in the last Update
    unsigned int GetTotalIndexCount() const { return m_totalIndices; }
    unsigned int GetTotalVertexCount() const { return m_totalVertices; }
    
    // Get the list of indices that form the BVH (Union vs Non-Union split)
    // This is needed for the Fragment Shader to know where the BVH ends
    unsigned int GetNumBVHPrimitives() const { return m_numBVHPrimitives; }

    // Memory Statistics
    struct MemStats {
        size_t usedGPU;
        size_t freeGPU;
        size_t totalGPU;
        int activeBlocks;
        int allocatedBlocks;
    };
    
    MemStats GetMemoryStats() const;

    // Get mesh bounds (for debug visualization)
    void GetBounds(float3& outMin, float3& outMax) const {
        outMin = h_grid.origin;
        outMax = make_float3(
            h_grid.origin.x + h_grid.width * h_grid.cellSize,
            h_grid.origin.y + h_grid.height * h_grid.cellSize,
            h_grid.origin.z + h_grid.depth * h_grid.cellSize
        );
    }

private:
    SDFGrid h_grid;
    SDFGrid d_grid;

    int m_packedSize = 0; // estimatedActiveBlocks * SDF_BLOCK_SIZE_CUBED (matches allocations)
    
    // Temp storage for Scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Intermediate buffers on GPU (now managed inside SDFGrid)
    
    unsigned int* d_totalVertsPtr = nullptr;
    unsigned int* d_totalIndicesPtr = nullptr;

    // Scene Data
    std::vector<SDFPrimitive> m_primitives;
    std::vector<BVHNode> m_bvhNodes;
    std::vector<GPUPrimitive> m_gpuPrimitives;
    
    unsigned int m_totalIndices = 0;
    unsigned int m_totalVertices = 0;
    unsigned int m_numBVHPrimitives = 0;
    
    // Internal tracking of allocated sizes
    size_t m_allocatedPrimBytes = 0;
    size_t m_allocatedBVHBytes = 0;
    size_t m_allocatedGridBytes = 0;
    
    // Helper to sort and build BVH
    void UpdateBVH();
};

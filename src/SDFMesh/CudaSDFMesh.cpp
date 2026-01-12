#include "CudaSDFMesh.h"
#include <algorithm>
#include <iostream> // for checkCudaErrors
#include <cmath>

// Re-definition of checkCudaErrors inside CudaSDFMesh.cpp as it is not included from main
inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// Externs from Wrappers.cu
extern "C" void launchScoutActiveBlocks(SDFGrid& grid, float time);
extern "C" void launchCountActiveBlockTriangles(SDFGrid& grid, int numActiveBlocks, float time);
extern "C" void launchGenerateActiveBlockTriangles(SDFGrid& grid, int numActiveBlocks, float time);

// Dual Contouring wrappers
extern "C" void launchBuildBlockToActiveMap(SDFGrid& grid, int numActiveBlocks);
extern "C" void launchDCMarkCells(SDFGrid& grid, int numActiveBlocks, float time);
extern "C" void launchDCSolveCellVertices(SDFGrid& grid, int numActiveBlocks, float time, unsigned int maxCellVertices, float qefBlend);
extern "C" void launchDCRecomputeNormalsAtVertices(SDFGrid& grid, int numActiveBlocks, float time);
extern "C" void launchDCSmoothCellNormals(SDFGrid& grid, int numActiveBlocks, float cosAngleThreshold);
extern "C" void launchDCCountQuads(SDFGrid& grid, int numActiveBlocks);
extern "C" void launchDCGenerateQuads(SDFGrid& grid, int numActiveBlocks, float time);

extern "C" void getScanStorageSize(unsigned int num_items, size_t* temp_storage_bytes);
extern "C" void launchScan(unsigned int* d_input, unsigned int* d_output, unsigned int numItems, void* d_temp_storage, size_t temp_storage_bytes);

CudaSDFMesh::CudaSDFMesh() {
    d_grid.d_activeBlocks = nullptr;
    d_grid.d_activeBlockCount = nullptr;
    d_grid.d_packetVertexCounts = nullptr;
    d_grid.d_packetVertexOffsets = nullptr;

    // Dual Contouring lazy buffers
    d_grid.d_blockToActiveId = nullptr;
    d_grid.d_dcCornerMasks = nullptr;
    d_grid.d_dcCellVertexOffsets = nullptr;
    d_grid.d_dcCellVertices = nullptr;
    d_grid.d_dcCellNormals = nullptr;
    d_grid.d_dcCellNormalsTmp = nullptr;
    // Initialize other pointers to nullptr
}

CudaSDFMesh::~CudaSDFMesh() {
    // Cleanup
    if (d_grid.d_activeBlocks) cudaFree(d_grid.d_activeBlocks);
    if (d_grid.d_activeBlockCount) cudaFree(d_grid.d_activeBlockCount);
    if (d_grid.d_packetVertexCounts) cudaFree(d_grid.d_packetVertexCounts);
    if (d_grid.d_packetVertexOffsets) cudaFree(d_grid.d_packetVertexOffsets);

    if (d_grid.d_blockToActiveId) cudaFree(d_grid.d_blockToActiveId);
    if (d_grid.d_dcCornerMasks) cudaFree(d_grid.d_dcCornerMasks);
    if (d_grid.d_dcCellVertexOffsets) cudaFree(d_grid.d_dcCellVertexOffsets);
    if (d_grid.d_dcCellVertices) cudaFree(d_grid.d_dcCellVertices);
    if (d_grid.d_dcCellNormals) cudaFree(d_grid.d_dcCellNormals);
    if (d_grid.d_dcCellNormalsTmp) cudaFree(d_grid.d_dcCellNormalsTmp);
    // if (d_grid.d_colors) cudaFree(d_grid.d_colors); // Removed
    if (d_grid.d_primitives) cudaFree(d_grid.d_primitives);
    if (d_grid.d_bvhNodes) cudaFree(d_grid.d_bvhNodes);
    
    if (d_totalVertsPtr) cudaFree(d_totalVertsPtr);
    if (d_totalIndicesPtr) cudaFree(d_totalIndicesPtr);
    if (d_temp_storage) cudaFree(d_temp_storage);
}

void CudaSDFMesh::Initialize(float cellSize, float3 boundsMin, float3 boundsMax) {
    if (cellSize <= 0.0f) {
        std::cerr << "Error: Invalid cell size (must be > 0)" << std::endl;
        exit(-1);
    }
    h_grid.cellSize = cellSize;
    h_grid.origin = boundsMin;
    
    float3 size = make_float3(
        boundsMax.x - boundsMin.x,
        boundsMax.y - boundsMin.y,
        boundsMax.z - boundsMin.z
    );
    
    // Basic validation
    if (size.x <= 0.0f || size.y <= 0.0f || size.z <= 0.0f) {
        std::cerr << "Error: Invalid bounds size (must be > 0)" << std::endl;
        exit(-1);
    }
    
    // Calculate dimensions
    h_grid.width = (unsigned int)ceilf(size.x / h_grid.cellSize);
    h_grid.height = (unsigned int)ceilf(size.y / h_grid.cellSize);
    h_grid.depth = (unsigned int)ceilf(size.z / h_grid.cellSize);
    
    // Check for unreasonable size (> 2048^3 equivalent or similar huge number)
    // Let's set a soft limit of ~8 billion voxels (2048^3) just as a sanity check for 32-bit int indexing in kernels if any
    unsigned long long totalVoxels = (unsigned long long)h_grid.width * h_grid.height * h_grid.depth;
    if (totalVoxels > 8589934592ULL) {
         std::cerr << "Error: Grid size too large (" << totalVoxels << " voxels). Increase cell size or reduce bounds." << std::endl;
         exit(-1);
    }
    
    // Block Config
    int bs = SDF_BLOCK_SIZE;
    h_grid.blocksDim = make_int3(
        (h_grid.width + bs - 1) / bs,
        (h_grid.height + bs - 1) / bs,
        (h_grid.depth + bs - 1) / bs
    );
    
    // Allocation Estimation
    // Max possible blocks is blocksDim.x*y*z
    h_grid.maxBlocks = h_grid.blocksDim.x * h_grid.blocksDim.y * h_grid.blocksDim.z; 
    
    // Reallocate if already allocated (rudimentary support for re-init)
    if(d_grid.d_activeBlocks) cudaFree(d_grid.d_activeBlocks);
    if(d_grid.d_activeBlockCount) cudaFree(d_grid.d_activeBlockCount);
    if(d_grid.d_packetVertexCounts) cudaFree(d_grid.d_packetVertexCounts);
    if(d_grid.d_packetVertexOffsets) cudaFree(d_grid.d_packetVertexOffsets);
    
    m_allocatedGridBytes = 0;
    
    checkCudaErrors(cudaMalloc(&d_grid.d_activeBlocks, h_grid.maxBlocks * sizeof(int)));
    m_allocatedGridBytes += h_grid.maxBlocks * sizeof(int);
    
    checkCudaErrors(cudaMalloc(&d_grid.d_activeBlockCount, sizeof(int)));
    m_allocatedGridBytes += sizeof(int);
    
    // Packed Voxel Buffers
    // Assume max 10% active blocks for dense data or cap at sensible limit
    int estimatedActiveBlocks = std::min(h_grid.maxBlocks, 200000); 
    int packedSize = estimatedActiveBlocks * SDF_BLOCK_SIZE_CUBED;
    m_packedSize = packedSize;
    
    checkCudaErrors(cudaMalloc(&d_grid.d_packetVertexCounts, packedSize * sizeof(unsigned int)));
    m_allocatedGridBytes += packedSize * sizeof(unsigned int);
    
    checkCudaErrors(cudaMalloc(&d_grid.d_packetVertexOffsets, packedSize * sizeof(unsigned int)));
    m_allocatedGridBytes += packedSize * sizeof(unsigned int);
    
    // CUB Temp Storage
    if(d_temp_storage) cudaFree(d_temp_storage);
    getScanStorageSize(packedSize, &temp_storage_bytes);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    m_allocatedGridBytes += temp_storage_bytes;
    
    if(!d_totalVertsPtr) {
        checkCudaErrors(cudaMalloc(&d_totalVertsPtr, sizeof(unsigned int)));
        m_allocatedGridBytes += sizeof(unsigned int);
    }
    if(!d_totalIndicesPtr) {
        checkCudaErrors(cudaMalloc(&d_totalIndicesPtr, sizeof(unsigned int)));
        m_allocatedGridBytes += sizeof(unsigned int);
    }
    
    // Upper bound estimation for GL buffers (clamped)
    // The user of this class manages the actual GL buffers, we just need to report max safely.
    // We use a large constant in main.cpp anyway.
    h_grid.maxVertices = 50000000; 
    h_grid.maxIndices = h_grid.maxVertices * 3;

    // Copy params
    d_grid.width = h_grid.width;
    d_grid.height = h_grid.height;
    d_grid.depth = h_grid.depth;
    d_grid.cellSize = h_grid.cellSize;
    d_grid.origin = h_grid.origin;
    d_grid.blocksDim = h_grid.blocksDim;
    d_grid.maxBlocks = h_grid.maxBlocks;
    d_grid.maxVertices = h_grid.maxVertices;

    // Dual Contouring buffers are lazy-allocated (only if technique is used)
    // Keep pointers nullptr here.
    
    h_grid.numPrimitives = 0;
    d_grid.d_primitives = nullptr;
    d_grid.d_bvhNodes = nullptr;
}

void CudaSDFMesh::AddPrimitive(const SDFPrimitive& prim) {
    m_primitives.push_back(prim);
}

void CudaSDFMesh::ClearPrimitives() {
    m_primitives.clear();
}

void CudaSDFMesh::UpdateBVH() {
    // Separate Unions and Others
    std::vector<int> unionIndices;
    std::vector<int> otherIndices;
    for(int i=0; i<(int)m_primitives.size(); ++i) {
        if(m_primitives[i].operation == SDF_UNION || m_primitives[i].operation == SDF_UNION_BLEND) {
            unionIndices.push_back(i);
        } else {
            otherIndices.push_back(i);
        }
    }
    
    // Reorder primitives (Unions first, then Others)
    std::vector<SDFPrimitive> sortedPrimitives;
    sortedPrimitives.reserve(m_primitives.size());
    for(int idx : unionIndices) sortedPrimitives.push_back(m_primitives[idx]);
    for(int idx : otherIndices) sortedPrimitives.push_back(m_primitives[idx]);
    
    // Update internal list with sorted version (important for consistency)
    m_primitives = sortedPrimitives; 
    
    // Build BVH for Union primitives
    m_bvhNodes.clear();
    if (!unionIndices.empty()) {
        std::vector<int> bvhBuildIndices(unionIndices.size());
        for(int i=0; i<(int)unionIndices.size(); ++i) bvhBuildIndices[i] = i; // 0 to N-1 in sorted list
        buildBVHRecursive(bvhBuildIndices.data(), bvhBuildIndices.size(), m_primitives, m_bvhNodes);
    }
    
    m_numBVHPrimitives = unionIndices.size();
    
    // Pack for GPU (Renderer consumption)
    packPrimitives(m_primitives, m_gpuPrimitives);
}

CudaSDFMesh::MemStats CudaSDFMesh::GetMemoryStats() const {
    MemStats stats;
    
    // Calculate application-specific memory usage
    size_t totalAllocated = m_allocatedGridBytes + m_allocatedPrimBytes + m_allocatedBVHBytes;
    
    // Calculate "used" as the actual data size within the allocated buffers
    // For the fixed grid buffers, we count the full allocation as used (reserved)
    size_t used = m_allocatedGridBytes;
    used += m_primitives.size() * sizeof(SDFPrimitive);
    used += m_bvhNodes.size() * sizeof(BVHNode);
    
    stats.totalGPU = totalAllocated;
    stats.usedGPU = used;
    stats.freeGPU = (totalAllocated > used) ? (totalAllocated - used) : 0;
    
    // Read back active blocks count (asynchronous if called mid-frame, but okay for stats)
    int numActiveBlocks = 0;
    if (d_grid.d_activeBlockCount) {
        cudaMemcpy(&numActiveBlocks, d_grid.d_activeBlockCount, sizeof(int), cudaMemcpyDeviceToHost);
    }
    stats.activeBlocks = numActiveBlocks;
    
    // Allocated blocks is maxBlocks (since we allocate the full pointer list)
    // But the dense voxel data is capped.
    stats.allocatedBlocks = std::min(d_grid.maxBlocks, 200000); // The number of blocks we have dense storage for
    
    return stats;
}

void CudaSDFMesh::Update(float time, float4* d_outVertices, float4* d_outColors, unsigned int* d_outIndices,
                         float2* d_outUVs, int* d_outPrimitiveIDs, float4* d_outNormals,
                         MeshExtractionTechnique technique, float dcNormalSmoothAngleDeg,
                         float dcQefBlend) {
    UpdateBVH();
    
    // Upload Primitives
    h_grid.numPrimitives = m_primitives.size();
    h_grid.numBVHPrimitives = m_numBVHPrimitives;
    d_grid.numPrimitives = h_grid.numPrimitives;
    d_grid.numBVHPrimitives = h_grid.numBVHPrimitives;
    
    static size_t primAllocSize = 0;
    if (primAllocSize < m_primitives.size() * sizeof(SDFPrimitive)) {
        if (d_grid.d_primitives) cudaFree(d_grid.d_primitives);
        primAllocSize = std::max(m_primitives.size() * sizeof(SDFPrimitive) * 2, (size_t)1024);
        checkCudaErrors(cudaMalloc(&d_grid.d_primitives, primAllocSize));
        m_allocatedPrimBytes = primAllocSize;
    }
    checkCudaErrors(cudaMemcpy(d_grid.d_primitives, m_primitives.data(), m_primitives.size() * sizeof(SDFPrimitive), cudaMemcpyHostToDevice));
    
    // Upload BVH
    h_grid.numBVHNodes = m_bvhNodes.size();
    d_grid.numBVHNodes = h_grid.numBVHNodes;
    
    static size_t bvhAllocSize = 0;
    if (bvhAllocSize < m_bvhNodes.size() * sizeof(BVHNode)) {
        if (d_grid.d_bvhNodes) cudaFree(d_grid.d_bvhNodes);
        bvhAllocSize = std::max(m_bvhNodes.size() * sizeof(BVHNode) * 2, (size_t)1024);
        checkCudaErrors(cudaMalloc(&d_grid.d_bvhNodes, bvhAllocSize));
        m_allocatedBVHBytes = bvhAllocSize;
    }
    checkCudaErrors(cudaMemcpy(d_grid.d_bvhNodes, m_bvhNodes.data(), m_bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
    
    // Assign Output Pointers
    d_grid.d_vertices = d_outVertices;
    d_grid.d_vertexColors = d_outColors;
    d_grid.d_indices = d_outIndices;
    d_grid.d_uvCoords = d_outUVs;              // NEW
    d_grid.d_primitiveIDs = d_outPrimitiveIDs; // NEW
    d_grid.d_normals = d_outNormals;           // NEW
    d_grid.d_totalVertices = d_totalVertsPtr;
    d_grid.d_totalIndices = d_totalIndicesPtr;
    
    // 1. Scout Active Blocks
    checkCudaErrors(cudaMemset(d_grid.d_activeBlockCount, 0, sizeof(int)));
    launchScoutActiveBlocks(d_grid, time);
    
    int numActiveBlocks = 0;
    checkCudaErrors(cudaMemcpy(&numActiveBlocks, d_grid.d_activeBlockCount, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (numActiveBlocks <= 0) {
        m_totalVertices = 0;
        m_totalIndices = 0;
        return;
    }

    // Cap to buffer size
    int maxSafeBlocks = 200000; // Must match Initialize estimation
    if (numActiveBlocks > maxSafeBlocks) {
        numActiveBlocks = maxSafeBlocks;
    }

    const int totalItems = numActiveBlocks * SDF_BLOCK_SIZE_CUBED;

    if (technique == MeshExtractionTechnique::MarchingCubes) {
        // 2. Count Triangles per Active Voxel
        launchCountActiveBlockTriangles(d_grid, numActiveBlocks, time);

        // 3. Scan
        launchScan(d_grid.d_packetVertexCounts, d_grid.d_packetVertexOffsets, totalItems, d_temp_storage, temp_storage_bytes);

        // 4. Generate Triangles (Soup)
        launchGenerateActiveBlockTriangles(d_grid, numActiveBlocks, time);

    } else {
        // --------------------------
        // Dual Contouring (Option A)
        // --------------------------
        // Lazy allocate DC buffers
        if (!d_grid.d_blockToActiveId) {
            checkCudaErrors(cudaMalloc(&d_grid.d_blockToActiveId, h_grid.maxBlocks * sizeof(int)));
            m_allocatedGridBytes += h_grid.maxBlocks * sizeof(int);
        }
        if (!d_grid.d_dcCornerMasks) {
            checkCudaErrors(cudaMalloc(&d_grid.d_dcCornerMasks, m_packedSize * sizeof(unsigned char)));
            m_allocatedGridBytes += m_packedSize * sizeof(unsigned char);
        }
        if (!d_grid.d_dcCellVertexOffsets) {
            checkCudaErrors(cudaMalloc(&d_grid.d_dcCellVertexOffsets, m_packedSize * sizeof(unsigned int)));
            m_allocatedGridBytes += m_packedSize * sizeof(unsigned int);
        }
        if (!d_grid.d_dcCellVertices) {
            // Store compacted cell vertices up to maxVertices entries (8 bytes each)
            checkCudaErrors(cudaMalloc(&d_grid.d_dcCellVertices, h_grid.maxVertices * sizeof(ushort4)));
            m_allocatedGridBytes += h_grid.maxVertices * sizeof(ushort4);
        }
        if (!d_grid.d_dcCellNormals) {
            // Store compacted cell normals up to maxVertices entries (8 bytes each)
            checkCudaErrors(cudaMalloc(&d_grid.d_dcCellNormals, h_grid.maxVertices * sizeof(short4)));
            m_allocatedGridBytes += h_grid.maxVertices * sizeof(short4);
        }
        if (!d_grid.d_dcCellNormalsTmp) {
            checkCudaErrors(cudaMalloc(&d_grid.d_dcCellNormalsTmp, h_grid.maxVertices * sizeof(short4)));
            m_allocatedGridBytes += h_grid.maxVertices * sizeof(short4);
        }

        // Build block -> activeBlockId mapping (needed for cross-block quad assembly)
        checkCudaErrors(cudaMemset(d_grid.d_blockToActiveId, 0xFF, h_grid.maxBlocks * sizeof(int))); // -1
        launchBuildBlockToActiveMap(d_grid, numActiveBlocks);

        // Pass 1: mark cells with crossings + store corner masks; counts = hasVertex (0/1)
        launchDCMarkCells(d_grid, numActiveBlocks, time);

        // Scan 1: hasVertex (counts) -> dcCellVertexOffsets (compact cell-vertex indices)
        launchScan(d_grid.d_packetVertexCounts, d_grid.d_dcCellVertexOffsets, totalItems, d_temp_storage, temp_storage_bytes);

        // Pass 2: solve cell vertices into compact buffer
        launchDCSolveCellVertices(d_grid, numActiveBlocks, time, h_grid.maxVertices, dcQefBlend);

        // NOTE: Per-cell normal recomputation and smoothing are no longer needed.
        // Normals are now computed per-triangle-vertex in dcGenerateQuads using the triangle's
        // face normal to offset the sampling position. This gives better normals at shared edges
        // because each triangle uses its own face direction for the offset.

        // Pass 3: count quads (as 2 tris = 6 soup vertices) into packetVertexCounts
        launchDCCountQuads(d_grid, numActiveBlocks);

        // Scan 2: triangle soup offsets
        launchScan(d_grid.d_packetVertexCounts, d_grid.d_packetVertexOffsets, totalItems, d_temp_storage, temp_storage_bytes);

        // Pass 4: generate soup vertices/UV/IDs/normals
        launchDCGenerateQuads(d_grid, numActiveBlocks, time);
    }

    // Readback Total Count
    unsigned int lastOffset, lastCount;
    checkCudaErrors(cudaMemcpy(&lastOffset, &d_grid.d_packetVertexOffsets[totalItems - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&lastCount, &d_grid.d_packetVertexCounts[totalItems - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    m_totalVertices = lastOffset + lastCount;
    m_totalIndices = 0; // No indices generated, just vertex soup
}

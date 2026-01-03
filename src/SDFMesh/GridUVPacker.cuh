#pragma once
#include "Commons.cuh"
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// Grid UV Packer - CUDA Implementation
// ============================================================================
// Based on grid-uv-packer algorithm, ported to CUDA for efficient packing
// of irregularly-shaped UV islands into a texture atlas.

namespace GridUVPacker {

// ----------------------------------------------------------------------------
// Data Structures
// ----------------------------------------------------------------------------

// Grid cell coordinate (2D)
struct CellCoord {
    int x;
    int y;
    
    __host__ __device__ CellCoord() : x(0), y(0) {}
    __host__ __device__ CellCoord(int x_, int y_) : x(x_), y(y_) {}
    
    __host__ __device__ CellCoord offset(int dx, int dy) const {
        return CellCoord(x + dx, y + dy);
    }
    
    __host__ __device__ static CellCoord zero() {
        return CellCoord(0, 0);
    }
};

// Bit-packed grid for efficient storage and collision detection
struct Grid {
    uint32_t* cells;  // Bit-packed: 32 cells per uint32
    int width;        // Grid dimensions in cells
    int height;
    int stride;       // Width in uint32 blocks = (width + 31) / 32
    
    __host__ Grid() : cells(nullptr), width(0), height(0), stride(0) {}
    
    __host__ Grid(int w, int h) : width(w), height(h) {
        stride = (width + 31) / 32;
        cudaMalloc(&cells, stride * height * sizeof(uint32_t));
        cudaMemset(cells, 0, stride * height * sizeof(uint32_t));
    }
    
    __host__ void free() {
        if (cells) {
            cudaFree(cells);
            cells = nullptr;
        }
    }
    
    __host__ __device__ int linearIndex(int x, int y) const {
        return y * width + x;
    }
    
    __host__ __device__ int totalCells() const {
        return stride * height * 32;  // Total bits available
    }
};

// Helper functions for bit manipulation (declarations only - implemented in .cu file)
__device__ inline bool getGridCell(const Grid& grid, int x, int y);
__device__ inline void setGridCell(Grid& grid, int x, int y, bool value);

// UV Island representation
struct Island {
    // Grid masks (device pointers)
    Grid mask;              // Shape mask
    Grid collisionMask;     // Dilated mask with margin
    Grid mask90;            // 90° CW rotated shape mask
    Grid collisionMask90;   // 90° CW rotated collision mask

    // Per-row active word bounds for collision masks (inclusive).
    // If a row is empty, minWord==0xFFFF and maxWord==0.
    uint16_t* d_collMinWord = nullptr;
    uint16_t* d_collMaxWord = nullptr;
    uint16_t* d_collMinWord90 = nullptr;
    uint16_t* d_collMaxWord90 = nullptr;
    
    // Original UV bounds in primitive space
    float2 uvMin;
    float2 uvMax;

    // Grid-space origin (cell coords) for this island's mask.
    // The island mask is rasterized in coordinates:
    //   cell = uv * gridResolution - minCell
    int2 minCell;
    
    // Triangle data for this island
    int numTriangles;
    uint32_t* d_triangleIndices;  // Device pointer to triangle indices
    
    // Metadata
    int primitiveID;
    int islandIndex;  // Unique index for this island
    
    __host__ Island() : numTriangles(0), d_triangleIndices(nullptr),
                       primitiveID(-1), islandIndex(-1) {
        uvMin = make_float2(0, 0);
        uvMax = make_float2(1, 1);
        minCell = make_int2(0, 0);
    }
};

// Island placement result
struct IslandPlacement {
    int islandIndex;  // Which island
    int gridX;        // Position in atlas grid
    int gridY;
    int rotation;     // 0=0°, 1=90°, 2=180°, 3=270°
    
    __host__ __device__ IslandPlacement() 
        : islandIndex(-1), gridX(0), gridY(0), rotation(0) {}
};

// Packing solution
struct Solution {
    IslandPlacement* placements;  // Device array of placements
    int numPlaced;                // Number of successfully placed islands
    Grid occupancyMask;           // Combined occupancy grid
    float fitness;                // Packing efficiency [0-1]
    int seed;                     // Random seed used
    int utilizedWidth;            // Actual used dimensions
    int utilizedHeight;
    
    __host__ Solution() : placements(nullptr), numPlaced(0), 
                         fitness(0.0f), seed(0), 
                         utilizedWidth(0), utilizedHeight(0) {}
};

// Configuration for packing
struct PackingConfig {
    int gridSize;           // Board size in cells (square board: gridSize x gridSize)
    int gridResolution;     // Cells per UV unit for quantization (e.g., 64–1024)
    int maxIterations;      // Number of solution attempts
    int marginCells;        // Margin in grid cells
    bool enableRotation;    // Allow 90° rotations
    int xAlignment;         // Placement X alignment in cells (v1 simplification)
    int maxCandidatesPerIsland; // Candidate X positions to evaluate per island (per attempt)
    int maxDropSteps;           // Cap on downward “drop” steps (0 = no cap)
    bool shuffleOrderPerAttempt; // Randomize island order per attempt (improves exploration)
    int randomSeed;         // Base random seed (0 = random)
    
    __host__ PackingConfig() 
        : gridSize(128), gridResolution(64), maxIterations(500), marginCells(2),
          enableRotation(true), xAlignment(8), maxCandidatesPerIsland(32), maxDropSteps(0),
          shuffleOrderPerAttempt(true),
          randomSeed(0) {}
};

// Chart extracted from marching cubes output
struct UVChart {
    int primitiveID;
    std::vector<uint32_t> triangleIndices;  // Triangle indices in mesh
    float2 uvMin;                           // Bounding box in primitive UV
    float2 uvMax;
    
    UVChart() : primitiveID(-1) {
        uvMin = make_float2(1e10f, 1e10f);
        uvMax = make_float2(-1e10f, -1e10f);
    }
};

// Final packed result (host-accessible)
struct PackedAtlas {
    std::vector<IslandPlacement> placements;
    float scalingFactor;     // Scale from grid space to [0,1]
    int atlasWidth;          // Final atlas dimensions
    int atlasHeight;
    float fitness;           // Packing quality
    bool success;            // Whether packing succeeded
    
    PackedAtlas() : scalingFactor(1.0f), atlasWidth(0), atlasHeight(0),
                   fitness(0.0f), success(false) {}
};

// ----------------------------------------------------------------------------
// Core API Functions
// ----------------------------------------------------------------------------

// Extract UV charts from marching cubes output
std::vector<UVChart> ExtractCharts(
    const float4* d_vertices,
    const float2* d_uvCoords,
    const int* d_primitiveIDs,  // NOTE: These are stored as floats, need conversion
    int numVertices
);

// Convert charts to islands (rasterize onto grid)
std::vector<Island> CreateIslands(
    const std::vector<UVChart>& charts,
    const float2* d_uvCoords,
    int gridResolution,
    int marginCells
);

// Main packing function
PackedAtlas PackIslands(
    const std::vector<Island>& islands,
    const PackingConfig& config
);

// Remap UVs to final atlas space
void RemapUVsToAtlas(
    float2* d_uvCoords,
    const int* d_primitiveIDs,  // NOTE: Download as float*, convert to int
    int numVertices,
    const std::vector<UVChart>& charts,
    const std::vector<Island>& islands,
    const PackedAtlas& atlas,
    int gridResolution
);

// Cleanup island resources
void FreeIslands(std::vector<Island>& islands);

// ----------------------------------------------------------------------------
// CUDA Kernel Declarations
// ----------------------------------------------------------------------------

// Rasterize triangles onto grid
__global__ void rasterizeIslandKernel(
    const float2* uvCoords,
    const uint32_t* triIndices,
    int numTriangles,
    int2 minCell,
    int gridResolution,
    Grid grid
);

// Dilate grid for margins
__global__ void dilateGridKernel(
    const Grid input,
    Grid output,
    int radius
);

// Rotate grid
__global__ void rotateGridKernel(
    const Grid input,
    Grid output,
    int rotation  // 0=0°, 1=90°, 2=180°, 3=270°
);

// Check collision between island and occupancy mask
__device__ bool checkIslandCollision(
    const Grid& occupancyMask,
    const Grid& islandMask,
    int offsetX,
    int offsetY,
    int rotation
);

// Write island to occupancy mask
__device__ void writeIslandToMask(
    Grid& occupancyMask,
    const Grid& islandMask,
    int offsetX,
    int offsetY,
    int rotation
);

// Pack single solution (called per thread block)
__global__ void packSolutionKernel(
    Solution* solutions,
    const Island* islands,
    int numIslands,
    int gridSize,
    int* seeds,
    bool enableRotation,
    int targetShelfWidth,
    int xAlignment,
    int maxCandidatesPerIsland,
    int maxDropSteps,
    bool shuffleOrderPerAttempt
);

// Remap UVs to atlas
__global__ void remapUVsKernel(
    float2* uvCoords,
    const float* primitiveIDs,  // Input as float
    int numVertices,
    const IslandPlacement* placementsByChart, // [numCharts]
    const int* primToChart,                  // [primToChartSize], maps primitiveID -> chartIdx
    int primToChartSize,
    const int2* chartMinCells,               // [numCharts]
    const int2* chartDims,                   // [numCharts] (rot0 width,height in cells)
    int numCharts,
    int gridResolution,
    int utilizedWidth,
    int utilizedHeight
);

// ----------------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------------

// Count active cells in grid
__global__ void countActiveCellsKernel(
    const Grid grid,
    int* count
);

// Calculate fitness (packing efficiency)
__device__ float calculateFitness(
    const Grid& mask,
    int utilizedWidth,
    int utilizedHeight
);

// Copy grid (device function)
__device__ void copyGrid(
    const Grid& src,
    Grid& dst
);

// Combine two grids with OR operation
__global__ void combineGridsKernel(
    Grid dest,
    const Grid src
);

} // namespace GridUVPacker


#include "GridUVPacker.cuh"
#include <curand_kernel.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <cmath>

namespace GridUVPacker {

// ============================================================================
// Device Function Implementations
// ============================================================================

__device__ inline bool getGridCell(const Grid& grid, int x, int y) {
    if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
        return false;
    }
    // IMPORTANT: honor per-row stride (padding to 32-bit words).
    // Using a tightly-packed linear index corrupts rows when width is not a multiple of 32.
    int blockIdx = y * grid.stride + (x >> 5); // x/32
    int bitIdx = x & 31;                       // x%32
    return (grid.cells[blockIdx] >> bitIdx) & 1;
}

__device__ inline void setGridCell(Grid& grid, int x, int y, bool value) {
    if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
        return;
    }
    int blockIdx = y * grid.stride + (x >> 5); // x/32
    int bitIdx = x & 31;                       // x%32
    
    if (value) {
        atomicOr(&grid.cells[blockIdx], 1u << bitIdx);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

__device__ inline float triangleArea2D(float2 a, float2 b, float2 c) {
    return 0.5f * fabsf((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y));
}

__device__ __forceinline__ int alignUpInt(int v, int a) {
    if (a <= 1) return v;
    int r = v % a;
    return (r == 0) ? v : (v + (a - r));
}

__device__ __forceinline__ int alignDownInt(int v, int a) {
    if (a <= 1) return v;
    return v - (v % a);
}

__device__ inline bool pointInTriangle(float2 p, float2 a, float2 b, float2 c) {
    float totalArea = triangleArea2D(a, b, c);
    float area1 = triangleArea2D(p, b, c);
    float area2 = triangleArea2D(a, p, c);
    float area3 = triangleArea2D(a, b, p);
    
    float epsilon = 1e-6f;
    return fabsf(totalArea - (area1 + area2 + area3)) < epsilon;
}

__device__ __forceinline__ float cross2(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

__device__ __forceinline__ float2 sub2(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ bool onSegment(float2 a, float2 b, float2 p) {
    // p colinear with ab assumed
    float minX = fminf(a.x, b.x) - 1e-6f;
    float maxX = fmaxf(a.x, b.x) + 1e-6f;
    float minY = fminf(a.y, b.y) - 1e-6f;
    float maxY = fmaxf(a.y, b.y) + 1e-6f;
    return (p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY);
}

__device__ __forceinline__ int orientSign(float2 a, float2 b, float2 c) {
    float v = cross2(sub2(b, a), sub2(c, a));
    const float eps = 1e-6f;
    if (v > eps) return 1;
    if (v < -eps) return -1;
    return 0;
}

__device__ __forceinline__ bool segmentsIntersect(float2 a, float2 b, float2 c, float2 d) {
    int o1 = orientSign(a, b, c);
    int o2 = orientSign(a, b, d);
    int o3 = orientSign(c, d, a);
    int o4 = orientSign(c, d, b);

    if (o1 != o2 && o3 != o4) return true;
    // Colinear / touching cases
    if (o1 == 0 && onSegment(a, b, c)) return true;
    if (o2 == 0 && onSegment(a, b, d)) return true;
    if (o3 == 0 && onSegment(c, d, a)) return true;
    if (o4 == 0 && onSegment(c, d, b)) return true;
    return false;
}

__device__ inline bool triangleOverlapsCell(
    float2 v0, float2 v1, float2 v2,
    int cellX, int cellY,
    float /*cellSizeIgnored*/
) {
    // Cell corners in *cell space* (cell size = 1).
    float minX = (float)cellX;
    float minY = (float)cellY;
    float maxX = (float)(cellX + 1);
    float maxY = (float)(cellY + 1);
    
    float2 corners[4] = {
        make_float2(minX, minY),
        make_float2(maxX, minY),
        make_float2(maxX, maxY),
        make_float2(minX, maxY)
    };
    
    // Check if any triangle vertex is inside cell
    if ((v0.x >= minX && v0.x <= maxX && v0.y >= minY && v0.y <= maxY) ||
        (v1.x >= minX && v1.x <= maxX && v1.y >= minY && v1.y <= maxY) ||
        (v2.x >= minX && v2.x <= maxX && v2.y >= minY && v2.y <= maxY)) {
        return true;
    }
    
    // Check if any cell corner is inside triangle
    for (int i = 0; i < 4; i++) {
        if (pointInTriangle(corners[i], v0, v1, v2)) {
            return true;
        }
    }

    // Check if any triangle edge intersects any cell edge.
    // This catches thin slivers where neither vertices nor corners are inside.
    float2 cellEdgesA[4] = { corners[0], corners[1], corners[2], corners[3] };
    float2 cellEdgesB[4] = { corners[1], corners[2], corners[3], corners[0] };

    float2 triA[3] = { v0, v1, v2 };
    float2 triB[3] = { v1, v2, v0 };

    #pragma unroll
    for (int e = 0; e < 3; e++) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            if (segmentsIntersect(triA[e], triB[e], cellEdgesA[r], cellEdgesB[r])) {
                return true;
            }
        }
    }
    
    // Check if triangle center is in cell (quick approximation)
    float2 center = make_float2((v0.x + v1.x + v2.x) / 3.0f,
                                 (v0.y + v1.y + v2.y) / 3.0f);
    if (center.x >= minX && center.x <= maxX && 
        center.y >= minY && center.y <= maxY) {
        return true;
    }
    
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void rasterizeIslandKernel(
    const float2* uvCoords,
    const uint32_t* triIndices,
    int numTriangles,
    int2 minCell,
    int gridResolution,
    Grid grid
) {
    int triIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (triIdx >= numTriangles) return;
    
    // Load triangle UVs
    uint32_t idx0 = triIndices[triIdx * 3 + 0];
    uint32_t idx1 = triIndices[triIdx * 3 + 1];
    uint32_t idx2 = triIndices[triIdx * 3 + 2];
    
    float2 uv0 = uvCoords[idx0];
    float2 uv1 = uvCoords[idx1];
    float2 uv2 = uvCoords[idx2];
    
    // Convert to local *cell space*:
    //   cell = uv * gridResolution - minCell
    float2 p0 = make_float2(uv0.x * gridResolution - (float)minCell.x,
                            uv0.y * gridResolution - (float)minCell.y);
    float2 p1 = make_float2(uv1.x * gridResolution - (float)minCell.x,
                            uv1.y * gridResolution - (float)minCell.y);
    float2 p2 = make_float2(uv2.x * gridResolution - (float)minCell.x,
                            uv2.y * gridResolution - (float)minCell.y);
    
    // Compute triangle bounding box in grid space
    float minU = fminf(fminf(p0.x, p1.x), p2.x);
    float maxU = fmaxf(fmaxf(p0.x, p1.x), p2.x);
    float minV = fminf(fminf(p0.y, p1.y), p2.y);
    float maxV = fmaxf(fmaxf(p0.y, p1.y), p2.y);
    
    int xMin = (int)floorf(minU);
    int xMax = (int)ceilf(maxU);
    int yMin = (int)floorf(minV);
    int yMax = (int)ceilf(maxV);
    
    // Clamp to grid bounds
    xMin = max(0, min(xMin, grid.width - 1));
    xMax = max(0, min(xMax, grid.width));
    yMin = max(0, min(yMin, grid.height - 1));
    yMax = max(0, min(yMax, grid.height));
    
    // Rasterize cells
    for (int y = yMin; y < yMax; y++) {
        for (int x = xMin; x < xMax; x++) {
            if (triangleOverlapsCell(p0, p1, p2, x, y, 1.0f)) {
                setGridCell(grid, x, y, true);
            }
        }
    }
}

__global__ void dilateGridKernel(
    const Grid input,
    Grid output,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output.width || y >= output.height) return;
    
    bool hasNeighbor = false;
    
    // Check circular neighborhood
    for (int dy = -radius; dy <= radius && !hasNeighbor; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy > radius*radius) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (getGridCell(input, nx, ny)) {
                hasNeighbor = true;
                break;
            }
        }
    }
    
    if (hasNeighbor) {
        setGridCell(output, x, y, true);
    }
}

__global__ void rotateGridKernel(
    const Grid input,
    Grid output,
    int rotation
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output.width || y >= output.height) return;
    
    int srcX, srcY;
    switch (rotation) {
        case 0: // 0°
            srcX = x;
            srcY = y;
            break;
        case 1: // 90° CW
            // output(w=input.height, h=input.width): (xo,yo) <- (xi=yo, yi=input.height-1-xo)
            srcX = y;
            srcY = input.height - 1 - x;
            break;
        case 2: // 180°
            srcX = input.width - 1 - x;
            srcY = input.height - 1 - y;
            break;
        case 3: // 270° CW
            // output(w=input.height, h=input.width): (xo,yo) <- (xi=input.width-1-yo, yi=xo)
            srcX = input.width - 1 - y;
            srcY = x;
            break;
        default:
            srcX = x;
            srcY = y;
    }
    
    bool value = getGridCell(input, srcX, srcY);
    if (value) {
        setGridCell(output, x, y, true);
    }
}

__global__ void computeRowWordBoundsKernel(
    const Grid mask,
    uint16_t* outMinWord,
    uint16_t* outMaxWord
) {
    int row = blockIdx.x;
    if (row >= mask.height) return;

    int stride = mask.stride;
    int tid = threadIdx.x;

    // Initialize to "empty"
    __shared__ int sMin;
    __shared__ int sMax;
    if (tid == 0) {
        sMin = stride;
        sMax = -1;
    }
    __syncthreads();

    // Each thread scans multiple words
    for (int w = tid; w < stride; w += blockDim.x) {
        uint32_t v = mask.cells[row * stride + w];
        if (v != 0) {
            atomicMin(&sMin, w);
            atomicMax(&sMax, w);
        }
    }
    __syncthreads();

    if (tid == 0) {
        if (sMax < 0) {
            outMinWord[row] = 0xFFFF;
            outMaxWord[row] = 0;
        } else {
            outMinWord[row] = (uint16_t)sMin;
            outMaxWord[row] = (uint16_t)sMax;
        }
    }
}

__device__ __forceinline__ bool checkIslandCollisionBitsAlignedRanged(
    const Grid& occupancyMask,
    const Grid& islandMask,
    const uint16_t* rowMinWord,
    const uint16_t* rowMaxWord,
    int offsetX,
    int offsetY
) {
    if (offsetX < 0 || offsetY < 0) return true;
    if (offsetX + islandMask.width > occupancyMask.width) return true;
    if (offsetY + islandMask.height > occupancyMask.height) return true;

    int wordShift = offsetX >> 5;
    int bitShift = offsetX & 31;

    // For each island row, test only active word range against board row.
    for (int y = 0; y < islandMask.height; y++) {
        uint16_t minW = rowMinWord ? rowMinWord[y] : 0;
        if (rowMinWord && minW == 0xFFFF) continue; // empty row
        uint16_t maxW = rowMaxWord ? rowMaxWord[y] : (uint16_t)(islandMask.stride - 1);

        const uint32_t* islRow = &islandMask.cells[y * islandMask.stride + minW];
        const uint32_t* brdRow = &occupancyMask.cells[(offsetY + y) * occupancyMask.stride + wordShift + minW];
        int wCount = (int)maxW - (int)minW + 1;

        if (bitShift == 0) {
            for (int w = 0; w < wCount; w++) {
                if ((brdRow[w] & islRow[w]) != 0) return true;
            }
        } else {
            uint32_t carry = 0;
            for (int w = 0; w < wCount; w++) {
                uint32_t cur = islRow[w];
                uint32_t shifted = (cur << bitShift) | carry;
                carry = cur >> (32 - bitShift);
                if ((brdRow[w] & shifted) != 0) return true;
            }
            // Final carry may spill into the next word (board has space due to bounds check above)
            if (carry != 0) {
                if ((wordShift + minW + wCount) < occupancyMask.stride) {
                    if ((brdRow[wCount] & carry) != 0) return true;
                }
            }
        }
    }

    return false;
}

__device__ __forceinline__ void writeIslandToMaskBitsAlignedRanged(
    Grid& occupancyMask,
    const Grid& islandMask,
    const uint16_t* rowMinWord,
    const uint16_t* rowMaxWord,
    int offsetX,
    int offsetY
) {
    int wordShift = offsetX >> 5;
    int bitShift = offsetX & 31;

    for (int y = 0; y < islandMask.height; y++) {
        uint16_t minW = rowMinWord ? rowMinWord[y] : 0;
        if (rowMinWord && minW == 0xFFFF) continue; // empty row
        uint16_t maxW = rowMaxWord ? rowMaxWord[y] : (uint16_t)(islandMask.stride - 1);

        const uint32_t* islRow = &islandMask.cells[y * islandMask.stride + minW];
        uint32_t* brdRow = &occupancyMask.cells[(offsetY + y) * occupancyMask.stride + wordShift + minW];
        int wCount = (int)maxW - (int)minW + 1;

        if (bitShift == 0) {
            for (int w = 0; w < wCount; w++) {
                brdRow[w] |= islRow[w];
            }
        } else {
            uint32_t carry = 0;
            for (int w = 0; w < wCount; w++) {
                uint32_t cur = islRow[w];
                uint32_t shifted = (cur << bitShift) | carry;
                carry = cur >> (32 - bitShift);
                brdRow[w] |= shifted;
            }
            if (carry != 0) {
                if ((wordShift + minW + wCount) < occupancyMask.stride) {
                    brdRow[wCount] |= carry;
                }
            }
        }
    }
}

__device__ bool checkIslandCollision(
    const Grid& occupancyMask,
    const Grid& islandMask,
    int offsetX,
    int offsetY,
    int rotation
) {
    // Get island dimensions based on rotation
    int islandW = (rotation == 1 || rotation == 3) ? islandMask.height : islandMask.width;
    int islandH = (rotation == 1 || rotation == 3) ? islandMask.width : islandMask.height;
    
    // Bounds check
    if (offsetX < 0 || offsetY < 0) return true;
    if (offsetX + islandW > occupancyMask.width) return true;
    if (offsetY + islandH > occupancyMask.height) return true;
    
    // Check for overlaps
    for (int y = 0; y < islandMask.height; y++) {
        for (int x = 0; x < islandMask.width; x++) {
            if (!getGridCell(islandMask, x, y)) continue;
            
            // Apply rotation transform
            int gridX, gridY;
            switch (rotation) {
                case 0:
                    gridX = offsetX + x;
                    gridY = offsetY + y;
                    break;
                case 1: // 90° CW
                    gridX = offsetX + y;
                    gridY = offsetY + (islandMask.width - 1 - x);
                    break;
                case 2: // 180°
                    gridX = offsetX + (islandMask.width - 1 - x);
                    gridY = offsetY + (islandMask.height - 1 - y);
                    break;
                case 3: // 270° CW
                    gridX = offsetX + (islandMask.height - 1 - y);
                    gridY = offsetY + x;
                    break;
                default:
                    gridX = offsetX + x;
                    gridY = offsetY + y;
            }
            
            if (getGridCell(occupancyMask, gridX, gridY)) {
                return true;  // Collision detected
            }
        }
    }
    
    return false;  // No collision
}

__device__ void writeIslandToMask(
    Grid& occupancyMask,
    const Grid& islandMask,
    int offsetX,
    int offsetY,
    int rotation
) {
    for (int y = 0; y < islandMask.height; y++) {
        for (int x = 0; x < islandMask.width; x++) {
            if (!getGridCell(islandMask, x, y)) continue;
            
            // Apply rotation transform
            int gridX, gridY;
            switch (rotation) {
                case 0:
                    gridX = offsetX + x;
                    gridY = offsetY + y;
                    break;
                case 1: // 90° CW
                    gridX = offsetX + y;
                    gridY = offsetY + (islandMask.width - 1 - x);
                    break;
                case 2: // 180°
                    gridX = offsetX + (islandMask.width - 1 - x);
                    gridY = offsetY + (islandMask.height - 1 - y);
                    break;
                case 3: // 270° CW
                    gridX = offsetX + (islandMask.height - 1 - y);
                    gridY = offsetY + x;
                    break;
                default:
                    gridX = offsetX + x;
                    gridY = offsetY + y;
            }
            
            setGridCell(occupancyMask, gridX, gridY, true);
        }
    }
}

__global__ void countActiveCellsKernel(
    const Grid grid,
    int* count
) {
    __shared__ int localCount[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    localCount[tid] = 0;
    __syncthreads();
    
    // Count bits in this thread's uint32 blocks
    int totalBlocks = grid.stride * grid.height;
    if (idx < totalBlocks) {
        localCount[tid] = __popc(grid.cells[idx]);
    }
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            localCount[tid] += localCount[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(count, localCount[0]);
    }
}

__device__ float calculateFitness(
    const Grid& mask,
    int utilizedWidth,
    int utilizedHeight
) {
    // Would need to count active cells, but for device function
    // we'll compute it externally
    int size = max(utilizedWidth, utilizedHeight);
    if (size == 0) return 0.0f;
    
    // Placeholder - actual counting done by kernel
    return 1.0f;
}

__global__ void combineGridsKernel(
    Grid dest,
    const Grid src
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalBlocks = dest.stride * dest.height;
    
    if (idx < totalBlocks && idx < src.stride * src.height) {
        atomicOr(&dest.cells[idx], src.cells[idx]);
    }
}

// Improved packing kernel with 2D layout (not just horizontal strip)
__global__ void packSolutionKernel(
    Solution* solutions,
    const Island* islands,
    int numIslands,
    int gridSize,
    int* seeds,
    bool enableRotation,
    int targetShelfWidth,  // pass optimal shelf width
    int xAlignment,        // aligned-X placement constraint
    int maxCandidatesPerIsland,
    int maxDropSteps,
    bool shuffleOrderPerAttempt
) {
    int solutionIdx = blockIdx.x;
    Solution& solution = solutions[solutionIdx];
    
    // Initialize RNG
    curandState rng;
    curand_init(seeds[solutionIdx], 0, 0, &rng);
    
    // Only thread 0 in each block does the packing
    if (threadIdx.x != 0) return;
    
    solution.numPlaced = 0;
    solution.utilizedWidth = 0;
    solution.utilizedHeight = 0;
    
    // Enforce aligned-X placement (v1 constraint).
    const int X_ALIGNMENT = max(1, xAlignment);

    // Use the passed shelf width only as a soft horizontal cap; never exceed the board.
    int shelfMaxWidth = min(gridSize, alignDownInt(max(1, targetShelfWidth), X_ALIGNMENT));

    // Clamp candidate count to a small fixed budget (thread0 does all the work).
    const int MAX_CAND = 64;
    int candBudget = max(4, min(MAX_CAND, maxCandidatesPerIsland));
    int dropCap = maxDropSteps; // 0 = no cap
    
    // Build per-attempt island order.
    const int MAX_ISLANDS_LOCAL = 256;
    int order[MAX_ISLANDS_LOCAL];
    int n = min(numIslands, MAX_ISLANDS_LOCAL);
    for (int i = 0; i < n; i++) order[i] = i;

    if (shuffleOrderPerAttempt && n > 1) {
        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--) {
            int j = (int)(curand(&rng) % (unsigned)(i + 1));
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }

    // Try to place each island with a small stochastic candidate search.
    for (int oi = 0; oi < n; oi++) {
        const Island& island = islands[order[oi]];
        bool placed = false;
        
        // v1: only allow 0° or 90°
        int maxRotations = enableRotation ? 2 : 1;
        int rotOrder = (enableRotation ? (curand(&rng) & 1) : 0); // randomize rotation order per island/attempt

        // Track best candidate across rotations + X candidates
        float bestScore = 1e30f;
        int bestX = -1, bestY = -1, bestRot = 0;
        int bestActualW = 0, bestActualH = 0;

        for (int rr = 0; rr < maxRotations; rr++) {
            int rotation = (rr ^ rotOrder); // 0/1
            
            // Select precomputed collision mask for rotation (rot0 / rot90)
            const Grid& coll = (rotation == 1) ? island.collisionMask90 : island.collisionMask;

            int islandW = coll.width;
            int islandH = coll.height;
            
            // Bounds check
            if (islandW > gridSize || islandH > gridSize) {
                continue;
            }

            // Actual dims == collision mask dims (includes gutter/margin already)
            int actualW = islandW;
            int actualH = islandH;

            // Candidate X generation (aligned):
            // Always include a few deterministic anchors, plus random aligned columns.
            int maxX = min(shelfMaxWidth, gridSize) - islandW;
            if (maxX < 0) continue;

            int cCount = 0;
            int candX[MAX_CAND];

            auto pushCand = [&](int x) {
                if (cCount >= candBudget) return;
                x = max(0, min(maxX, x));
                x = alignDownInt(x, X_ALIGNMENT);
                // de-dup small list
                for (int k = 0; k < cCount; k++) if (candX[k] == x) return;
                candX[cCount++] = x;
            };

            pushCand(0);
            pushCand(alignDownInt(solution.utilizedWidth - islandW, X_ALIGNMENT));
            pushCand(alignDownInt(maxX, X_ALIGNMENT));

            while (cCount < candBudget) {
                // random aligned x in [0, maxX]
                unsigned r = curand(&rng);
                int x = (int)(r % (unsigned)(maxX + 1));
                pushCand(x);
            }

            // Evaluate candidates
            for (int ci = 0; ci < cCount; ci++) {
                int x = candX[ci];

                // Start at current top and "drop" down while still collision-free.
                int y = min(solution.utilizedHeight, gridSize - islandH);
                if (y < 0) continue;

                // Ensure initial is collision-free (should be, but guard anyway)
                while (y <= gridSize - islandH && checkIslandCollisionBitsAlignedRanged(solution.occupancyMask, coll,
                                                                                         (rotation == 1) ? island.d_collMinWord90 : island.d_collMinWord,
                                                                                         (rotation == 1) ? island.d_collMaxWord90 : island.d_collMaxWord,
                                                                                         x, y)) {
                    y++;
                }
                if (y > gridSize - islandH) continue; // no fit at this X

                int steps = 0;
                while (y > 0) {
                    if (dropCap > 0 && steps >= dropCap) break;
                    if (checkIslandCollisionBitsAlignedRanged(solution.occupancyMask, coll,
                                                              (rotation == 1) ? island.d_collMinWord90 : island.d_collMinWord,
                                                              (rotation == 1) ? island.d_collMaxWord90 : island.d_collMaxWord,
                                                              x, y - 1)) break;
                    y--;
                    steps++;
                }

                int newW = max(solution.utilizedWidth, x + actualW);
                int newH = max(solution.utilizedHeight, y + actualH);
                int maxDim = max(newW, newH);
                float area = (float)(newW * newH);

                // Simple v1 score: minimize area, slight bias for squareness (smaller maxDim)
                float score = area + 0.01f * (float)(maxDim * maxDim);

                if (score < bestScore) {
                    bestScore = score;
                    bestX = x; bestY = y; bestRot = rotation;
                    bestActualW = actualW; bestActualH = actualH;
                }
            }
        }

        if (bestX >= 0) {
            // Commit best candidate
            const Grid& coll = (bestRot == 1) ? island.collisionMask90 : island.collisionMask;
            writeIslandToMaskBitsAlignedRanged(solution.occupancyMask, coll,
                                               (bestRot == 1) ? island.d_collMinWord90 : island.d_collMinWord,
                                               (bestRot == 1) ? island.d_collMaxWord90 : island.d_collMaxWord,
                                               bestX, bestY);

            IslandPlacement& placement = solution.placements[solution.numPlaced];
            placement.islandIndex = island.islandIndex;
            placement.gridX = bestX;
            placement.gridY = bestY;
            placement.rotation = bestRot;

            solution.numPlaced++;
            solution.utilizedWidth = max(solution.utilizedWidth, bestX + bestActualW);
            solution.utilizedHeight = max(solution.utilizedHeight, bestY + bestActualH);
            placed = true;
        }

        if (!placed) break; // failed to place this island
    }
    
    // Calculate fitness based on utilized area
    if (solution.numPlaced > 0 && solution.utilizedWidth > 0 && solution.utilizedHeight > 0) {
        float placementRatio = (float)solution.numPlaced / (float)numIslands;
        int maxDim = max(solution.utilizedWidth, solution.utilizedHeight);
        float area = (float)(solution.utilizedWidth * solution.utilizedHeight);
        float squareness = area / (float)(maxDim * maxDim);
        
        solution.fitness = placementRatio * squareness;
    } else {
        solution.fitness = 0.0f;
    }
}

__global__ void remapUVsKernel(
    float2* uvCoords,
    const float* primitiveIDs,  // Input as float
    int numVertices,
    const IslandPlacement* placementsByChart, // [numCharts]
    const int* primToChart,                  // [primToChartSize]
    int primToChartSize,
    const int2* chartMinCells,               // [numCharts]
    const int2* chartDims,                   // [numCharts]
    int numCharts,
    int gridResolution,
    int utilizedWidth,           // Actual utilized width in cells
    int utilizedHeight           // Actual utilized height in cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    
    // Convert float primitive ID to int
    int primID = (int)(primitiveIDs[idx] + 0.5f);  // Round to nearest int
    if (primID < 0) return;
    
    int chartIdx = (primID >= 0 && primID < primToChartSize) ? primToChart[primID] : -1;
    if (chartIdx < 0) return;

    const IslandPlacement placement = placementsByChart[chartIdx];
    if (placement.islandIndex < 0) return;

    const int2 minCell = chartMinCells[chartIdx];
    const int2 dims0 = chartDims[chartIdx];

    // Convert UV to island-local cell space
    const float2 uv = uvCoords[idx];
    float2 cell = make_float2(uv.x * gridResolution - (float)minCell.x,
                              uv.y * gridResolution - (float)minCell.y);

    // Apply v1 rotation (0° or 90° CW)
    if (placement.rotation == 1) {
        // (x,y) -> (y, W-1-x)
        cell = make_float2(cell.y, (float)(dims0.x - 1) - cell.x);
    }

    float2 atlasCell = make_float2((float)placement.gridX + cell.x,
                                   (float)placement.gridY + cell.y);

    float denom = (float)max(utilizedWidth, utilizedHeight);
    if (denom <= 0.0f) return;

    uvCoords[idx] = make_float2(atlasCell.x / denom, atlasCell.y / denom);
}

// ============================================================================
// Host API Implementation
// ============================================================================

std::vector<UVChart> ExtractCharts(
    const float4* d_vertices,
    const float2* d_uvCoords,
    const int* d_primitiveIDs,
    int numVertices
) {
    // Download data from GPU
    std::vector<float4> vertices(numVertices);
    std::vector<float2> uvCoords(numVertices);
    std::vector<float> primitiveIDsFloat(numVertices);  // Download as float!
    
    cudaMemcpy(vertices.data(), d_vertices, numVertices * sizeof(float4), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(uvCoords.data(), d_uvCoords, numVertices * sizeof(float2), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(primitiveIDsFloat.data(), d_primitiveIDs, numVertices * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Convert float primitive IDs to int
    std::vector<int> primitiveIDs(numVertices);
    for (int i = 0; i < numVertices; i++) {
        primitiveIDs[i] = (int)(primitiveIDsFloat[i] + 0.5f);  // Round to nearest
    }
    
    // Group triangles by primitive ID
    std::map<int, std::vector<uint32_t>> primToTris;
    
    for (int i = 0; i < numVertices; i += 3) {
        int primID = primitiveIDs[i];
        if (primID >= 0) {
            uint32_t triIdx = i / 3;
            primToTris[primID].push_back(triIdx);
        }
    }
    
    // Create charts
    std::vector<UVChart> charts;
    for (auto& [primID, tris] : primToTris) {
        UVChart chart;
        chart.primitiveID = primID;
        chart.triangleIndices = tris;
        
        // Compute UV bounds
        chart.uvMin = make_float2(1e10f, 1e10f);
        chart.uvMax = make_float2(-1e10f, -1e10f);
        
        for (uint32_t triIdx : tris) {
            for (int v = 0; v < 3; v++) {
                float2 uv = uvCoords[triIdx * 3 + v];
                chart.uvMin.x = fminf(chart.uvMin.x, uv.x);
                chart.uvMin.y = fminf(chart.uvMin.y, uv.y);
                chart.uvMax.x = fmaxf(chart.uvMax.x, uv.x);
                chart.uvMax.y = fmaxf(chart.uvMax.y, uv.y);
            }
        }
        
        charts.push_back(chart);
    }
    
    std::cout << "Extracted " << charts.size() << " UV charts from " 
              << numVertices / 3 << " triangles" << std::endl;
    
    return charts;
}

std::vector<Island> CreateIslands(
    const std::vector<UVChart>& charts,
    const float2* d_uvCoords,
    int gridResolution,
    int marginCells
) {
    std::vector<Island> islands;
    islands.reserve(charts.size());
    
    // Create islands from charts
    for (size_t i = 0; i < charts.size(); i++) {
        const UVChart& chart = charts[i];
        
        Island island;
        island.primitiveID = chart.primitiveID;
        island.islandIndex = (int)i;
        island.uvMin = chart.uvMin;
        island.uvMax = chart.uvMax;
        island.numTriangles = (int)chart.triangleIndices.size();

        // Spec-aligned: global quantization grid.
        // Convert UV bounds -> cell bounds at fixed gridResolution, then add gutter/margin in cell units.
        int minCellX = (int)floorf(chart.uvMin.x * (float)gridResolution) - marginCells;
        int minCellY = (int)floorf(chart.uvMin.y * (float)gridResolution) - marginCells;
        int maxCellX = (int)ceilf(chart.uvMax.x * (float)gridResolution) + marginCells;
        int maxCellY = (int)ceilf(chart.uvMax.y * (float)gridResolution) + marginCells;

        int islandWidth = max(1, maxCellX - minCellX);
        int islandHeight = max(1, maxCellY - minCellY);

        island.minCell = make_int2(minCellX, minCellY);

        float2 uvRange = make_float2(chart.uvMax.x - chart.uvMin.x, chart.uvMax.y - chart.uvMin.y);
        std::cout << "Island " << i << " (prim " << chart.primitiveID << "): "
                  << islandWidth << "x" << islandHeight << " cells"
                  << ", UV range: " << uvRange.x << "x" << uvRange.y
                  << ", gridResolution: " << gridResolution
                  << ", gutter: " << marginCells
                  << ", " << chart.triangleIndices.size() << " triangles" << std::endl;

        island.mask = Grid(islandWidth, islandHeight);
        island.collisionMask = Grid(islandWidth, islandHeight);
        
        // Upload triangle indices
        std::vector<uint32_t> triIndices;
        for (uint32_t triIdx : chart.triangleIndices) {
            triIndices.push_back(triIdx * 3 + 0);
            triIndices.push_back(triIdx * 3 + 1);
            triIndices.push_back(triIdx * 3 + 2);
        }
        
        cudaMalloc(&island.d_triangleIndices, triIndices.size() * sizeof(uint32_t));
        cudaMemcpy(island.d_triangleIndices, triIndices.data(), 
                   triIndices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Rasterize island directly into island.mask in cell space.
        int numTriangles = (int)chart.triangleIndices.size();
        int threadsPerBlock = 256;
        int blocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

        rasterizeIslandKernel<<<blocks, threadsPerBlock>>>(
            d_uvCoords,
            island.d_triangleIndices,
            numTriangles,
            island.minCell,
            gridResolution,
            island.mask
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[WRAPPER] ERROR launching rasterizeIslandKernel: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        
        // Create collision mask (dilated)
        if (marginCells > 0) {
            dim3 blockDim(16, 16);
            dim3 gridDim(
                (islandWidth + blockDim.x - 1) / blockDim.x,
                (islandHeight + blockDim.y - 1) / blockDim.y
            );
            
            dilateGridKernel<<<gridDim, blockDim>>>(
                island.mask,
                island.collisionMask,
                marginCells
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[WRAPPER] ERROR launching dilateGridKernel: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        } else {
            // Copy mask to collision mask
            cudaMemcpy(island.collisionMask.cells, island.mask.cells,
                      island.mask.stride * island.mask.height * sizeof(uint32_t),
                      cudaMemcpyDeviceToDevice);
        }

        // Precompute 90° CW rotated masks (for fast packing rotation checks).
        island.mask90 = Grid(island.mask.height, island.mask.width);
        island.collisionMask90 = Grid(island.collisionMask.height, island.collisionMask.width);

        {
            dim3 blockDim(16, 16);
            dim3 gridDim(
                (island.mask90.width + blockDim.x - 1) / blockDim.x,
                (island.mask90.height + blockDim.y - 1) / blockDim.y
            );
            rotateGridKernel<<<gridDim, blockDim>>>(island.mask, island.mask90, 1);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[WRAPPER] ERROR launching rotateGridKernel (mask): %s\n", cudaGetErrorString(err));
            }
        }

        {
            dim3 blockDim(16, 16);
            dim3 gridDim(
                (island.collisionMask90.width + blockDim.x - 1) / blockDim.x,
                (island.collisionMask90.height + blockDim.y - 1) / blockDim.y
            );
            rotateGridKernel<<<gridDim, blockDim>>>(island.collisionMask, island.collisionMask90, 1);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[WRAPPER] ERROR launching rotateGridKernel (collisionMask): %s\n", cudaGetErrorString(err));
            }
        }

        cudaDeviceSynchronize();

        // Precompute per-row active word bounds for collision masks (rot0/rot90)
        cudaMalloc(&island.d_collMinWord, island.collisionMask.height * sizeof(uint16_t));
        cudaMalloc(&island.d_collMaxWord, island.collisionMask.height * sizeof(uint16_t));
        cudaMalloc(&island.d_collMinWord90, island.collisionMask90.height * sizeof(uint16_t));
        cudaMalloc(&island.d_collMaxWord90, island.collisionMask90.height * sizeof(uint16_t));

        computeRowWordBoundsKernel<<<island.collisionMask.height, 128>>>(island.collisionMask, island.d_collMinWord, island.d_collMaxWord);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[WRAPPER] ERROR launching computeRowWordBoundsKernel (rot0): %s\n", cudaGetErrorString(err));
        }
        computeRowWordBoundsKernel<<<island.collisionMask90.height, 128>>>(island.collisionMask90, island.d_collMinWord90, island.d_collMaxWord90);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[WRAPPER] ERROR launching computeRowWordBoundsKernel (rot90): %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        
        islands.push_back(island);
    }
    
    // Sort islands by area (largest first) - common packing heuristic
    std::sort(islands.begin(), islands.end(), 
        [](const Island& a, const Island& b) {
            int areaA = a.mask.width * a.mask.height;
            int areaB = b.mask.width * b.mask.height;
            return areaA > areaB;  // Descending order
        });
    
    std::cout << "Created " << islands.size() << " islands for packing (sorted by size)" << std::endl;
    
    return islands;
}

PackedAtlas PackIslands(
    const std::vector<Island>& islands,
    const PackingConfig& config
) {
    PackedAtlas result;
    
    if (islands.empty()) {
        std::cerr << "ERROR: No islands to pack!" << std::endl;
        result.success = false;
        return result;
    }
    
    std::cout << "Packing " << islands.size() << " islands with " 
              << config.maxIterations << " iterations..." << std::endl;
    std::cout << "Grid size: " << config.gridSize << "x" << config.gridSize 
              << ", margin: " << config.marginCells << " cells" << std::endl;
    
    // Check if any island is too large
    bool hasOversizedIsland = false;
    for (const auto& island : islands) {
        if (island.mask.width > config.gridSize || island.mask.height > config.gridSize) {
            std::cerr << "WARNING: Island " << island.islandIndex 
                      << " (" << island.mask.width << "x" << island.mask.height 
                      << ") is too large for grid (" << config.gridSize << "x" << config.gridSize << ")" << std::endl;
            hasOversizedIsland = true;
        }
    }
    
    if (hasOversizedIsland) {
        std::cerr << "ERROR: Some islands are too large for the grid!" << std::endl;
        std::cerr << "Solution: Increase grid size (e.g., 1024, 2048, or 4096)" << std::endl;
        result.success = false;
        return result;
    }
    
    // Upload islands to device
    Island* d_islands;
    cudaMalloc(&d_islands, islands.size() * sizeof(Island));
    cudaMemcpy(d_islands, islands.data(), islands.size() * sizeof(Island),
               cudaMemcpyHostToDevice);
    
    // Allocate solutions
    int numSolutions = min(config.maxIterations, 256);  // Limit concurrent solutions
    Solution* d_solutions;
    cudaMalloc(&d_solutions, numSolutions * sizeof(Solution));
    
    // Initialize solutions
    std::vector<Solution> h_solutions(numSolutions);
    for (int i = 0; i < numSolutions; i++) {
        h_solutions[i].occupancyMask = Grid(config.gridSize, config.gridSize);
        cudaMalloc(&h_solutions[i].placements, islands.size() * sizeof(IslandPlacement));
        h_solutions[i].seed = config.randomSeed + i;
    }
    cudaMemcpy(d_solutions, h_solutions.data(), numSolutions * sizeof(Solution),
               cudaMemcpyHostToDevice);
    
    // Generate random seeds
    std::vector<int> seeds(numSolutions);
    srand(config.randomSeed == 0 ? (unsigned int)time(nullptr) : config.randomSeed);
    for (int i = 0; i < numSolutions; i++) {
        seeds[i] = rand();
    }
    int* d_seeds;
    cudaMalloc(&d_seeds, numSolutions * sizeof(int));
    cudaMemcpy(d_seeds, seeds.data(), numSolutions * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate optimal shelf width for roughly square packing
    // Find the largest island width to ensure shelf can fit at least 2-3 islands
    int maxIslandWidth = 0;
    int totalArea = 0;
    for (const auto& island : islands) {
        maxIslandWidth = max(maxIslandWidth, island.mask.width);
        totalArea += island.mask.width * island.mask.height;
    }
    
    // Target a roughly square layout, but ensure we can fit at least 2 islands per row
    int targetDimension = (int)sqrt((float)totalArea * 1.3f);  // 30% padding
    int minShelfWidth = maxIslandWidth * 2 + maxIslandWidth / 2;  // Fit ~2.5 islands
    int shelfWidth = max(minShelfWidth, min(targetDimension, config.gridSize));
    
    std::cout << "Total island area: " << totalArea << " cells² "
              << "→ Target shelf width: " << shelfWidth << " cells "
              << "(max island: " << maxIslandWidth << ")" << std::endl;
    
    // Launch packing kernel
    std::cout << "Launching " << numSolutions << " parallel packing attempts..." << std::endl;
    
    packSolutionKernel<<<numSolutions, 256>>>(
        d_solutions,
        d_islands,
        (int)islands.size(),
        config.gridSize,
        d_seeds,
        config.enableRotation,
        shelfWidth,            // target shelf width
        config.xAlignment,     // aligned-X constraint
        config.maxCandidatesPerIsland,
        config.maxDropSteps,
        config.shuffleOrderPerAttempt
    );
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("[WRAPPER] ERROR launching packSolutionKernel: %s\n", cudaGetErrorString(launchErr));
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during packing: " << cudaGetErrorString(err) << std::endl;
        result.success = false;
        
        // Cleanup
        for (int i = 0; i < numSolutions; i++) {
            h_solutions[i].occupancyMask.free();
            cudaFree(h_solutions[i].placements);
        }
        cudaFree(d_solutions);
        cudaFree(d_islands);
        cudaFree(d_seeds);
        
        return result;
    }
    
    // Download solutions
    cudaMemcpy(h_solutions.data(), d_solutions, numSolutions * sizeof(Solution),
               cudaMemcpyDeviceToHost);
    
    // Find best solution
    int bestIdx = -1;
    float bestFitness = 0.0f;
    int mostPlaced = 0;
    
    for (int i = 0; i < numSolutions; i++) {
        if (h_solutions[i].numPlaced > mostPlaced) {
            mostPlaced = h_solutions[i].numPlaced;
            bestIdx = i;
            bestFitness = h_solutions[i].fitness;
        } else if (h_solutions[i].numPlaced == mostPlaced && 
                   h_solutions[i].fitness > bestFitness) {
            bestIdx = i;
            bestFitness = h_solutions[i].fitness;
        }
    }
    
    std::cout << "Best solution placed " << mostPlaced << " / " << islands.size() 
              << " islands (fitness: " << bestFitness << ")" << std::endl;
    
    if (bestIdx < 0 || mostPlaced == 0) {
        std::cerr << "ERROR: No valid packing solution found!" << std::endl;
        std::cerr << "This may indicate islands are too large for the grid." << std::endl;
        result.success = false;
    } else {
        const Solution& best = h_solutions[bestIdx];
        
        // Download placements
        result.placements.resize(best.numPlaced);
        cudaMemcpy(result.placements.data(), best.placements,
                   best.numPlaced * sizeof(IslandPlacement),
                   cudaMemcpyDeviceToHost);
        
        result.atlasWidth = best.utilizedWidth;
        result.atlasHeight = best.utilizedHeight;
        result.fitness = best.fitness;
        
        // ===== DIAGNOSTIC STATISTICS =====
        std::cout << "\n=== PACKING DIAGNOSTICS ===" << std::endl;
        std::cout << "Original utilized area: " << result.atlasWidth << "x" << result.atlasHeight << std::endl;
        
        // Download occupancy mask to analyze
        int maskSize = best.occupancyMask.stride * best.occupancyMask.height;
        std::vector<uint32_t> occupancyData(maskSize);
        cudaMemcpy(occupancyData.data(), best.occupancyMask.cells,
                   maskSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // Count occupied cells in utilized area
        int occupiedCells = 0;
        for (int y = 0; y < result.atlasHeight; y++) {
            for (int x = 0; x < result.atlasWidth; x++) {
                int linearIdx = y * best.occupancyMask.width + x;
                int blockIdx = linearIdx / 32;
                int bitIdx = linearIdx % 32;
                if (occupancyData[blockIdx] & (1u << bitIdx)) {
                    occupiedCells++;
                }
            }
        }
        
        // Calculate per-island statistics
        int totalIslandPixels = 0;
        std::cout << "\nPer-Island Placement:" << std::endl;
        for (size_t i = 0; i < result.placements.size(); i++) {
            const auto& placement = result.placements[i];
            
            // Find island by islandIndex (islands array was sorted, so indices don't match positions)
            const Island* island = nullptr;
            for (const auto& isl : islands) {
                if (isl.islandIndex == placement.islandIndex) {
                    island = &isl;
                    break;
                }
            }
            
            if (!island) {
                std::cerr << "ERROR: Could not find island " << placement.islandIndex << std::endl;
                continue;
            }
            
            // Count actual island pixels (non-margin)
            int islandPixels = 0;
            std::vector<uint32_t> maskData(island->mask.stride * island->mask.height);
            cudaMemcpy(maskData.data(), island->mask.cells,
                       maskData.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            for (int y = 0; y < island->mask.height; y++) {
                for (int x = 0; x < island->mask.width; x++) {
                    int linearIdx = y * island->mask.width + x;
                    int blockIdx = linearIdx / 32;
                    int bitIdx = linearIdx % 32;
                    if (maskData[blockIdx] & (1u << bitIdx)) {
                        islandPixels++;
                    }
                }
            }
            
            totalIslandPixels += islandPixels;
            
            int w = island->collisionMask.width;  // Use collision mask for true dimensions
            int h = island->collisionMask.height;
            float coverage = (w * h > 0) ? (100.0f * islandPixels / (w * h)) : 0.0f;
            
            std::cout << "  Island " << placement.islandIndex 
                      << " @ (" << placement.gridX << "," << placement.gridY << ") "
                      << w << "x" << h << " cells"
                      << " [ends at Y=" << (placement.gridY + h) << "]"
                      << ", " << islandPixels << " pixels"
                      << " (" << coverage << "% coverage)" << std::endl;
        }
        
        // Calculate statistics
        int utilizedArea = result.atlasWidth * result.atlasHeight;
        float packingEfficiency = (utilizedArea > 0) ? (100.0f * occupiedCells / utilizedArea) : 0.0f;
        float islandEfficiency = (occupiedCells > 0) ? (100.0f * totalIslandPixels / occupiedCells) : 0.0f;
        
        std::cout << "\nSpace Utilization:" << std::endl;
        std::cout << "  Bounding box: " << utilizedArea << " cells" << std::endl;
        std::cout << "  Occupied (collision mask): " << occupiedCells << " cells" << std::endl;
        std::cout << "  Island pixels (actual geometry): " << totalIslandPixels << " cells" << std::endl;
        std::cout << "  Packing efficiency: " << packingEfficiency << "% (occupied / bounding box)" << std::endl;
        std::cout << "  Island coverage: " << islandEfficiency << "% (geometry / occupied)" << std::endl;
        std::cout << "  Overall efficiency: " << (100.0f * totalIslandPixels / utilizedArea) << "% (geometry / bounding box)" << std::endl;
        
        // Check for overlaps by verifying placement
        std::cout << "\nOverlap Detection:" << std::endl;
        bool hasOverlap = false;
        std::vector<uint32_t> verifyMask(maskSize, 0);
        
        for (size_t i = 0; i < result.placements.size(); i++) {
            const auto& placement = result.placements[i];
            
            // Find island by islandIndex (islands array was sorted)
            const Island* island = nullptr;
            for (const auto& isl : islands) {
                if (isl.islandIndex == placement.islandIndex) {
                    island = &isl;
                    break;
                }
            }
            
            if (!island) continue;
            
            // Download collision mask
            std::vector<uint32_t> collisionData(island->collisionMask.stride * island->collisionMask.height);
            cudaMemcpy(collisionData.data(), island->collisionMask.cells,
                       collisionData.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            int overlapCount = 0;
            
            // Check each cell of the island with proper rotation handling
            for (int y = 0; y < island->collisionMask.height; y++) {
                for (int x = 0; x < island->collisionMask.width; x++) {
                    int linearIdx = y * island->collisionMask.width + x;
                    int blockIdx = linearIdx / 32;
                    int bitIdx = linearIdx % 32;
                    
                    if (collisionData[blockIdx] & (1u << bitIdx)) {
                        // This cell is occupied in the island
                        // Apply rotation transform (same logic as writeIslandToMask)
                        int atlasX, atlasY;
                        switch (placement.rotation) {
                            case 0: // No rotation
                                atlasX = placement.gridX + x;
                                atlasY = placement.gridY + y;
                                break;
                            case 1: // 90° CW
                                atlasX = placement.gridX + y;
                                atlasY = placement.gridY + (island->collisionMask.width - 1 - x);
                                break;
                            case 2: // 180°
                                atlasX = placement.gridX + (island->collisionMask.width - 1 - x);
                                atlasY = placement.gridY + (island->collisionMask.height - 1 - y);
                                break;
                            case 3: // 270° CW
                                atlasX = placement.gridX + (island->collisionMask.height - 1 - y);
                                atlasY = placement.gridY + x;
                                break;
                            default:
                                atlasX = placement.gridX + x;
                                atlasY = placement.gridY + y;
                        }
                        
                        if (atlasX >= 0 && atlasX < best.occupancyMask.width &&
                            atlasY >= 0 && atlasY < best.occupancyMask.height) {
                            
                            int atlasLinearIdx = atlasY * best.occupancyMask.width + atlasX;
                            int atlasBlockIdx = atlasLinearIdx / 32;
                            int atlasBitIdx = atlasLinearIdx % 32;
                            
                            // Check if this cell was already set
                            if (verifyMask[atlasBlockIdx] & (1u << atlasBitIdx)) {
                                overlapCount++;
                                hasOverlap = true;
                            }
                            
                            // Mark as occupied
                            verifyMask[atlasBlockIdx] |= (1u << atlasBitIdx);
                        }
                    }
                }
            }
            
            if (overlapCount > 0) {
                std::cout << "  WARNING: Island " << placement.islandIndex 
                          << " has " << overlapCount << " overlapping cells!" << std::endl;
            }
        }
        
        if (!hasOverlap) {
            std::cout << "  ✓ No overlaps detected" << std::endl;
        } else {
            std::cout << "  ✗ OVERLAPS DETECTED!" << std::endl;
        }
        
        std::cout << "=========================\n" << std::endl;
        
        // Force square for better texture utilization
        int maxDim = max(result.atlasWidth, result.atlasHeight);
        result.atlasWidth = maxDim;
        result.atlasHeight = maxDim;
        std::cout << "Normalized to square: " << result.atlasWidth << "x" << result.atlasHeight << std::endl;
        
        // Calculate scaling factor
        result.scalingFactor = (maxDim > 0) ? (float)config.gridSize / (float)maxDim : 1.0f;
        
        result.success = (best.numPlaced == (int)islands.size());
        
        if (result.success) {
            std::cout << "Packing successful! All " << result.placements.size() 
                      << " islands placed" << std::endl;
        } else {
            std::cout << "Partial packing: " << result.placements.size() << " / " 
                      << islands.size() << " islands placed" << std::endl;
        }
        
        std::cout << "Atlas size: " << result.atlasWidth << "x" << result.atlasHeight 
                  << " cells (scale: " << result.scalingFactor << ")" << std::endl;
    }
    
    // Cleanup
    for (int i = 0; i < numSolutions; i++) {
        h_solutions[i].occupancyMask.free();
        cudaFree(h_solutions[i].placements);
    }
    cudaFree(d_solutions);
    cudaFree(d_islands);
    cudaFree(d_seeds);
    
    return result;
}

void RemapUVsToAtlas(
    float2* d_uvCoords,
    const int* d_primitiveIDs,
    int numVertices,
    const std::vector<UVChart>& charts,
    const std::vector<Island>& islands,  // Need islands for cell sizes
    const PackedAtlas& atlas,
    int gridResolution
) {
    if (!atlas.success || atlas.placements.empty()) {
        std::cerr << "ERROR: Cannot remap UVs - invalid atlas!" << std::endl;
        return;
    }
    
    std::cout << "Remapping UVs (spec): " << numVertices << " vertices, "
              << charts.size() << " charts, " << atlas.placements.size() << " placements" << std::endl;

    // Build per-chart placement table (indexed by chartIdx/islandIndex)
    std::vector<IslandPlacement> placementsByChart(charts.size());
    for (auto& p : placementsByChart) p = IslandPlacement(); // islandIndex=-1
    for (const auto& p : atlas.placements) {
        if (p.islandIndex >= 0 && p.islandIndex < (int)placementsByChart.size()) {
            placementsByChart[p.islandIndex] = p;
        }
    }

    // Build per-chart minCell + dims (rot0) tables
    std::vector<int2> chartMinCells(charts.size(), make_int2(0, 0));
    std::vector<int2> chartDims(charts.size(), make_int2(1, 1));
    for (const auto& isl : islands) {
        if (isl.islandIndex >= 0 && isl.islandIndex < (int)charts.size()) {
            chartMinCells[isl.islandIndex] = isl.minCell;
            chartDims[isl.islandIndex] = make_int2(isl.mask.width, isl.mask.height);
        }
    }

    // Build primitiveID -> chartIdx lookup (sparse, but primIDs are small in this app)
    int maxPrim = -1;
    for (const auto& c : charts) maxPrim = max(maxPrim, c.primitiveID);
    int primToChartSize = maxPrim + 1;
    std::vector<int> primToChart(std::max(primToChartSize, 1), -1);
    for (int ci = 0; ci < (int)charts.size(); ++ci) {
        int pid = charts[ci].primitiveID;
        if (pid >= 0 && pid < (int)primToChart.size()) primToChart[pid] = ci;
    }

    // Upload to device
    IslandPlacement* d_placementsByChart = nullptr;
    int* d_primToChart = nullptr;
    int2* d_chartMinCells = nullptr;
    int2* d_chartDims = nullptr;

    cudaMalloc(&d_placementsByChart, placementsByChart.size() * sizeof(IslandPlacement));
    cudaMalloc(&d_primToChart, primToChart.size() * sizeof(int));
    cudaMalloc(&d_chartMinCells, chartMinCells.size() * sizeof(int2));
    cudaMalloc(&d_chartDims, chartDims.size() * sizeof(int2));

    cudaMemcpy(d_placementsByChart, placementsByChart.data(),
               placementsByChart.size() * sizeof(IslandPlacement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primToChart, primToChart.data(),
               primToChart.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chartMinCells, chartMinCells.data(),
               chartMinCells.size() * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chartDims, chartDims.data(),
               chartDims.size() * sizeof(int2), cudaMemcpyHostToDevice);
    
    // Launch remapping kernel
    int threadsPerBlock = 256;
    int blocks = (numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    remapUVsKernel<<<blocks, threadsPerBlock>>>(
        d_uvCoords,
        (const float*)d_primitiveIDs,  // Cast to float* for kernel
        numVertices,
        d_placementsByChart,
        d_primToChart,
        (int)primToChart.size(),
        d_chartMinCells,
        d_chartDims,
        (int)charts.size(),
        gridResolution,
        atlas.atlasWidth,     // Actual utilized width
        atlas.atlasHeight     // Actual utilized height
    );
    cudaError_t remapErr = cudaGetLastError();
    if (remapErr != cudaSuccess) {
        printf("[WRAPPER] ERROR launching remapUVsKernel: %s\n", cudaGetErrorString(remapErr));
    }
    cudaDeviceSynchronize();
    
    // Debug: Download a few UVs to verify remapping worked
    std::vector<float2> testUVs(min(10, numVertices));
    cudaMemcpy(testUVs.data(), d_uvCoords, testUVs.size() * sizeof(float2), cudaMemcpyDeviceToHost);
    std::cout << "Sample remapped UVs:" << std::endl;
    for (size_t i = 0; i < testUVs.size(); i++) {
        std::cout << "  UV[" << i << "] = (" << testUVs[i].x << ", " << testUVs[i].y << ")" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_placementsByChart);
    cudaFree(d_primToChart);
    cudaFree(d_chartMinCells);
    cudaFree(d_chartDims);
    
    std::cout << "UV remapping complete (normalized to " << atlas.atlasWidth 
              << "x" << atlas.atlasHeight << " utilized area)" << std::endl;
}

void FreeIslands(std::vector<Island>& islands) {
    for (Island& island : islands) {
        island.mask.free();
        island.collisionMask.free();
        island.mask90.free();
        island.collisionMask90.free();
        if (island.d_collMinWord) cudaFree(island.d_collMinWord);
        if (island.d_collMaxWord) cudaFree(island.d_collMaxWord);
        if (island.d_collMinWord90) cudaFree(island.d_collMinWord90);
        if (island.d_collMaxWord90) cudaFree(island.d_collMaxWord90);
        island.d_collMinWord = island.d_collMaxWord = nullptr;
        island.d_collMinWord90 = island.d_collMaxWord90 = nullptr;
        if (island.d_triangleIndices) {
            cudaFree(island.d_triangleIndices);
            island.d_triangleIndices = nullptr;
        }
    }
    islands.clear();
}

} // namespace GridUVPacker


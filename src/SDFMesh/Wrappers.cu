
#include "Commons.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Wrapper functions to launch kernels from main.cpp (which is .cpp)

extern __global__ void scoutActiveBlocks(SDFGrid grid, float time);
extern __global__ void countActiveBlockTriangles(SDFGrid grid, float time);
extern __global__ void generateActiveBlockTriangles(SDFGrid grid, float time);

// Dual Contouring kernels (implemented in MarchingCubesKernels.cu for now)
extern __global__ void buildBlockToActiveMap(SDFGrid grid);
extern __global__ void dcMarkCells(SDFGrid grid, float time);
extern __global__ void dcSolveCellVertices(SDFGrid grid, float time, unsigned int maxCellVertices, float qefBlend);
extern __global__ void dcSmoothCellNormals(SDFGrid grid, float cosAngleThreshold);
extern __global__ void dcCountQuads(SDFGrid grid);
extern __global__ void dcGenerateQuads(SDFGrid grid, float time);

extern "C" void launchScoutActiveBlocks(SDFGrid& grid, float time) {
    unsigned int numBlocks = grid.blocksDim.x * grid.blocksDim.y * grid.blocksDim.z;
    int threads = 256;
    int blocks = (numBlocks + threads - 1) / threads;
    scoutActiveBlocks<<<blocks, threads>>>(grid, time);
}

extern "C" void launchCountActiveBlockTriangles(SDFGrid& grid, int numActiveBlocks, float time) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512 for 8^3
    countActiveBlockTriangles<<<numActiveBlocks, threads>>>(grid, time);
}

extern "C" void launchGenerateActiveBlockTriangles(SDFGrid& grid, int numActiveBlocks, float time) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512 for 8^3
    generateActiveBlockTriangles<<<numActiveBlocks, threads>>>(grid, time);
}

// --------------------------------------------------------------------------
// Dual Contouring wrappers
// --------------------------------------------------------------------------

extern "C" void launchBuildBlockToActiveMap(SDFGrid& grid, int numActiveBlocks) {
    if (numActiveBlocks == 0) return;
    int threads = 256;
    int blocks = (numActiveBlocks + threads - 1) / threads;
    buildBlockToActiveMap<<<blocks, threads>>>(grid);
}

extern "C" void launchDCMarkCells(SDFGrid& grid, int numActiveBlocks, float time) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512
    dcMarkCells<<<numActiveBlocks, threads>>>(grid, time);
}

extern "C" void launchDCSolveCellVertices(SDFGrid& grid, int numActiveBlocks, float time, unsigned int maxCellVertices, float qefBlend) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512
    dcSolveCellVertices<<<numActiveBlocks, threads>>>(grid, time, maxCellVertices, qefBlend);
}

extern "C" void launchDCSmoothCellNormals(SDFGrid& grid, int numActiveBlocks, float cosAngleThreshold) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512
    dcSmoothCellNormals<<<numActiveBlocks, threads>>>(grid, cosAngleThreshold);
}

extern "C" void launchDCCountQuads(SDFGrid& grid, int numActiveBlocks) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512
    dcCountQuads<<<numActiveBlocks, threads>>>(grid);
}

extern "C" void launchDCGenerateQuads(SDFGrid& grid, int numActiveBlocks, float time) {
    if (numActiveBlocks == 0) return;
    int threads = SDF_BLOCK_SIZE_CUBED; // 512
    dcGenerateQuads<<<numActiveBlocks, threads>>>(grid, time);
}

extern "C" void getScanStorageSize(unsigned int num_items, size_t* temp_storage_bytes) {
    void* d_temp = NULL;
    unsigned int* d_in = NULL;
    unsigned int* d_out = NULL;
    cub::DeviceScan::ExclusiveSum(d_temp, *temp_storage_bytes, d_in, d_out, num_items);
}

extern "C" void launchScan(unsigned int* d_input, unsigned int* d_output, unsigned int numItems, void* d_temp_storage, size_t temp_storage_bytes) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, numItems);
}

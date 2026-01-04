#pragma once

#include <cuda_runtime.h>
#include <algorithm>

// Simplified Vector Types and Math Helpers

struct __align__(16) float4_ {
    float x, y, z, w;
};

struct __align__(16) float3_ {
    float x, y, z;
};

// SDF Primitive Types
enum PrimitiveType {
    SDF_SPHERE = 0,
    SDF_BOX,
    SDF_TORUS,
    SDF_CYLINDER,
    SDF_CAPSULE,
    SDF_CONE,
    SDF_ROUNDED_BOX,
    SDF_ROUNDED_CYLINDER,
    SDF_ROUNDED_CONE,
    SDF_ELLIPSOID,
    SDF_TRIANGULAR_PRISM,
    SDF_OCTOHEDRON,
    SDF_HEX_PRISM,
    SDF_SOLID_ANGLE,
    SDF_CUSTOM_FLOAT, // For custom primitives if needed
    SDF_CUSTOM_VECTOR
};

// SDF Operations
enum OperationType {
    SDF_UNION = 0,
    SDF_SUBTRACT,
    SDF_INTERSECT,
    SDF_UNION_BLEND,
    SDF_SUBTRACT_BLEND,
    SDF_INTERSECT_BLEND
};

// SDF Displacements
enum DisplacementType {
    DISP_NONE = 0,
    DISP_SINE, // Basic test
    DISP_TWIST,
    DISP_BEND,
    DISP_ELONGATE,
    DISP_ROUND,
    DISP_ONION,
    DISP_EXTRUSION,
    DISP_REVOLUTION
};

// UV Mapping Modes
enum UVMode {
    UV_PRIMITIVE = 0,       // Use primitive's canonical mapping
    UV_WORLD_TRIPLANAR,     // World-space triplanar projection
    UV_PLANAR_X,            // Planar projection along X axis
    UV_PLANAR_Y,            // Planar projection along Y axis
    UV_PLANAR_Z             // Planar projection along Z axis
};

// Optimized Struct Layout for Alignment (128 bytes stride)
struct SDFPrimitive {
    // 16-byte aligned members first
    float4 rotation;      // 16 bytes
    
    float dispParams[4];  // 16 bytes
    
    // Groups of 16 bytes
    float3 position;      // 12 bytes
    float blendFactor;    // 4 bytes
    
    float3 scale;         // 12 bytes
    float rounding;       // 4 bytes
    
    float3 color;         // 12 bytes
    float annular;        // 4 bytes
    
    float params[6];      // 24 bytes
    
    int type;             // 4 bytes
    int operation;        // 4 bytes
    
    int displacement;     // 4 bytes
    int _pad1;            // 4 bytes (align to 8)
    
    // UV Parameters (add 32 bytes)
    float2 uvScale;       // 8 bytes
    float2 uvOffset;      // 8 bytes
    float uvRotation;     // 4 bytes
    int uvMode;           // 4 bytes (UVMode enum)
    int textureID;        // 4 bytes
    int atlasIslandID;    // 4 bytes (assigned during atlas packing, -1 if not used)
    
    float _pad2[2];       // 8 bytes (to maintain 16-byte alignment)
};

struct BVHNode {
    float min[3];
    float max[3];
    int left;  // Child index or primitive index (if leaf)
    int right; // Child index or -1 (if leaf)
};

// Macro Block Settings
#define SDF_BLOCK_SIZE 8
#define SDF_BLOCK_SIZE_CUBED (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)

struct SDFGrid {
    // Removed dense buffers
    // float* d_sdf; 
    // unsigned int* d_vertexCounts; 
    // unsigned int* d_vertexOffsets; 
    // float3* d_colors;
    
    // Active Block Management
    int* d_activeBlocks;      // List of active block IDs
    int* d_activeBlockCount;  // Counter
    int maxBlocks;            // Max possible blocks (for safety checks)
    
    int3 blocksDim;           // Dimensions in blocks
    
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    float3 origin;
    float cellSize;
    
    // Scene Data
    SDFPrimitive* d_primitives;
    unsigned int numPrimitives;
    unsigned int numBVHPrimitives; // Number of primitives in the BVH (Unions)

    // BVH Data
    BVHNode* d_bvhNodes;
    unsigned int numBVHNodes;
    
    // Data for MC (Packed)
    unsigned int* d_packetVertexCounts; // Per-active-voxel vertex/triangle count
    unsigned int* d_packetVertexOffsets; // Scanned offsets

    // --- Dual Contouring (packed, optional/lazy allocated) ---
    // Mapping from global linear block index -> activeBlockId, or -1 if inactive.
    int* d_blockToActiveId;

    // Per packed-cell (activeBlockId * SDF_BLOCK_SIZE_CUBED + tid):
    // - 8-bit corner sign mask (bit i corresponds to marchingCubeCorners[i])
    unsigned char* d_dcCornerMasks;

    // Scan result for "hasVertex" (0/1) counts: maps packed-cell -> compact DC cell-vertex index.
    // This must persist across quad counting / generation, so it cannot reuse d_packetVertexOffsets.
    unsigned int* d_dcCellVertexOffsets;

    // Compact DC cell-vertex buffer (size = maxVertices entries). Stored as quantized offset within cell.
    // Decode: p = cellMin + (ushort.xyz / 65535.0f) * cellSize.
    ushort4* d_dcCellVertices;

    // Compact DC cell-vertex normal buffer (size = maxVertices entries).
    // Stored as signed normalized int16 in [-32767, 32767] for xyz.
    short4* d_dcCellNormals;
    short4* d_dcCellNormalsTmp;
    
    // Results
    float4* d_vertices; // Output vertices
    float4* d_vertexColors; // Output vertex colors
    unsigned int* d_indices; // Output indices (optional/trivial)
    
    // UV Output (NEW)
    float2* d_uvCoords;     // Output UV coordinates (parallel to d_vertices)
    int* d_primitiveIDs;    // Which primitive each vertex came from
    
    // Normal Output (NEW)
    float4* d_normals;      // Output normals (parallel to d_vertices)
    
    unsigned int maxVertices;
    unsigned int maxIndices;
    
    unsigned int* d_totalVertices; // Single value on device
    unsigned int* d_totalIndices; // Single value on device
};

// Helper functions similar to PhysX ones

__host__ __device__ inline bool outOfRange(const SDFGrid& grid, int index) {
    return index >= (grid.width * grid.height * grid.depth);
}

__host__ __device__ inline int4 getGridCoordinates(const SDFGrid& grid, int index) {
    int4 coords;
    coords.x = index % grid.width;
    int temp = index / grid.width;
    coords.y = temp % grid.height;
    coords.z = temp / grid.height;
    coords.w = 0;
    return coords;
}

__host__ __device__ inline bool outOfBounds(const SDFGrid& grid, int4 coords) {
    // Marching cubes looks at cell + 1, so we need to be careful not to go out of bounds
    return (coords.x >= grid.width - 1 || coords.y >= grid.height - 1 || coords.z >= grid.depth - 1);
}

__host__ __device__ inline float3 getLocation(const SDFGrid& grid, int4 xyz) {
    return make_float3(
        grid.origin.x + xyz.x * grid.cellSize,
        grid.origin.y + xyz.y * grid.cellSize,
        grid.origin.z + xyz.z * grid.cellSize
    );
}

// Math helpers
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

// Float2 Operators
__host__ __device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float2 operator*(const float2& a, float s) {
    return make_float2(a.x * s, a.y * s);
}

__host__ __device__ inline float2 operator*(float s, const float2& a) {
    return make_float2(a.x * s, a.y * s);
}

// Standard mix (linear interpolation)
__host__ __device__ inline float mix(float x, float y, float a) {
    return x * (1.0f - a) + y * a;
}

__host__ __device__ inline float clamp(float v, float min_v, float max_v) {
    return fmaxf(min_v, fminf(v, max_v));
}

__host__ __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Quaternion Rotation
__host__ __device__ inline float3 rotateVector(float3 v, float4 q) {
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s*s - dot(u, u)) * v + 2.0f * s * make_float3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

__host__ __device__ inline float3 invRotateVector(float3 v, float4 q) {
    // Inverse rotation is just using conjugate quaternion (-x, -y, -z, w)
    float4 invQ = make_float4(-q.x, -q.y, -q.z, q.w);
    return rotateVector(v, invQ);
}

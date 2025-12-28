# Primitive UV Mapping - Implementation Guide

This guide provides step-by-step instructions for implementing the primitive-based UV mapping system.

## Quick Reference

**Files to Modify:**
- `src/SDFMesh/Commons.cuh` - Data structures
- `src/SDFMesh/MarchingCubesKernels.cu` - UV generation kernels  
- `src/SDFMesh/CudaSDFMesh.h/cpp` - CPU-side management
- `src/main.cpp` - Integration and atlas packing

**Estimated Effort:** 5-7 weeks (2-3 weeks for MVP)

---

## Phase 1: Core UV Generation (Week 1-2)

### Step 1.1: Extend SDFPrimitive Structure

**File**: `src/SDFMesh/Commons.cuh`

```cpp
struct SDFPrimitive {
    // ... existing fields ...
    
    // ADD NEW FIELDS (32 bytes total)
    float2 uvScale;        // 8 bytes - UV tiling/stretching
    float2 uvOffset;       // 8 bytes - UV translation
    float uvRotation;      // 4 bytes - UV rotation in radians
    int uvMode;            // 4 bytes - UVMode enum value
    int textureID;         // 4 bytes - texture index (future use)
    int atlasIslandID;     // 4 bytes - assigned during packing (-1 if unused)
    
    // Adjust padding to maintain 16-byte stride alignment
    // Total struct size should be multiple of 16
};

// ADD NEW ENUM
enum UVMode {
    UV_PRIMITIVE = 0,       // Use primitive's canonical mapping
    UV_WORLD_TRIPLANAR,     // World-space triplanar projection
    UV_PLANAR_X,            // Planar projection along X axis
    UV_PLANAR_Y,            // Planar projection along Y axis
    UV_PLANAR_Z             // Planar projection along Z axis
};
```

**Testing**: Compile and ensure no size/alignment warnings.

---

### Step 1.2: Extend SDFGrid Structure

**File**: `src/SDFMesh/Commons.cuh`

```cpp
struct SDFGrid {
    // ... existing fields ...
    
    // ADD NEW FIELDS
    float2* d_uvCoords;        // Output UV coordinates
    int* d_primitiveIDs;       // Which primitive each vertex belongs to
};
```

---

### Step 1.3: Update Helper Functions

**File**: `src/SDFMesh/CudaSDFUtil.h`

```cpp
// UPDATE ALL CreateXxxPrim() functions to initialize UV fields

inline SDFPrimitive CreatePrimitive(/* ... existing params ... */) {
    SDFPrimitive prim;
    // ... existing initialization ...
    
    // ADD DEFAULT UV INITIALIZATION
    prim.uvScale = make_float2(1.0f, 1.0f);
    prim.uvOffset = make_float2(0.0f, 0.0f);
    prim.uvRotation = 0.0f;
    prim.uvMode = UV_PRIMITIVE;
    prim.textureID = 0;
    prim.atlasIslandID = -1;
    
    return prim;
}
```

---

### Step 1.4: Create UV Mapping Functions

**File**: `src/SDFMesh/MarchingCubesKernels.cu` (add near top)

```cuda
// UV MAPPING FUNCTIONS

__device__ float2 uvSphere(float3 p) {
    float len = length(p);
    if (len < 1e-6f) return make_float2(0.5f, 0.5f);
    
    p = p * (1.0f / len);  // Normalize
    float theta = atan2f(p.z, p.x);
    float phi = asinf(clamp(p.y, -1.0f, 1.0f));
    
    return make_float2(
        (theta + M_PI) / (2.0f * M_PI),
        (phi + M_PI_2) / M_PI
    );
}

__device__ float2 uvBox(float3 p, float3 normal) {
    float3 absN = make_float3(fabsf(normal.x), fabsf(normal.y), fabsf(normal.z));
    
    if (absN.x > absN.y && absN.x > absN.z) {
        // X-face
        return make_float2(p.z * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
    } else if (absN.y > absN.z) {
        // Y-face
        return make_float2(p.x * 0.5f + 0.5f, p.z * 0.5f + 0.5f);
    } else {
        // Z-face
        return make_float2(p.x * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
    }
}

__device__ float2 uvCylinder(float3 p, float height) {
    float theta = atan2f(p.z, p.x);
    float u = (theta + M_PI) / (2.0f * M_PI);
    float v = clamp((p.y / height) * 0.5f + 0.5f, 0.0f, 1.0f);
    return make_float2(u, v);
}

__device__ float2 uvTorus(float3 p, float majorRadius, float minorRadius) {
    float theta = atan2f(p.z, p.x);
    
    float2 q = make_float2(length(make_float2(p.x, p.z)) - majorRadius, p.y);
    float phi = atan2f(q.y, q.x);
    
    return make_float2(
        (theta + M_PI) / (2.0f * M_PI),
        (phi + M_PI) / (2.0f * M_PI)
    );
}

__device__ float2 uvCapsule(float3 p, float height, float radius) {
    // Similar to cylinder but handle hemispherical caps
    if (p.y > height) {
        // Top cap (sphere)
        float3 pCap = make_float3(p.x, p.y - height, p.z);
        return uvSphere(pCap);
    } else if (p.y < 0.0f) {
        // Bottom cap (sphere)
        float3 pCap = make_float3(p.x, p.y, p.z);
        return uvSphere(pCap);
    } else {
        // Cylindrical body
        return uvCylinder(p, height);
    }
}

__device__ float2 transformUV(float2 uv, float2 scale, float2 offset, float rotation) {
    // Apply scale
    uv = make_float2(uv.x * scale.x, uv.y * scale.y);
    
    // Apply rotation
    if (fabsf(rotation) > 1e-6f) {
        float c = cosf(rotation);
        float s = sinf(rotation);
        float2 rotated = make_float2(
            uv.x * c - uv.y * s,
            uv.x * s + uv.y * c
        );
        uv = rotated;
    }
    
    // Apply offset
    uv = uv + offset;
    
    return uv;
}

__device__ float3 approximateLocalNormal(float3 localPos, const SDFPrimitive& prim, float time) {
    // Simple gradient approximation
    const float eps = 0.001f;
    
    // Helper lambda (if C++14) or just inline the SDF evaluation
    // For simplicity, approximate from position assuming canonical shapes
    // For box: normal is direction of largest component
    // For sphere: normal is direction from center
    
    if (prim.type == SDF_SPHERE) {
        return localPos * (1.0f / (length(localPos) + 1e-6f));
    } else if (prim.type == SDF_BOX || prim.type == SDF_ROUNDED_BOX) {
        float3 absP = abs_f3(localPos);
        if (absP.x > absP.y && absP.x > absP.z) {
            return make_float3(copysignf(1.0f, localPos.x), 0.0f, 0.0f);
        } else if (absP.y > absP.z) {
            return make_float3(0.0f, copysignf(1.0f, localPos.y), 0.0f);
        } else {
            return make_float3(0.0f, 0.0f, copysignf(1.0f, localPos.z));
        }
    }
    
    // Default: use position direction
    return localPos * (1.0f / (length(localPos) + 1e-6f));
}

__device__ float2 computePrimitiveUV(
    const SDFPrimitive& prim,
    float3 localPos,
    float time
) {
    float2 uv = make_float2(0.0f, 0.0f);
    
    // Compute normal approximation
    float3 localNormal = approximateLocalNormal(localPos, prim, time);
    
    // Select UV mapping based on primitive type
    switch(prim.type) {
        case SDF_SPHERE:
            uv = uvSphere(localPos);
            break;
        case SDF_BOX:
        case SDF_ROUNDED_BOX:
            uv = uvBox(localPos, localNormal);
            break;
        case SDF_CYLINDER:
        case SDF_ROUNDED_CYLINDER:
            uv = uvCylinder(localPos, prim.params[0]);
            break;
        case SDF_TORUS:
            uv = uvTorus(localPos, prim.params[0], prim.params[1]);
            break;
        case SDF_CAPSULE:
            uv = uvCapsule(localPos, prim.params[0], prim.params[1]);
            break;
        case SDF_HEX_PRISM:
        case SDF_TRIANGULAR_PRISM:
            // Use cylindrical approximation
            uv = uvCylinder(localPos, prim.params[1]);
            break;
        case SDF_CONE:
        case SDF_ROUNDED_CONE:
            // Cylindrical with varying radius
            uv = uvCylinder(localPos, prim.params[0]);
            break;
        default:
            // Planar fallback (XY projection)
            uv = make_float2(localPos.x * 0.5f + 0.5f, localPos.y * 0.5f + 0.5f);
            break;
    }
    
    // Apply UV transforms
    uv = transformUV(uv, prim.uvScale, prim.uvOffset, prim.uvRotation);
    
    return uv;
}
```

**Testing**: Add debug printf in one UV function to verify it's called.

---

### Step 1.5: Modify `map()` Function

**File**: `src/SDFMesh/MarchingCubesKernels.cu`

Update the `map()` signature and implementation:

```cuda
// OLD SIGNATURE:
// __device__ void map(float3 p_world, const SDFGrid& grid, float time, 
//                     float& outDist, float3& outColor)

// NEW SIGNATURE:
__device__ void map(
    float3 p_world, 
    const SDFGrid& grid, 
    float time, 
    float& outDist, 
    float3& outColor,
    int& outPrimitiveID,        // NEW
    float3& outLocalPos         // NEW (pre-distortion local position)
) {
    float d = 1e10f;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    int closestPrimID = -1;
    float3 closestLocalPos = make_float3(0.0f, 0.0f, 0.0f);
    
    // BVH Traversal (UNIONS)
    // ... existing BVH code ...
    
    while (stackPtr > 0 && iter < 256) {
        // ... existing node traversal ...
        
        if (node.right == -1) { // Leaf
            int i = node.left;
            SDFPrimitive prim = grid.d_primitives[i];
            
            float3 p = p_world - prim.position;
            p = invRotateVector(p, prim.rotation);
            p = make_float3(p.x / prim.scale.x, p.y / prim.scale.y, p.z / prim.scale.z);
            
            // STORE PRE-DISTORTION POSITION
            float3 p_preDisp = p;
            
            // Apply displacement
            if (prim.displacement == DISP_TWIST) p = dispTwist(p, prim.dispParams[0]);
            else if (prim.displacement == DISP_BEND) p = dispBend(p, prim.dispParams[0]);
            else if (prim.displacement == DISP_SINE) p.y += sinf(p.x * prim.dispParams[0] + time) * prim.dispParams[1];
            
            // Evaluate SDF
            float d_prim = /* ... existing SDF evaluation ... */;
            
            // ... existing scale/annular/rounding adjustments ...
            
            // Combine
            if (d == 1e10f) {
                d = d_prim;
                color = prim.color;
                closestPrimID = i;                    // TRACK PRIMITIVE
                closestLocalPos = p_preDisp;          // TRACK LOCAL POS
            } else {
                switch(prim.operation) {
                    case SDF_UNION:
                        if (d_prim < d) {
                            d = d_prim;
                            color = prim.color;
                            closestPrimID = i;              // UPDATE
                            closestLocalPos = p_preDisp;    // UPDATE
                        }
                        break;
                    case SDF_SUBTRACT:
                        {
                            float d_old = d;
                            d = opSubtract(d, d_prim);
                            if (d > d_old) {  // Subtracted surface is visible
                                color = prim.color;
                                closestPrimID = i;              // Interior uses cutter's UV
                                closestLocalPos = p_preDisp;
                            }
                        }
                        break;
                    // ... other operations ...
                }
            }
        }
    }
    
    // Global Primitives (NON-UNIONS)
    for (int i = grid.numBVHPrimitives; i < grid.numPrimitives; ++i) {
        // ... same logic as above ...
    }
    
    outDist = d;
    outColor = color;
    outPrimitiveID = closestPrimID;      // OUTPUT
    outLocalPos = closestLocalPos;       // OUTPUT
}
```

**Important**: Update ALL calls to `map()` throughout the file to include new parameters.

---

### Step 1.6: Update Marching Cubes Kernels

**File**: `src/SDFMesh/MarchingCubesKernels.cu`

Modify `generateActiveBlockTriangles`:

```cuda
__global__ void generateActiveBlockTriangles(SDFGrid grid, float time) {
    // ... existing voxel processing code ...
    
    // Generate Triangles
    unsigned int write = first;
    for (int i = 0; i < num; i += 3) {
        float3 pTri[3];
        // ... existing triangle vertex interpolation ...
        
        if (isDegenerateTri(pTri, grid.cellSize)) continue;
        if (write + 2 >= grid.maxVertices) break;

        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float3 p = pTri[j];

            float dist; 
            float3 color;
            int primitiveID;          // NEW
            float3 localPos;          // NEW
            
            // UPDATED MAP CALL
            map(p, grid, time, dist, color, primitiveID, localPos);

            // Write position and color (existing)
            grid.d_vertices[write + j] = make_float4(p.x, p.y, p.z, 1.0f);
            if (grid.d_vertexColors) {
                grid.d_vertexColors[write + j] = make_float4(color.x, color.y, color.z, 1.0f);
            }
            
            // WRITE PRIMITIVE ID (NEW)
            if (grid.d_primitiveIDs) {
                grid.d_primitiveIDs[write + j] = primitiveID;
            }
            
            // COMPUTE AND WRITE UVS (NEW)
            if (grid.d_uvCoords && primitiveID >= 0) {
                SDFPrimitive prim = grid.d_primitives[primitiveID];
                float2 uv = computePrimitiveUV(prim, localPos, time);
                grid.d_uvCoords[write + j] = uv;
            } else if (grid.d_uvCoords) {
                // Fallback for invalid primitive ID
                grid.d_uvCoords[write + j] = make_float2(0.0f, 0.0f);
            }
        }
        
        // ... rest of kernel ...
    }
}
```

Also update `countActiveBlockTriangles` kernel (simpler - just update `map()` calls):

```cuda
__global__ void countActiveBlockTriangles(SDFGrid grid, float time) {
    // ... existing code ...
    
    for (int i = 0; i < 8; i++) {
        float3 p = getLocation(grid, make_int4(xyz.x + dx, xyz.y + dy, xyz.z + dz, 0));
        
        float d; 
        float3 c;
        int primID;       // NEW (unused here)
        float3 localPos;  // NEW (unused here)
        
        map(p, grid, time, d, c, primID, localPos);  // UPDATED CALL
        
        cubeVal[i] = d;
        if (d > isoThreshold) code |= (1 << i);
    }
    
    // ... rest of kernel ...
}
```

---

### Step 1.7: Update CudaSDFMesh Class

**File**: `src/SDFMesh/CudaSDFMesh.h`

```cpp
class CudaSDFMesh {
public:
    // ... existing methods ...
    
    // NEW: Enable/disable UV generation
    void EnableUVGeneration(bool enable) { m_generateUVs = enable; }
    
    // MODIFIED: Update signature to include UV output
    void Update(
        float time, 
        float4* d_outVertices, 
        float4* d_outColors, 
        unsigned int* d_outIndices,
        float2* d_outUVs = nullptr,          // NEW (optional)
        int* d_outPrimitiveIDs = nullptr     // NEW (optional)
    );
    
private:
    bool m_generateUVs = false;  // NEW
};
```

**File**: `src/SDFMesh/CudaSDFMesh.cpp`

```cpp
void CudaSDFMesh::Update(
    float time, 
    float4* d_outVertices, 
    float4* d_outColors, 
    unsigned int* d_outIndices,
    float2* d_outUVs,
    int* d_outPrimitiveIDs
) {
    // ... existing code ...
    
    // Assign Output Pointers
    d_grid.d_vertices = d_outVertices;
    d_grid.d_vertexColors = d_outColors;
    d_grid.d_indices = d_outIndices;
    d_grid.d_uvCoords = d_outUVs;              // NEW
    d_grid.d_primitiveIDs = d_outPrimitiveIDs; // NEW
    
    // ... rest of existing code ...
}
```

---

### Step 1.8: Test with Single Primitive

**File**: `src/main.cpp`

Add test code:

```cpp
// In initGL(), after creating buffers, add:

// UV Buffer (for testing)
GLuint uvbo_test;
glGenBuffers(1, &uvbo_test);
glBindBuffer(GL_ARRAY_BUFFER, uvbo_test);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);

// Register with CUDA
struct cudaGraphicsResource* cudaUVBO;
cudaGraphicsGLRegisterBuffer(&cudaUVBO, uvbo_test, cudaGraphicsRegisterFlagsWriteDiscard);

// In main loop, enable UV generation:
mesh.EnableUVGeneration(true);

// Map UV buffer
float2* d_uvPtr;
cudaGraphicsMapResources(1, &cudaUVBO, 0);
cudaGraphicsResourceGetMappedPointer((void**)&d_uvPtr, &size, cudaUVBO);

// Call Update with UV output
mesh.Update(time, d_vboPtr, d_cboPtr, d_iboPtr, d_uvPtr, nullptr);

cudaGraphicsUnmapResources(1, &cudaUVBO, 0);

// Optionally: Download and print first few UVs to verify
float2 testUVs[10];
cudaMemcpy(testUVs, d_uvPtr, 10 * sizeof(float2), cudaMemcpyDeviceToHost);
for (int i = 0; i < 10; ++i) {
    std::cout << "UV[" << i << "]: " << testUVs[i].x << ", " << testUVs[i].y << std::endl;
}
```

**Expected Output**: UVs should be in [0,1] range (or slightly outside due to transforms).

---

## Phase 2: Visualization & Testing (Week 3)

### Step 2.1: Modify Fragment Shader

**File**: `src/SDFMesh/CudaSDFUtil.h` (or create new shader file)

Update fragment shader to use UVs:

```glsl
// In fragmentShaderSource, modify main():

void main() {
    // ... existing lighting calculation ...
    
    if (useTexture == 1) {
        // Use UV coordinates for texture sampling
        vec3 texColor = texture(texture1, TexCoord).rgb;
        color = texColor * (diff + 0.2);
    } else {
        // Use SDF color
        color = sdfColor * (diff + 0.2);
    }
    
    // DEBUG: Visualize UVs as colors
    // color = vec3(TexCoord.x, TexCoord.y, 0.0);
    
    FragColor = vec4(color, 1.0);
}
```

### Step 2.2: Create Checkerboard Test Texture

```cpp
GLuint CreateCheckerboardTexture(int size = 256) {
    std::vector<unsigned char> data(size * size * 3);
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int checkX = (x / (size / 8)) % 2;
            int checkY = (y / (size / 8)) % 2;
            unsigned char color = ((checkX ^ checkY) == 0) ? 255 : 0;
            
            int idx = (y * size + x) * 3;
            data[idx + 0] = color;
            data[idx + 1] = color;
            data[idx + 2] = color;
        }
    }
    
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    return tex;
}
```

### Step 2.3: Visual Testing

Test each primitive type individually:

1. **Sphere**: Should show equirectangular checkerboard, seam at back
2. **Box**: Each face should be flat checkerboard
3. **Cylinder**: Checkerboard wraps around, seam at back
4. **Torus**: Double wrapping (ring and tube)

Take screenshots for documentation.

---

## Phase 3: Atlas Mode (Week 4-5)

### Step 3.1: Implement Chart Extraction

**File**: `src/main.cpp` (or create `src/atlas_packer.cpp`)

```cpp
struct UVChart {
    int primitiveID;
    std::vector<uint32_t> triangleIndices;
    float2 uvMin, uvMax;
};

std::vector<UVChart> ExtractCharts(
    const std::vector<float4>& vertices,
    const std::vector<int>& primitiveIDs,
    const std::vector<float2>& uvCoords
) {
    std::map<int, std::vector<uint32_t>> primToTris;
    
    for (size_t i = 0; i < primitiveIDs.size(); i += 3) {
        int primID = primitiveIDs[i];
        if (primID >= 0) {
            primToTris[primID].push_back(i / 3);
        }
    }
    
    std::vector<UVChart> charts;
    for (auto& [primID, tris] : primToTris) {
        UVChart chart;
        chart.primitiveID = primID;
        chart.triangleIndices = tris;
        
        chart.uvMin = make_float2(1e10f, 1e10f);
        chart.uvMax = make_float2(-1e10f, -1e10f);
        
        for (uint32_t triIdx : tris) {
            for (int v = 0; v < 3; ++v) {
                float2 uv = uvCoords[triIdx * 3 + v];
                chart.uvMin.x = fminf(chart.uvMin.x, uv.x);
                chart.uvMin.y = fminf(chart.uvMin.y, uv.y);
                chart.uvMax.x = fmaxf(chart.uvMax.x, uv.x);
                chart.uvMax.y = fmaxf(chart.uvMax.y, uv.y);
            }
        }
        
        charts.push_back(chart);
    }
    
    return charts;
}
```

### Step 3.2: Integrate Rect Packing

Download stb_rect_pack.h and integrate:

```cpp
#define STB_RECT_PACK_IMPLEMENTATION
#include "stb_rect_pack.h"

struct PackedChart {
    int chartIndex;
    float2 atlasMin, atlasMax;
    float2 scale;
};

std::vector<PackedChart> PackChartsIntoAtlas(
    const std::vector<UVChart>& charts,
    int atlasSize = 2048,
    int padding = 4
) {
    stbrp_context context;
    std::vector<stbrp_node> nodes(atlasSize);
    stbrp_init_target(&context, atlasSize, atlasSize, nodes.data(), atlasSize);
    
    std::vector<stbrp_rect> rects(charts.size());
    for (size_t i = 0; i < charts.size(); ++i) {
        float width = (charts[i].uvMax.x - charts[i].uvMin.x);
        float height = (charts[i].uvMax.y - charts[i].uvMin.y);
        
        rects[i].w = (int)(width * atlasSize) + padding * 2;
        rects[i].h = (int)(height * atlasSize) + padding * 2;
        rects[i].id = i;
    }
    
    stbrp_pack_rects(&context, rects.data(), rects.size());
    
    std::vector<PackedChart> packed;
    for (const auto& rect : rects) {
        if (!rect.was_packed) {
            std::cerr << "Warning: Chart " << rect.id << " could not be packed!" << std::endl;
            continue;
        }
        
        PackedChart pc;
        pc.chartIndex = rect.id;
        pc.atlasMin = make_float2(
            (float)(rect.x + padding) / atlasSize,
            (float)(rect.y + padding) / atlasSize
        );
        pc.atlasMax = make_float2(
            (float)(rect.x + rect.w - padding) / atlasSize,
            (float)(rect.y + rect.h - padding) / atlasSize
        );
        pc.scale = make_float2(
            pc.atlasMax.x - pc.atlasMin.x,
            pc.atlasMax.y - pc.atlasMin.y
        );
        
        packed.push_back(pc);
    }
    
    return packed;
}
```

### Step 3.3: Remap UVs

```cpp
void RemapUVsToAtlas(
    std::vector<float2>& uvCoords,
    const std::vector<int>& primitiveIDs,
    const std::vector<UVChart>& charts,
    const std::vector<PackedChart>& packed
) {
    std::map<int, const PackedChart*> primToChart;
    for (const auto& p : packed) {
        const UVChart& chart = charts[p.chartIndex];
        primToChart[chart.primitiveID] = &p;
    }
    
    for (size_t i = 0; i < uvCoords.size(); ++i) {
        int primID = primitiveIDs[i];
        if (primID < 0) continue;
        
        auto it = primToChart.find(primID);
        if (it == primToChart.end()) continue;
        
        const PackedChart& packed = *it->second;
        const UVChart& chart = charts[packed.chartIndex];
        
        float2 uv = uvCoords[i];
        float2 normalized = make_float2(
            (uv.x - chart.uvMin.x) / (chart.uvMax.x - chart.uvMin.x + 1e-6f),
            (uv.y - chart.uvMin.y) / (chart.uvMax.y - chart.uvMin.y + 1e-6f)
        );
        
        uvCoords[i] = make_float2(
            packed.atlasMin.x + normalized.x * packed.scale.x,
            packed.atlasMin.y + normalized.y * packed.scale.y
        );
    }
}
```

### Step 3.4: Integrate into Main Loop

```cpp
// Add key binding for atlas mode
if (key == GLFW_KEY_A) {
    g_triggerAtlasMode = true;
}

// In main loop:
if (g_triggerAtlasMode) {
    g_triggerAtlasMode = false;
    
    // 1. Generate mesh with UVs
    mesh.EnableUVGeneration(true);
    // ... map buffers and generate ...
    
    // 2. Download data
    std::vector<float4> vertices(vertexCount);
    std::vector<float2> uvCoords(vertexCount);
    std::vector<int> primIDs(vertexCount);
    cudaMemcpy(vertices.data(), d_vbo, vertexCount * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(uvCoords.data(), d_uvbo, vertexCount * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(primIDs.data(), d_primIDs, vertexCount * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 3. Extract and pack
    auto charts = ExtractCharts(vertices, primIDs, uvCoords);
    auto packed = PackChartsIntoAtlas(charts, 2048, 4);
    RemapUVsToAtlas(uvCoords, primIDs, charts, packed);
    
    // 4. Upload remapped UVs
    cudaMemcpy(d_uvbo, uvCoords.data(), vertexCount * sizeof(float2), cudaMemcpyHostToDevice);
    
    std::cout << "Atlas packing complete: " << packed.size() << " charts" << std::endl;
}
```

---

## Phase 4: Shader Integration for Direct Mode (Week 6)

### Step 4.1: Update Vertex Shader

**File**: `src/SDFMesh/CudaSDFUtil.h` (or separate shader file)

```glsl
inline const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec4 aPos;
    layout (location = 1) in vec4 aColor;
    layout (location = 2) in vec2 aTexCoord;
    layout (location = 3) in float aPrimitiveID;  // NEW: Primitive ID
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 FragPosWorld;
    out vec2 TexCoord;
    out vec4 VertColor;
    flat out int PrimitiveID;  // NEW: flat = no interpolation
    
    void main() {
        vec4 worldPos = model * vec4(aPos.xyz, 1.0);
        gl_Position = projection * view * worldPos;
        FragPos = aPos.xyz;
        FragPosWorld = worldPos.xyz;
        TexCoord = aTexCoord;
        VertColor = aColor;
        PrimitiveID = int(aPrimitiveID);
    }
)";
```

### Step 4.2: Update Fragment Shader

```glsl
inline const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    in vec3 FragPos;
    in vec2 TexCoord;
    in vec4 VertColor;
    flat in int PrimitiveID;  // NEW
    
    uniform float time;
    uniform samplerBuffer bvhNodes;
    uniform sampler2DArray textureArray;  // NEW: For direct mode
    uniform sampler2D atlasTexture;       // NEW: For atlas mode
    uniform int useTexture;  // 0 = SDF Color, 1 = Texture Array, 2 = Atlas
    
    // ... existing Primitive structs and map() function ...
    
    void main() {
        // ... existing SDF evaluation and lighting ...
        
        vec3 color;
        
        if (useTexture == 1) {
            // DIRECT MODE: Sample from texture array using primitive ID
            vec3 texColor = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
            color = texColor * (diff + 0.2);
        } else if (useTexture == 2) {
            // ATLAS MODE: Sample from single texture with remapped UVs
            vec3 texColor = texture(atlasTexture, TexCoord).rgb;
            color = texColor * (diff + 0.2);
        } else {
            // SDF COLOR MODE: Use vertex colors
            color = VertColor.rgb * (diff + 0.2);
        }
        
        FragColor = vec4(color, 1.0);
    }
)";
```

### Step 4.3: Setup Primitive ID Vertex Attribute

**File**: `src/main.cpp` in `initGL()`

```cpp
// In initGL(), after creating VBO, CBO, UVBO:

// PRIMID BO (float primitive IDs, location 3)
GLuint primidbo;
glGenBuffers(1, &primidbo);
glBindBuffer(GL_ARRAY_BUFFER, primidbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);
glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
// Don't enable yet - will enable after UV generation
// glEnableVertexAttribArray(3);

// Register with CUDA for writing
struct cudaGraphicsResource* cudaPrimIDBO;
cudaError_t errPrimIDBO = cudaGraphicsGLRegisterBuffer(&cudaPrimIDBO, primidbo, 
                                                         cudaGraphicsRegisterFlagsWriteDiscard);
if (errPrimIDBO != cudaSuccess) {
    std::cerr << "CUDA Error Register PrimID BO: " << cudaGetErrorString(errPrimIDBO) << std::endl;
    exit(-1);
}
```

### Step 4.4: Write Primitive IDs in Kernel

**File**: `src/SDFMesh/MarchingCubesKernels.cu`

Modify `generateActiveBlockTriangles` to write primitive IDs as floats:

```cuda
// In the vertex write loop:
if (grid.d_primitiveIDs) {
    // Write as float for OpenGL vertex attribute
    grid.d_primitiveIDs[write + j] = primitiveID;
}
```

**Note**: The kernel already writes `int`, but we'll cast in the shader.

### Step 4.5: Create and Manage Texture Array

**File**: `src/main.cpp` (add new class or functions)

```cpp
class TextureArrayManager {
public:
    TextureArrayManager(int maxTextures = 8, int resolution = 1024) 
        : m_maxTextures(maxTextures), m_resolution(resolution) {
        
        glGenTextures(1, &m_textureArray);
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        
        // Allocate storage
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 
                     m_resolution, m_resolution, m_maxTextures,
                     0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        
        // Set parameters
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
        std::cout << "Created texture array: " << m_maxTextures << " layers, " 
                  << m_resolution << "x" << m_resolution << std::endl;
    }
    
    ~TextureArrayManager() {
        if (m_textureArray) glDeleteTextures(1, &m_textureArray);
    }
    
    int LoadTextureToLayer(const std::string& filename, int layer) {
        if (layer >= m_maxTextures) {
            std::cerr << "Error: Layer " << layer << " exceeds max " << m_maxTextures << std::endl;
            return -1;
        }
        
        int width, height, channels;
        unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
        
        if (!data) {
            std::cerr << "Failed to load texture: " << filename << std::endl;
            return -1;
        }
        
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, 
                        width, height, 1, GL_RGB, GL_UNSIGNED_BYTE, data);
        
        stbi_image_free(data);
        
        m_layerFiles[layer] = filename;
        std::cout << "Loaded '" << filename << "' to layer " << layer << std::endl;
        
        return layer;
    }
    
    void GenerateMipmaps() {
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }
    
    void Bind(int textureUnit = 0) {
        glActiveTexture(GL_TEXTURE0 + textureUnit);
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
    }
    
    GLuint GetTextureID() const { return m_textureArray; }
    
private:
    GLuint m_textureArray = 0;
    int m_maxTextures;
    int m_resolution;
    std::map<int, std::string> m_layerFiles;
};

// Global instance
TextureArrayManager* g_textureArrayManager = nullptr;
```

### Step 4.6: Initialize Texture Array in main()

**File**: `src/main.cpp` in `main()` after primitive creation

```cpp
// Create texture array manager
g_textureArrayManager = new TextureArrayManager(8, 1024);

// Load textures into layers (match with primitive textureID)
g_textureArrayManager->LoadTextureToLayer("textures/metal.jpg", 0);
g_textureArrayManager->LoadTextureToLayer("textures/wood.jpg", 1);
g_textureArrayManager->LoadTextureToLayer("textures/stone.jpg", 2);
g_textureArrayManager->LoadTextureToLayer("textures/fabric.jpg", 3);
g_textureArrayManager->GenerateMipmaps();

// Assign texture layers to primitives
scenePrimitives[0].textureID = 0;  // Torus uses metal
scenePrimitives[1].textureID = 1;  // Hex prism uses wood
scenePrimitives[2].textureID = 2;  // Cone uses stone
scenePrimitives[3].textureID = 3;  // Cylinder uses fabric
scenePrimitives[4].textureID = 0;  // Sphere uses metal

// Note: textureID corresponds to the layer in the texture array
// This mapping is handled automatically by passing primitive ID to shader
```

### Step 4.7: Update Render Loop

**File**: `src/main.cpp` in main render loop

```cpp
// In render loop, before drawing:

glUseProgram(shaderProgram);

// ... existing uniform setup ...

// Bind texture array
g_textureArrayManager->Bind(0);  // Texture unit 0
glUniform1i(glGetUniformLocation(shaderProgram, "textureArray"), 0);

// Set rendering mode
if (g_useTextureArray && g_isUnwrapped) {
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);  // Texture array mode
} else if (g_hasProjectedTexture) {
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 2);  // Atlas mode
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_bakedTextureID);
    glUniform1i(glGetUniformLocation(shaderProgram, "atlasTexture"), 1);
} else {
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);  // SDF color mode
}

// Draw
glBindVertexArray(vao);
if (g_isUnwrapped) {
    glDrawElements(GL_TRIANGLES, (GLsizei)g_indexCount, GL_UNSIGNED_INT, 0);
} else {
    glDrawArrays(GL_TRIANGLES, 0, mesh.GetTotalVertexCount());
}
```

### Step 4.8: Add Key Binding for Texture Array Mode

**File**: `src/main.cpp` in `key_callback()`

```cpp
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_W) {
            static bool wireframe = false;
            wireframe = !wireframe;
            glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
        }
        if (key == GLFW_KEY_U) {
            g_triggerUnwrap = true;
        }
        if (key == GLFW_KEY_P) {
            g_triggerProjection = true;
        }
        if (key == GLFW_KEY_T) {  // NEW: Toggle texture array mode
            g_useTextureArray = !g_useTextureArray;
            std::cout << "Texture array mode: " << (g_useTextureArray ? "ON" : "OFF") << std::endl;
        }
        if (key == GLFW_KEY_A) {  // NEW: Atlas mode
            g_triggerAtlasMode = true;
        }
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, true);
        }
    }
}
```

---

## Phase 5: Polish (Week 7)

### Seam Fixing

Add seam detection to UV generation:

```cuda
__device__ void fixUVSeams(float2& uv0, float2& uv1, float2& uv2) {
    const float threshold = 0.5f;
    
    if (fabsf(uv1.x - uv0.x) > threshold) {
        if (uv1.x > uv0.x) uv1.x -= 1.0f;
        else uv1.x += 1.0f;
    }
    if (fabsf(uv2.x - uv0.x) > threshold) {
        if (uv2.x > uv0.x) uv2.x -= 1.0f;
        else uv2.x += 1.0f;
    }
    
    if (fabsf(uv1.y - uv0.y) > threshold) {
        if (uv1.y > uv0.y) uv1.y -= 1.0f;
        else uv1.y += 1.0f;
    }
    if (fabsf(uv2.y - uv0.y) > threshold) {
        if (uv2.y > uv0.y) uv2.y -= 1.0f;
        else uv2.y += 1.0f;
    }
}
```

Call after computing triangle UVs.

---

## Testing Checklist

- [ ] Single sphere with checkerboard texture
- [ ] Single box with checkerboard texture
- [ ] Single cylinder with checkerboard texture
- [ ] Single torus with checkerboard texture
- [ ] Multiple primitives (union) - correct UV assignment
- [ ] Subtraction operation - interior uses cutter's UVs
- [ ] Twisted primitive - UVs remain stable
- [ ] Bent primitive - UVs remain stable
- [ ] Atlas packing with 2+ primitives
- [ ] Projection baking onto atlas
- [ ] No memory leaks (valgrind/cuda-memcheck)
- [ ] Performance < 30% overhead

---

## Debugging Tips

1. **Visualize UVs as colors**: `color = vec3(TexCoord.x, TexCoord.y, 0.0);`
2. **Check primitive IDs**: Print first 100 to verify assignment
3. **Verify local positions**: Should be in primitive's local space
4. **Test with one primitive at a time**: Isolate issues
5. **Use cuda-gdb**: Step through kernel execution

---

## Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| UVs all (0,0) | Buffer not allocated | Check d_uvCoords != nullptr |
| Black texture | UVs out of range | Clamp UVs or check clamping mode |
| Seams visible | Wrap-around not handled | Apply seam fixing |
| Wrong primitive UVs | Primitive ID incorrect | Debug map() function |
| Crash in kernel | Null pointer | Check all grid pointers |

---

## Next Steps After Implementation

1. Performance profiling and optimization
2. Add more UV modes (triplanar, planar)
3. Support for multiple textures per scene
4. Export mesh with UVs to OBJ/FBX
5. GUI for adjusting UV parameters
6. Animated UV transforms



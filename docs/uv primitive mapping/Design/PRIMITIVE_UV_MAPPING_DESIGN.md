# Primitive-Based UV Mapping System - Design Document

## Executive Summary

This document outlines a comprehensive design for adding per-primitive UV mapping capabilities to the CudaSDF marching cubes mesh generation system. The goal is to enable each SDF primitive to define its own UV parameterization, which will be propagated through operations (union/subtract/intersect) and distortions, ultimately appearing on the final extracted mesh surface.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Design Goals](#design-goals)
3. [Feature Overview](#feature-overview)
4. [Technical Architecture](#technical-architecture)
5. [Implementation Details](#implementation-details)
6. [Challenges & Solutions](#challenges--solutions)
7. [API Design](#api-design)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Testing Strategy](#testing-strategy)
10. [Future Enhancements](#future-enhancements)

---

## 1. Problem Statement

### Current Limitations
- Generated mesh has no UV coordinates
- UV unwrapping is a post-process that treats the mesh as a single surface
- No relationship between primitive definitions and final UV layout
- Texture mapping requires separate unwrap step with unpredictable results

### Desired Capabilities
- Each primitive should define its own UV space
- UV coordinates should be generated during marching cubes extraction
- Support for texture projection and atlas packing
- Handle operations (subtract/intersect) gracefully
- Account for distortions (twist, bend, etc.)

---

## 2. Design Goals

### Primary Goals
1. **Per-Primitive UV Control**: Each primitive can specify its own UV parameterization
2. **Operation Awareness**: Properly handle UVs when combining primitives
3. **Distortion Handling**: UV coordinates adjust for twist, bend, and other distortions
4. **Atlas Mode**: Optional packing of primitive UVs into a unified texture atlas
5. **Performance**: Minimal overhead during marching cubes generation

### Secondary Goals
1. Texture assignment per primitive
2. UV transform controls (scale, offset, rotation)
3. Seamless blending at operation boundaries
4. Support for both procedural and texture-based materials

---

## 3. Feature Overview

### 3.1 Per-Primitive UV Parameterization

Each primitive type will have a canonical UV mapping:

| Primitive Type | UV Mapping Strategy |
|----------------|-------------------|
| **Sphere** | Spherical/equirectangular (θ, φ) |
| **Box** | Cubic projection (6 faces, auto-select based on normal) |
| **Cylinder** | Cylindrical (wrap angle, height) |
| **Torus** | Toroidal (major angle, minor angle) |
| **Cone** | Radial + height |
| **Capsule** | Cylindrical with spherical caps |
| **Hex Prism** | Planar projection on hexagonal space |
| **Rounded variants** | Inherit from base shape |

### 3.2 UV Transform Controls

Each primitive will support:
- **UV Scale**: `float2 uvScale` - scale UV coordinates
- **UV Offset**: `float2 uvOffset` - translate UV coordinates
- **UV Rotation**: `float uvRotation` - rotate UVs (in radians)
- **Texture ID**: `int textureID` - reference to texture (for multi-texture support)
- **UV Mode**: `enum UVMode { PRIMITIVE, WORLD_TRIPLANAR, CUSTOM }`

### 3.3 Rendering Modes

Two operating modes:

#### **Direct Mode with Texture Array** (Default for Multi-Primitive)
- Each primitive's UVs map directly to [0,1] range
- Multiple textures handled via 2D texture array (`sampler2DArray`)
- Primitive ID passed to fragment shader to select texture layer
- No packing required
- Best for: Multiple primitives with different materials
- GPU Support: OpenGL 3.0+, widely supported

#### **Atlas Mode** (Advanced/Projection Baking)
- Each primitive's exposed surface becomes an island
- Islands are packed into unified [0,1] UV space
- All primitives share a single 2D texture
- Requires primitive ID tracking during marching cubes
- Post-process chart extraction and packing
- Best for: Texture projection baking, single texture workflow

### 3.4 Operation Handling

**Union**: Use UV from closest primitive (smallest SDF value)

**Subtract**: 
- Interior surfaces use the subtracted primitive's UVs
- Exterior surfaces use the base primitive's UVs

**Intersect**: Use UV from primitive with larger SDF value (defines the surface)

**Blended Operations**: Interpolate UVs based on blend factor `h`

---

## 4. Technical Architecture

### 4.1 Data Structure Extensions

#### Extended SDFPrimitive Structure

```cpp
struct SDFPrimitive {
    // ... existing fields ...
    
    // UV Parameters (add 32 bytes)
    float2 uvScale;        // 8 bytes
    float2 uvOffset;       // 8 bytes
    float uvRotation;      // 4 bytes
    int uvMode;            // 4 bytes (UVMode enum)
    int textureID;         // 4 bytes
    int atlasIslandID;     // 4 bytes (assigned during atlas packing, -1 if not used)
    
    // Padding to maintain alignment
    // Total struct size should remain multiple of 16
};
```

#### New UV Modes Enum

```cpp
enum UVMode {
    UV_PRIMITIVE = 0,       // Use primitive's canonical mapping
    UV_WORLD_TRIPLANAR,     // World-space triplanar projection
    UV_CUSTOM,              // Reserved for future custom UV functions
    UV_PLANAR_X,            // Planar projection along X axis
    UV_PLANAR_Y,            // Planar projection along Y axis
    UV_PLANAR_Z             // Planar projection along Z axis
};
```

#### Extended SDFGrid Structure

```cpp
struct SDFGrid {
    // ... existing fields ...
    
    // UV Output
    float2* d_uvCoords;     // Output UV coordinates (parallel to d_vertices)
    int* d_primitiveIDs;    // Which primitive each vertex came from
};
```

### 4.2 Kernel Modifications

#### Modified `map()` Function Signature

The SDF evaluation function needs to return additional information:

```cuda
__device__ void map(
    float3 p_world, 
    const SDFGrid& grid, 
    float time, 
    float& outDist,           // SDF distance
    float3& outColor,         // Color
    int& outPrimitiveID,      // NEW: which primitive
    float3& outLocalPos       // NEW: position in primitive local space
);
```

#### UV Generation Function

New device function to compute UVs from primitive local coordinates:

```cuda
__device__ float2 computePrimitiveUV(
    const SDFPrimitive& prim,
    float3 localPos,          // Position in primitive's local space
    float3 localNormal        // Normal in primitive's local space (approximated)
) {
    float2 uv = make_float2(0.0f, 0.0f);
    
    switch(prim.type) {
        case SDF_SPHERE:
            uv = uvSphere(localPos);
            break;
        case SDF_BOX:
            uv = uvBox(localPos, localNormal);
            break;
        case SDF_CYLINDER:
            uv = uvCylinder(localPos, prim.params[0]);
            break;
        case SDF_TORUS:
            uv = uvTorus(localPos, prim.params[0], prim.params[1]);
            break;
        // ... other cases ...
    }
    
    // Apply UV transforms
    uv = transformUV(uv, prim.uvScale, prim.uvOffset, prim.uvRotation);
    
    return uv;
}
```

#### Individual UV Mapping Functions

```cuda
__device__ float2 uvSphere(float3 p) {
    // Spherical coordinates
    float theta = atan2(p.z, p.x);           // Azimuth [-π, π]
    float phi = asin(clamp(p.y / length(p), -1.0f, 1.0f)); // Elevation [-π/2, π/2]
    
    return make_float2(
        (theta + M_PI) / (2.0f * M_PI),      // u: [0, 1]
        (phi + M_PI_2) / M_PI                // v: [0, 1]
    );
}

__device__ float2 uvBox(float3 p, float3 n) {
    // Select face based on dominant normal direction
    float3 absN = make_float3(fabsf(n.x), fabsf(n.y), fabsf(n.z));
    
    if (absN.x > absN.y && absN.x > absN.z) {
        // X-dominant face
        return make_float2(p.z * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
    } else if (absN.y > absN.z) {
        // Y-dominant face
        return make_float2(p.x * 0.5f + 0.5f, p.z * 0.5f + 0.5f);
    } else {
        // Z-dominant face
        return make_float2(p.x * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
    }
}

__device__ float2 uvCylinder(float3 p, float height) {
    float theta = atan2(p.z, p.x);
    float u = (theta + M_PI) / (2.0f * M_PI);
    float v = (p.y / height) * 0.5f + 0.5f;  // Normalize by height
    return make_float2(u, v);
}

__device__ float2 uvTorus(float3 p, float majorRadius, float minorRadius) {
    // Major angle (around the ring)
    float theta = atan2(p.z, p.x);
    
    // Minor angle (around the tube)
    float2 q = make_float2(length(make_float2(p.x, p.z)) - majorRadius, p.y);
    float phi = atan2(q.y, q.x);
    
    return make_float2(
        (theta + M_PI) / (2.0f * M_PI),
        (phi + M_PI) / (2.0f * M_PI)
    );
}

__device__ float2 transformUV(float2 uv, float2 scale, float2 offset, float rotation) {
    // Apply scale
    uv = uv * scale;
    
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
```

### 4.3 Modified Marching Cubes Kernels

#### Updated `generateActiveBlockTriangles` Kernel

```cuda
__global__ void generateActiveBlockTriangles(SDFGrid grid, float time) {
    // ... existing voxel and triangle extraction code ...
    
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float3 p = pTri[j];

        float dist; 
        float3 color;
        int primitiveID;
        float3 localPos;
        
        // Extended map call
        map(p, grid, time, dist, color, primitiveID, localPos);

        grid.d_vertices[write + j] = make_float4(p.x, p.y, p.z, 1.0f);
        grid.d_vertexColors[write + j] = make_float4(color.x, color.y, color.z, 1.0f);
        
        // NEW: Store primitive ID
        if (grid.d_primitiveIDs) {
            grid.d_primitiveIDs[write + j] = primitiveID;
        }
        
        // NEW: Compute and store UVs
        if (grid.d_uvCoords && primitiveID >= 0) {
            SDFPrimitive prim = grid.d_primitives[primitiveID];
            
            // Approximate local normal (could be refined)
            float3 localNormal = computeLocalNormal(grid, p, time, prim);
            
            float2 uv = computePrimitiveUV(prim, localPos, localNormal);
            grid.d_uvCoords[write + j] = uv;
        }
    }
    
    write += 3;
}
```

### 4.4 Atlas Packing System

#### Chart Extraction

After marching cubes generation, identify contiguous regions per primitive:

```cpp
struct UVChart {
    int primitiveID;
    std::vector<uint32_t> triangleIndices;  // Triangle indices in this chart
    float2 uvMin, uvMax;                     // Bounding box in primitive UV space
};

std::vector<UVChart> ExtractCharts(
    const std::vector<float4>& vertices,
    const std::vector<int>& primitiveIDs,
    const std::vector<float2>& uvCoords
) {
    // Group triangles by primitive ID
    std::map<int, std::vector<uint32_t>> primToTris;
    
    for (size_t i = 0; i < primitiveIDs.size(); i += 3) {
        int primID = primitiveIDs[i];  // Assume all 3 vertices of a triangle share primitive ID
        if (primID >= 0) {
            primToTris[primID].push_back(i / 3);
        }
    }
    
    // Create charts (could be further subdivided for disconnected regions)
    std::vector<UVChart> charts;
    for (auto& [primID, tris] : primToTris) {
        UVChart chart;
        chart.primitiveID = primID;
        chart.triangleIndices = tris;
        
        // Compute UV bounds
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

#### Rect Packing Algorithm

Use a simple rect-packing algorithm to pack UV islands:

```cpp
struct PackedChart {
    int chartIndex;
    float2 atlasMin;    // Position in atlas [0,1]
    float2 atlasMax;
    float2 scale;       // Scale factor from primitive UV to atlas UV
};

std::vector<PackedChart> PackChartsIntoAtlas(
    const std::vector<UVChart>& charts,
    int atlasSize = 2048,
    int padding = 4
) {
    // Use a simple shelf packing or more sophisticated algorithm
    // Returns mapping from chart to atlas position
    
    // This is a simplified placeholder
    std::vector<PackedChart> packed;
    
    // TODO: Implement rect packing (could use stb_rect_pack or custom)
    
    return packed;
}
```

#### UV Remapping

After packing, remap all UVs to atlas space:

```cpp
void RemapUVsToAtlas(
    std::vector<float2>& uvCoords,
    const std::vector<int>& primitiveIDs,
    const std::vector<UVChart>& charts,
    const std::vector<PackedChart>& packed
) {
    // Build lookup from primitive ID to packed chart
    std::map<int, const PackedChart*> primToChart;
    for (const auto& p : packed) {
        const UVChart& chart = charts[p.chartIndex];
        primToChart[chart.primitiveID] = &p;
    }
    
    // Remap each UV
    for (size_t i = 0; i < uvCoords.size(); ++i) {
        int primID = primitiveIDs[i];
        if (primID < 0) continue;
        
        auto it = primToChart.find(primID);
        if (it == primToChart.end()) continue;
        
        const PackedChart& packed = *it->second;
        const UVChart& chart = charts[packed.chartIndex];
        
        // Normalize UV to [0,1] within chart bounds
        float2 uv = uvCoords[i];
        float2 normalized = make_float2(
            (uv.x - chart.uvMin.x) / (chart.uvMax.x - chart.uvMin.x),
            (uv.y - chart.uvMin.y) / (chart.uvMax.y - chart.uvMin.y)
        );
        
        // Map to atlas position
        uvCoords[i] = make_float2(
            packed.atlasMin.x + normalized.x * (packed.atlasMax.x - packed.atlasMin.x),
            packed.atlasMin.y + normalized.y * (packed.atlasMax.y - packed.atlasMin.y)
        );
    }
}
```

### 4.5 Shader Integration for Direct Mode

#### Vertex Shader

```glsl
#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in float aPrimitiveID;  // NEW: Primitive ID as vertex attribute

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 FragPosWorld;
out vec2 TexCoord;
flat out int PrimitiveID;  // NEW: Pass to fragment shader (flat = no interpolation)

void main() {
    vec4 worldPos = model * vec4(aPos.xyz, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = aPos.xyz;
    FragPosWorld = worldPos.xyz;
    TexCoord = aTexCoord;
    PrimitiveID = int(aPrimitiveID);  // NEW: Convert float to int
}
```

#### Fragment Shader (Direct Mode with Texture Array)

```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;
flat in int PrimitiveID;  // NEW: Received from vertex shader

uniform float time;
uniform samplerBuffer bvhNodes;
uniform sampler2DArray textureArray;  // NEW: 2D texture array for multiple textures
uniform int useTexture;  // 0 = SDF Color, 1 = Texture Array, 2 = Atlas texture
uniform sampler2D atlasTexture;  // NEW: For atlas mode

// ... existing Primitive struct and uniforms ...

void main() {
    // ... existing SDF evaluation and lighting ...
    
    vec3 color;
    
    if (useTexture == 1) {
        // DIRECT MODE: Use texture array with primitive ID as layer
        vec3 texColor = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
        color = texColor * (diff + 0.2);
    } else if (useTexture == 2) {
        // ATLAS MODE: Use single texture with remapped UVs
        vec3 texColor = texture(atlasTexture, TexCoord).rgb;
        color = texColor * (diff + 0.2);
    } else {
        // NO TEXTURE: Use SDF color
        color = sdfColor * (diff + 0.2);
    }
    
    FragColor = vec4(color, 1.0);
}
```

#### OpenGL Setup for Texture Array

```cpp
// Create texture array
GLuint textureArray;
glGenTextures(1, &textureArray);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);

// Allocate storage for N textures
int numTextures = 8;  // Max number of different primitive textures
int texWidth = 1024;
int texHeight = 1024;
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, texWidth, texHeight, numTextures, 
             0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

// Set texture parameters
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);

// Load each texture into a layer
for (int i = 0; i < numTextures; ++i) {
    unsigned char* data = LoadTextureData(textureFilenames[i], &width, &height);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, width, height, 1,
                    GL_RGB, GL_UNSIGNED_BYTE, data);
    FreeTextureData(data);
}

glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

// In render loop:
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glUniform1i(glGetUniformLocation(shaderProgram, "textureArray"), 0);
glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);  // Direct mode
```

#### Texture ID Management

```cpp
// Map primitive index to texture layer
class TextureManager {
public:
    void AssignTextureToLayer(const std::string& filename, int layer) {
        m_textureFiles[layer] = filename;
    }
    
    void SetPrimitiveTexture(SDFPrimitive& prim, const std::string& textureName) {
        // Find or assign layer for this texture
        int layer = GetOrCreateLayer(textureName);
        prim.textureID = layer;
    }
    
private:
    std::map<int, std::string> m_textureFiles;
    std::map<std::string, int> m_textureToLayer;
    int m_nextLayer = 0;
    
    int GetOrCreateLayer(const std::string& name) {
        auto it = m_textureToLayer.find(name);
        if (it != m_textureToLayer.end()) {
            return it->second;
        }
        
        int layer = m_nextLayer++;
        m_textureToLayer[name] = layer;
        m_textureFiles[layer] = name;
        return layer;
    }
};
```

---

## 5. Implementation Details

### 5.1 Primitive Configuration API

```cpp
// Example: Configure a sphere with custom UVs
SDFPrimitive sphere = CreateSpherePrim(...);
sphere.uvScale = make_float2(2.0f, 1.0f);    // Tile 2x horizontally
sphere.uvOffset = make_float2(0.0f, 0.0f);
sphere.uvRotation = 0.0f;
sphere.uvMode = UV_PRIMITIVE;
sphere.textureID = 0;
```

### 5.2 Modified CudaSDFMesh Interface

```cpp
class CudaSDFMesh {
public:
    // ... existing methods ...
    
    // NEW: Enable UV generation
    void EnableUVGeneration(bool enable);
    
    // NEW: Set atlas mode
    void SetAtlasMode(bool atlasMode);
    
    // NEW: Update with UV output
    void Update(
        float time, 
        float4* d_outVertices, 
        float4* d_outColors, 
        unsigned int* d_outIndices,
        float2* d_outUVs,           // NEW: UV output
        int* d_outPrimitiveIDs      // NEW: Primitive ID output
    );
    
    // NEW: Get UV data for CPU-side atlas packing
    void GetUVData(
        std::vector<float2>& uvCoords,
        std::vector<int>& primitiveIDs
    );
    
    // NEW: Upload remapped UVs after atlas packing
    void UploadRemappedUVs(const std::vector<float2>& uvCoords);
    
private:
    bool m_generateUVs = false;
    bool m_atlasMode = false;
};
```

### 5.3 Main Application Flow

```cpp
// In main loop:
if (g_triggerAtlasUnwrap) {
    g_triggerAtlasUnwrap = false;
    
    // 1. Generate mesh with primitive UVs
    mesh.EnableUVGeneration(true);
    mesh.SetAtlasMode(true);
    mesh.Update(time, d_vbo, d_cbo, d_ibo, d_uvbo, d_primIDs);
    
    // 2. Download UV and primitive ID data
    std::vector<float2> uvCoords;
    std::vector<int> primIDs;
    mesh.GetUVData(uvCoords, primIDs);
    
    // 3. Extract charts per primitive
    std::vector<UVChart> charts = ExtractCharts(vertices, primIDs, uvCoords);
    
    // 4. Pack charts into atlas
    std::vector<PackedChart> packed = PackChartsIntoAtlas(charts);
    
    // 5. Remap UVs to atlas space
    RemapUVsToAtlas(uvCoords, primIDs, charts, packed);
    
    // 6. Upload remapped UVs
    mesh.UploadRemappedUVs(uvCoords);
    
    g_isUnwrapped = true;
}
```

---

## 6. Challenges & Solutions

### Challenge 1: Primitive Identification at Blend Zones

**Problem**: When operations use smooth blending, multiple primitives contribute to a surface point. Which primitive's UVs should be used?

**Solution**: 
- Track the "dominant" primitive (one with smallest absolute SDF value after blending)
- Store blend factor `h` and use it to interpolate UVs if needed
- For atlas mode, assign blended regions to the dominant primitive's chart

### Challenge 2: Distortion Handling

**Problem**: Twist, bend, and other distortions modify vertex positions but UVs should remain stable in pre-distortion space.

**Solution**:
- Compute local position BEFORE applying distortion
- Use pre-distortion local position for UV calculation
- Store both pre- and post-distortion positions in `map()` call

Implementation:
```cuda
__device__ void map(...) {
    // Transform to local space
    float3 p_local = worldToLocal(p_world, prim);
    
    // Store pre-distortion position for UV calculation
    float3 p_preDistortion = p_local;
    
    // Apply distortion
    if (prim.displacement == DISP_TWIST) {
        p_local = dispTwist(p_local, prim.dispParams[0]);
    }
    
    // Evaluate SDF with distorted position
    float d_prim = sdSphere(p_local, prim.params[0]);
    
    // ... but use pre-distortion position for UV ...
    outLocalPos = p_preDistortion;  // This is what gets passed to UV computation
}
```

### Challenge 3: Subtraction UV Handling

**Problem**: Subtracted primitives create interior surfaces that should use the cutting primitive's UVs, but exterior surfaces should use the base primitive's UVs.

**Solution**:
- In `map()`, track whether current surface is from subtraction
- If `d = max(d1, -d2)` results in `-d2` being larger, mark primitive ID as the subtracted primitive
- Use sign of SDF value to determine inside vs outside

```cuda
case SDF_SUBTRACT:
    float d_new = opSubtract(d, d_prim);
    if (d_new > d) {
        // Subtracted primitive is defining the surface
        outPrimitiveID = i;
        outLocalPos = p_local;
    }
    d = d_new;
    break;
```

### Challenge 4: Performance Impact

**Problem**: Additional computations (local normal, UV mapping) could slow down marching cubes.

**Solution**:
- Make UV generation optional (disabled by default)
- Use `__device__ __forceinline__` for UV functions
- Share computation between color and UV (both need primitive ID)
- Consider computing UVs only for final vertex write, not during SDF evaluation

### Challenge 5: UV Seams and Discontinuities

**Problem**: Spherical and cylindrical UVs have seams (e.g., θ wrapping from 2π to 0).

**Solution**:
- Detect UV discontinuities between triangle vertices
- If UV jump is > 0.5, adjust by adding/subtracting 1.0 to maintain continuity
- For atlas mode, may need to duplicate vertices along seams

```cuda
__device__ void fixUVSeams(float2& uv0, float2& uv1, float2& uv2) {
    // Check for seam crossings
    if (fabsf(uv1.x - uv0.x) > 0.5f) {
        if (uv1.x > uv0.x) uv1.x -= 1.0f;
        else uv1.x += 1.0f;
    }
    if (fabsf(uv2.x - uv0.x) > 0.5f) {
        if (uv2.x > uv0.x) uv2.x -= 1.0f;
        else uv2.x += 1.0f;
    }
    // Repeat for Y if needed
}
```

---

## 7. API Design

### 7.1 User-Facing API

```cpp
// Create primitive with UV configuration
SDFPrimitive CreateSphereWithUV(
    float3 position,
    float radius,
    float2 uvScale = make_float2(1.0f, 1.0f),
    float2 uvOffset = make_float2(0.0f, 0.0f),
    float uvRotation = 0.0f,
    UVMode uvMode = UV_PRIMITIVE
);

// Enable/disable UV generation
mesh.SetUVGenerationEnabled(true);

// Choose mode
mesh.SetUVMode(UVAtlasMode::DIRECT);  // or UVAtlasMode::PACKED

// Generate mesh with UVs
mesh.Update(time, d_vertices, d_colors, d_indices, d_uvs, d_primIDs);

// Optional: Perform atlas packing
if (atlasMode) {
    AtlasPackingResult result = mesh.PackUVAtlas(2048, 4);  // size, padding
    // UVs are automatically remapped
}
```

### 7.2 Enums and Constants

```cpp
enum class UVAtlasMode {
    DIRECT,      // Each primitive maps to [0,1]
    PACKED       // Pack all primitives into single [0,1] atlas
};

enum UVMode {
    UV_PRIMITIVE = 0,
    UV_WORLD_TRIPLANAR,
    UV_PLANAR_X,
    UV_PLANAR_Y,
    UV_PLANAR_Z,
    UV_CUSTOM
};
```

---

## 8. Implementation Roadmap

### Phase 1: Core UV Generation (1-2 weeks)
1. Extend `SDFPrimitive` struct with UV parameters
2. Implement UV mapping functions for all primitive types
3. Modify `map()` to output primitive ID and local position
4. Update marching cubes kernels to generate UVs
5. Test with single primitives in direct mode

### Phase 2: Operation Handling (1 week)
1. Implement proper primitive ID selection for union/subtract/intersect
2. Handle blended operations with UV interpolation
3. Test with multiple primitives and operations
4. Verify UV continuity across boundaries

### Phase 3: Distortion Support (1 week)
1. Ensure pre-distortion positions are used for UV calculation
2. Test with twist, bend, and other distortions
3. Verify UVs remain stable under animation

### Phase 4: Shader Integration for Direct Mode (1 week)
1. Add primitive ID as vertex attribute
2. Update vertex shader to pass primitive ID (flat)
3. Update fragment shader to support texture arrays
4. Implement TextureArrayManager class
5. Load textures to array layers
6. Test multi-primitive rendering with different textures

### Phase 5: Atlas Packing (1-2 weeks)
1. Implement chart extraction from primitive IDs
2. Integrate rect packing algorithm (or use stb_rect_pack)
3. Implement UV remapping to atlas space
4. Handle UV seam duplication if needed
5. Test with complex multi-primitive scenes

### Phase 6: Polish & Optimization (1 week)
1. Performance profiling and optimization
2. Handle edge cases (empty charts, overlapping UVs)
3. Add debug visualization modes
4. Documentation and examples

**Total Estimated Time: 6-8 weeks**

---

## 9. Testing Strategy

### Unit Tests
- Each UV mapping function (sphere, cylinder, etc.)
- UV transform application
- Primitive ID propagation through operations

### Integration Tests
- Single primitive mesh generation with UVs
- Multi-primitive union with correct UV assignment
- Subtraction with interior/exterior UV differentiation
- Blended operations with UV interpolation
- Distorted primitives maintain correct UVs

### Visual Tests
- Checkerboard texture on each primitive type
- UV debugging visualization (display UV as RGB)
- Atlas packing visualization
- Seam detection (look for texture discontinuities)

### Performance Tests
- Benchmark with/without UV generation
- Measure overhead of primitive ID tracking
- Profile atlas packing performance

---

## 10. Future Enhancements

### Advanced Features
1. **Triplanar Mapping**: World-space triplanar projection for organic shapes
2. **Custom UV Functions**: User-defined UV generation via callbacks
3. **Multi-Texture Support**: Different textures per primitive with automatic blending
4. **UV Animation**: Animate UV transforms over time
5. **Procedural Textures**: Integrate noise functions in primitive space

### Optimizations
1. **Warp-Level UV Computation**: Use shuffle operations to share UV calculations
2. **Cached Primitive Data**: Store frequently accessed primitive params in shared memory
3. **LOD-Based UV Resolution**: Adjust UV precision based on distance
4. **GPU Atlas Packing**: Move rect packing to CUDA for real-time updates

### Usability
1. **Preset UV Configurations**: Library of common UV setups (brick, tile, wood grain)
2. **Visual UV Editor**: GUI for adjusting UV parameters with live preview
3. **UV Seam Minimization**: Automatic seam placement for best results
4. **Export Support**: Save mesh+UVs to OBJ, FBX, glTF

---

## Appendix A: Memory Requirements

### Per-Vertex Data
- Position: 16 bytes (float4)
- Color: 16 bytes (float4)
- UV: 8 bytes (float2)
- Primitive ID: 4 bytes (int)
- **Total per vertex: 44 bytes** (vs 32 bytes without UVs)

### For 20M vertices:
- Without UVs: ~640 MB
- With UVs: ~880 MB
- **Additional cost: ~240 MB (~37.5% increase)**

### Per-Primitive Data
- Current: 128 bytes
- With UV params: ~160 bytes (+32 bytes)
- **Marginal impact** (typically < 100 primitives)

---

## Appendix B: Alternative Approaches Considered

### 1. Post-Process UV Generation
**Rejected**: Requires analyzing final mesh topology, loses primitive information, computationally expensive.

### 2. Per-Triangle Primitive Tracking
**Rejected**: Storage intensive, complicates rendering, harder to interpolate.

### 3. Texture Array Instead of Atlas
**Considered**: Avoids packing overhead, but limited by hardware (max texture layers), harder for projection baking.

### 4. Vertex Shader UV Generation
**Rejected**: Requires passing primitive data to shaders, doesn't help with projection baking, less flexible.

---

## Appendix C: Code Snippets

### Example: Creating a UV-Mapped Scene

```cpp
// Torus with tiled texture
SDFPrimitive torus = CreateTorusPrim(
    make_float3(-0.5f, 0.5f, 0.0f),
    make_float4(0.0f, 0.0f, 0.0f, 1.0f),
    make_float3(0.2f, 0.8f, 0.2f),
    SDF_UNION,
    0.0f,
    0.4f,
    0.15f
);
torus.uvScale = make_float2(4.0f, 2.0f);  // Tile 4x2
torus.uvMode = UV_PRIMITIVE;

// Sphere with single texture
SDFPrimitive sphere = CreateSpherePrim(
    make_float3(0.0f, 0.0f, 0.0f),
    make_float4(0.0f, 0.0f, 0.0f, 1.0f),
    make_float3(0.5f, 0.0f, 0.5f),
    SDF_SUBTRACT,
    0.0f,
    0.6f
);
sphere.uvScale = make_float2(1.0f, 1.0f);
sphere.uvMode = UV_PRIMITIVE;

mesh.AddPrimitive(torus);
mesh.AddPrimitive(sphere);

// Enable UV generation
mesh.SetUVGenerationEnabled(true);

// Choose atlas mode
mesh.SetUVMode(UVAtlasMode::PACKED);

// Update mesh
mesh.Update(time, d_vertices, d_colors, d_indices, d_uvs, d_primIDs);

// Pack atlas
mesh.PackUVAtlas(2048, 4);
```

---

## Conclusion

This design provides a comprehensive foundation for implementing primitive-based UV mapping in the CudaSDF system. The approach balances flexibility, performance, and usability while maintaining compatibility with the existing architecture.

**Key Benefits:**
- Maintains primitive-to-surface relationship
- Predictable UV layouts per primitive
- Handles operations and distortions gracefully
- Supports both direct and atlas modes
- Enables texture projection workflows

**Next Steps:**
1. Review and refine this design document
2. Prototype core UV generation (Phase 1)
3. Iterate based on testing results
4. Expand to full implementation



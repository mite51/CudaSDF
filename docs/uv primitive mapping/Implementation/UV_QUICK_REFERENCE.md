# UV Mapping Quick Reference Card

## Function Reference

### UV Mapping Functions (Device)

```cuda
// Compute UV for a primitive
__device__ float2 computePrimitiveUV(
    const SDFPrimitive& prim,
    float3 localPos,
    float time
);

// Individual primitive mappings
__device__ float2 uvSphere(float3 p);
__device__ float2 uvBox(float3 p, float3 normal);
__device__ float2 uvCylinder(float3 p, float height);
__device__ float2 uvTorus(float3 p, float majorRadius, float minorRadius);
__device__ float2 uvCapsule(float3 p, float height, float radius);
__device__ float2 uvCone(float3 p, float height);

// Transform UVs
__device__ float2 transformUV(float2 uv, float2 scale, float2 offset, float rotation);
```

### API Functions (Host)

```cpp
// Update mesh with UV generation
mesh.Update(time, d_vertices, d_colors, d_indices, 
            d_uvs,          // Can be nullptr to disable
            d_primitiveIDs  // Can be nullptr to disable
);
```

---

## Structure Fields

### SDFPrimitive UV Fields

```cpp
float2 uvScale;        // UV tiling (default: 1, 1)
float2 uvOffset;       // UV translation (default: 0, 0)
float uvRotation;      // Rotation in radians (default: 0)
int uvMode;            // UV_PRIMITIVE, UV_WORLD_TRIPLANAR, etc.
int textureID;         // Texture layer index (default: 0)
int atlasIslandID;     // Atlas island ID (default: -1)
```

### SDFGrid UV Fields

```cpp
float2* d_uvCoords;        // Output UV coordinates
int* d_primitiveIDs;       // Primitive ID per vertex
```

---

## Enums

```cpp
enum UVMode {
    UV_PRIMITIVE = 0,       // Use primitive's canonical mapping
    UV_WORLD_TRIPLANAR,     // World-space triplanar
    UV_PLANAR_X,            // Planar X
    UV_PLANAR_Y,            // Planar Y
    UV_PLANAR_Z             // Planar Z
};
```

---

## Shader Uniforms

```glsl
uniform sampler2DArray textureArray;  // For direct mode
uniform sampler2D texture1;           // For atlas/single texture
uniform int useTexture;               // 0=SDF, 1=Array, 2=Single
```

### Shader Inputs/Outputs

```glsl
// Vertex shader
layout (location = 3) in float aPrimitiveID;
flat out int PrimitiveID;

// Fragment shader
flat in int PrimitiveID;
in vec2 TexCoord;
```

---

## UV Mapping Strategies by Primitive

| Primitive | Strategy | Notes |
|-----------|----------|-------|
| Sphere | Spherical (θ, φ) | Seam at back (θ=0/2π) |
| Box | Cubic projection | Face selected by normal |
| Cylinder | Cylindrical wrap | Seam at back |
| Torus | Toroidal (2 angles) | Major + minor ring |
| Capsule | Cyl + sphere caps | Transitions at height |
| Cone | Cylindrical | Radius varies with height |
| Hex/Tri Prism | Cylindrical | Approximation |
| Ellipsoid | Spherical | Uses normalized coords |
| Octahedron | Box-like | Face selection |

---

## Common Patterns

### Pattern 1: Tiled Texture

```cpp
primitive.uvScale = make_float2(4.0f, 2.0f);  // 4x2 tiling
primitive.uvOffset = make_float2(0.0f, 0.0f);
primitive.uvRotation = 0.0f;
```

### Pattern 2: Rotated Texture

```cpp
primitive.uvScale = make_float2(1.0f, 1.0f);
primitive.uvRotation = 0.785f;  // 45 degrees (π/4)
```

### Pattern 3: Offset Animation

```cpp
// In update loop:
primitive.uvOffset.x = fmodf(time * 0.1f, 1.0f);  // Scroll horizontally
```

### Pattern 4: Different Texture Per Primitive

```cpp
torus.textureID = 0;     // Metal
cylinder.textureID = 1;  // Wood
sphere.textureID = 2;    // Stone
```

---

## OpenGL Setup

### Vertex Attributes

```cpp
// Location 0: Vertex position (float4)
// Location 1: Vertex color (float4)
// Location 2: UV coordinates (float2)  ← NEW
// Location 3: Primitive ID (float)     ← NEW
```

### Texture Array Setup

```cpp
glGenTextures(1, &textureArray);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 1024, 1024, 8, ...);

// Load to layer
glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, w, h, 1, ...);

// Generate mipmaps once
glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

// Bind and use
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glUniform1i(loc_textureArray, 0);
glUniform1i(loc_useTexture, 1);  // Direct mode
```

---

## Rendering Modes

| Mode | useTexture | Shader Code |
|------|-----------|-------------|
| SDF Color | 0 | `color = sdfColor * lighting` |
| Texture Array | 1 | `texture(textureArray, vec3(UV, PrimID))` |
| Single/Atlas | 2 | `texture(texture1, UV)` |

---

## Debugging

### Visualize UVs

```glsl
FragColor = vec4(TexCoord.x, TexCoord.y, 0.0, 1.0);
```
Expected: Red-green gradient

### Visualize Primitive IDs

```glsl
FragColor = vec4(float(PrimitiveID) / 8.0, 0.0, 0.0, 1.0);
```
Expected: Different red shades per primitive

### Validate UV Range

```cpp
// Download UVs to CPU
std::vector<float2> uvs(vertexCount);
cudaMemcpy(uvs.data(), d_uvs, vertexCount * sizeof(float2), cudaMemcpyDeviceToHost);

// Check range
for (auto& uv : uvs) {
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        std::cout << "UV out of range: " << uv.x << ", " << uv.y << std::endl;
    }
}
```

---

## Performance

- **Overhead**: ~15-25% (with UV generation enabled)
- **Memory**: +12 bytes per vertex (8 UV + 4 primitive ID)
- **Disable**: Pass `nullptr` for zero overhead
- **Inline functions**: All UV functions are `__device__ __forceinline__`

---

## Coordinate Spaces

```
World Space
    ↓ (translate, rotate, scale)
Local Space
    ↓ (displacement: twist, bend, sine)
Distorted Local Space (for SDF evaluation)

UV Calculation Uses: PRE-DISTORTION Local Space
```

This ensures UVs remain stable when primitives are twisted/bent.

---

## Operation Behavior

| Operation | UV Source |
|-----------|-----------|
| Union | Primitive with smallest SDF |
| Subtract | Cutter (interior), Base (exterior) |
| Intersect | Primitive with largest SDF |
| Blend | Dominant primitive (or interpolate) |

---

## Memory Layout

```
Vertex Data per vertex (44 bytes):
- Position: float4 (16 bytes)
- Color: float4 (16 bytes)
- UV: float2 (8 bytes)
- Primitive ID: int (4 bytes)
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| UVs all (0,0) | Buffer not allocated | Check d_uvCoords != nullptr |
| Black texture | Wrong texture layer | Check textureID matches loaded layer |
| Seams visible | UV wrap discontinuity | Apply seam fixing (future enhancement) |
| Wrong UVs | Primitive ID incorrect | Debug map() function |
| Crash | Null pointer | Validate all grid pointers |

---

## File Locations

- **Data structures**: `src/SDFMesh/Commons.cuh`
- **UV functions**: `src/SDFMesh/MarchingCubesKernels.cu`
- **API**: `src/SDFMesh/CudaSDFMesh.h/cpp`
- **Helpers**: `src/SDFMesh/CudaSDFUtil.h`
- **Shaders**: `src/SDFMesh/CudaSDFUtil.h`

---

## Quick Start Checklist

- [ ] Allocate UV and primitive ID buffers (OpenGL)
- [ ] Register buffers with CUDA
- [ ] Set primitive UV parameters (scale, offset, rotation)
- [ ] Map buffers before Update()
- [ ] Call Update() with UV pointers
- [ ] Unmap buffers after Update()
- [ ] Setup vertex attributes (locations 2, 3)
- [ ] Create texture array (optional)
- [ ] Load textures to layers (optional)
- [ ] Set shader uniforms (textureArray, useTexture)
- [ ] Render and verify

---

## Example: Minimal Integration

```cpp
// 1. Buffers
GLuint uvbo, primidbo;
cudaGraphicsResource *cudaUVBO, *cudaPrimIDBO;
// ... allocate and register ...

// 2. Primitive
primitive.uvScale = make_float2(2.0f, 1.0f);
primitive.textureID = 0;

// 3. Render loop
float2* d_uv; int* d_prim;
cudaGraphicsMapResources(1, &cudaUVBO, 0);
cudaGraphicsMapResources(1, &cudaPrimIDBO, 0);
cudaGraphicsResourceGetMappedPointer((void**)&d_uv, &size, cudaUVBO);
cudaGraphicsResourceGetMappedPointer((void**)&d_prim, &size, cudaPrimIDBO);

mesh.Update(time, d_vbo, d_cbo, d_ibo, d_uv, d_prim);

cudaGraphicsUnmapResources(1, &cudaUVBO, 0);
cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);

// 4. Shader
glUniform1i(loc_useTexture, 1);  // Enable texture array mode
glDrawArrays(GL_TRIANGLES, 0, count);
```

---

## Resources

- Design Doc: `docs/PRIMITIVE_UV_MAPPING_DESIGN.md`
- Implementation Guide: `docs/PRIMITIVE_UV_MAPPING_IMPLEMENTATION.md`
- Summary: `docs/UV_IMPLEMENTATION_COMPLETE.md`
- Integration: `docs/INTEGRATION_GUIDE.md`


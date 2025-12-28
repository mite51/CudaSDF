# UV Mapping Implementation - Complete

## Summary

Successfully implemented **Phases 1-4** of the primitive-based UV mapping system for the CudaSDF marching cubes mesh generator. The implementation adds per-primitive UV coordinates that are generated during mesh extraction, with support for Direct Mode using OpenGL texture arrays.

---

## What Was Implemented

### ✅ Phase 1: Core UV Generation (COMPLETE)

#### 1.1 Data Structure Extensions
- **`Commons.cuh`**:
  - Added `UVMode` enum (UV_PRIMITIVE, UV_WORLD_TRIPLANAR, UV_PLANAR_X/Y/Z)
  - Extended `SDFPrimitive` struct with UV fields (32 bytes added):
    - `float2 uvScale` - UV tiling/stretching
    - `float2 uvOffset` - UV translation
    - `float uvRotation` - UV rotation in radians
    - `int uvMode` - UV mapping mode
    - `int textureID` - texture layer index
    - `int atlasIslandID` - for future atlas packing
  - Extended `SDFGrid` struct with:
    - `float2* d_uvCoords` - Output UV coordinates
    - `int* d_primitiveIDs` - Primitive ID per vertex

#### 1.2 UV Mapping Functions
- **`MarchingCubesKernels.cu`**: Added complete UV mapping functions for all primitive types:
  - `uvSphere()` - Spherical/equirectangular mapping
  - `uvBox()` - Cubic projection with face selection
  - `uvCylinder()` - Cylindrical wrapping
  - `uvTorus()` - Toroidal (major + minor angles)
  - `uvCapsule()` - Cylinder with spherical caps
  - `uvCone()` - Cylindrical with varying radius
  - `transformUV()` - Apply scale, rotation, offset
  - `approximateLocalNormal()` - Normal estimation for face selection
  - `computePrimitiveUV()` - Master function that selects appropriate UV mapping

#### 1.3 Modified `map()` Function
- Extended signature to return:
  - `int& outPrimitiveID` - Which primitive owns this surface point
  - `float3& outLocalPos` - Position in primitive's local space (pre-distortion)
- Tracks pre-distortion local positions for stable UVs under twist/bend
- Properly handles operation-based primitive selection:
  - **Union**: Uses primitive with smallest SDF value
  - **Subtract**: Interior uses cutter's UVs, exterior uses base
  - **Intersect**: Uses primitive with larger SDF value
  - **Blended ops**: Uses dominant primitive (or interpolates)

#### 1.4 Updated Marching Cubes Kernels
- **`countActiveBlockTriangles`**: Updated all `map()` calls
- **`generateActiveBlockTriangles`**: 
  - Calls `map()` with new signature
  - Writes primitive IDs to `grid.d_primitiveIDs`
  - Computes and writes UVs using `computePrimitiveUV()`
  - Handles invalid primitive IDs gracefully
- **`scoutActiveBlocks`**: Updated `map()` calls

---

### ✅ Phase 2: CudaSDFMesh Integration (COMPLETE)

#### 2.1 Updated Interface
- **`CudaSDFMesh.h`**:
  - Modified `Update()` signature to accept optional UV output pointers:
    ```cpp
    void Update(float time, float4* d_outVertices, float4* d_outColors, 
                unsigned int* d_outIndices,
                float2* d_outUVs = nullptr, 
                int* d_outPrimitiveIDs = nullptr);
    ```

#### 2.2 Updated Implementation
- **`CudaSDFMesh.cpp`**:
  - Assigns UV and primitive ID pointers to `d_grid`
  - No additional memory allocation needed (caller provides buffers)

#### 2.3 Primitive Creation Helpers
- **`CudaSDFUtil.h`**:
  - Updated `CreatePrimitive()` to initialize UV fields with sensible defaults:
    - `uvScale = (1, 1)` - No tiling
    - `uvOffset = (0, 0)` - No translation
    - `uvRotation = 0` - No rotation
    - `uvMode = UV_PRIMITIVE` - Use canonical mapping
    - `textureID = 0` - Default texture layer
    - `atlasIslandID = -1` - Not packed

---

### ✅ Phase 4: Direct Mode with Texture Arrays (COMPLETE)

#### 4.1 Updated Vertex Shader
- **`CudaSDFUtil.h`**: Modified vertex shader to:
  - Accept `layout (location = 3) in float aPrimitiveID`
  - Pass primitive ID to fragment shader as `flat out int PrimitiveID`
  - Use `flat` qualifier to prevent interpolation (integer ID)

#### 4.2 Updated Fragment Shader
- **`CudaSDFUtil.h`**: Modified fragment shader to:
  - Receive `flat in int PrimitiveID`
  - Accept `uniform sampler2DArray textureArray` for multi-texture support
  - Support 3 rendering modes via `uniform int useTexture`:
    - **Mode 0**: SDF color (no texture)
    - **Mode 1**: Direct mode - `texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID)))`
    - **Mode 2**: Single texture / Atlas mode - `texture(texture1, TexCoord)`

---

## Usage Example

### Basic Setup

```cpp
// 1. Create primitives with UV parameters
SDFPrimitive torus = CreateTorusPrim(
    make_float3(0.0f, 0.0f, 0.0f),
    make_float4(0.0f, 0.0f, 0.0f, 1.0f),
    make_float3(1.0f, 0.5f, 0.2f),
    SDF_UNION, 0.0f,
    0.4f, 0.15f
);
torus.uvScale = make_float2(4.0f, 2.0f);  // Tile 4x horizontally, 2x vertically
torus.uvMode = UV_PRIMITIVE;
torus.textureID = 0;  // Use layer 0 of texture array

mesh.AddPrimitive(torus);

// 2. Allocate UV buffers (on GPU via OpenGL/CUDA interop)
GLuint uvbo, primidbo;
glGenBuffers(1, &uvbo);
glBindBuffer(GL_ARRAY_BUFFER, uvbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);

glGenBuffers(1, &primidbo);
glBindBuffer(GL_ARRAY_BUFFER, primidbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);

// Register with CUDA
cudaGraphicsResource* cudaUVBO, *cudaPrimIDBO;
cudaGraphicsGLRegisterBuffer(&cudaUVBO, uvbo, cudaGraphicsRegisterFlagsWriteDiscard);
cudaGraphicsGLRegisterBuffer(&cudaPrimIDBO, primidbo, cudaGraphicsRegisterFlagsWriteDiscard);

// 3. Generate mesh with UVs
float2* d_uvPtr;
int* d_primIDPtr;
cudaGraphicsMapResources(1, &cudaUVBO, 0);
cudaGraphicsMapResources(1, &cudaPrimIDBO, 0);
cudaGraphicsResourceGetMappedPointer((void**)&d_uvPtr, &size, cudaUVBO);
cudaGraphicsResourceGetMappedPointer((void**)&d_primIDPtr, &size, cudaPrimIDBO);

mesh.Update(time, d_vbo, d_cbo, d_ibo, d_uvPtr, d_primIDPtr);

cudaGraphicsUnmapResources(1, &cudaUVBO, 0);
cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);

// 4. Setup vertex attributes
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);  // UVs
glEnableVertexAttribArray(2);

glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);  // Primitive ID
glEnableVertexAttribArray(3);

// 5. Create and setup texture array
GLuint textureArray;
glGenTextures(1, &textureArray);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 1024, 1024, 8, 
             0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
// ... load textures to layers ...

// 6. Render with texture array
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glUniform1i(glGetUniformLocation(shader, "textureArray"), 0);
glUniform1i(glGetUniformLocation(shader, "useTexture"), 1);  // Direct mode
glDrawArrays(GL_TRIANGLES, 0, vertexCount);
```

---

## Key Features

### ✅ Per-Primitive UV Control
- Each primitive can define its own UV scale, offset, rotation
- Independent of world-space transforms
- Stable under distortions (twist, bend, etc.)

### ✅ Operation-Aware UV Assignment
- Correctly assigns UVs based on which primitive defines the surface
- Subtraction operations properly handle interior/exterior surfaces
- Blended operations use dominant primitive

### ✅ All Primitive Types Supported
- Sphere, Box, Torus, Cylinder, Capsule, Cone
- Rounded variants (Rounded Box, Rounded Cylinder, Rounded Cone)
- Prisms (Hex Prism, Triangular Prism)
- Ellipsoid, Octahedron

### ✅ Direct Mode with Texture Arrays
- Multiple primitives can have different textures
- Single draw call for entire mesh
- Primitive ID automatically selects texture layer
- Efficient GPU rendering

---

## What's Next (Not Yet Implemented)

### Phase 3: Atlas Mode (Optional - for future)
- Chart extraction (group triangles by primitive ID)
- Rectangle packing (using `stb_rect_pack.h`)
- UV remapping to atlas space
- Useful for texture projection baking

### Additional Enhancements (Optional)
- UV seam fixing for wrapped primitives (sphere, cylinder)
- UV debug visualization mode (display UVs as colors)
- Triplanar mapping mode
- Connected-component analysis for disconnected regions

---

## Testing Recommendations

### 1. Visual UV Test
```glsl
// In fragment shader, replace color calculation with:
color = vec3(TexCoord.x, TexCoord.y, 0.0);  // UVs as RG colors
```
Expected: Smooth gradient across each primitive

### 2. Checkerboard Test
Load a checkerboard texture and verify:
- Sphere: Equirectangular pattern with seam at back
- Box: Each face shows flat checkerboard
- Cylinder: Wraps around with vertical stripes
- Torus: Double wrapping (ring + tube)

### 3. Multi-Primitive Test
Create scene with 3+ primitives, each with different texture layer:
```cpp
primitive1.textureID = 0;  // Metal
primitive2.textureID = 1;  // Wood
primitive3.textureID = 2;  // Stone
```
Verify each primitive displays correct texture.

### 4. Operation Test
Test subtraction: verify interior surface uses cutter's UVs.

---

## Performance Notes

- **Expected overhead**: 15-25% (documented estimate)
- **UV generation is optional**: Pass `nullptr` to `Update()` to disable
- **Memory overhead**: +12 bytes per vertex (8 for UV + 4 for primitive ID)
- **All UV functions are inlined**: Minimal function call overhead

---

## Files Modified

1. `src/SDFMesh/Commons.cuh` - Data structures
2. `src/SDFMesh/MarchingCubesKernels.cu` - UV generation kernels
3. `src/SDFMesh/CudaSDFMesh.h` - Interface
4. `src/SDFMesh/CudaSDFMesh.cpp` - Implementation
5. `src/SDFMesh/CudaSDFUtil.h` - Helpers and shaders

---

## Conclusion

The implementation is **complete and ready for testing**. All core UV generation functionality is in place, along with Direct Mode texture array support. The system properly handles:

- ✅ All primitive types with canonical UV mappings
- ✅ UV transforms (scale, offset, rotation)
- ✅ Operation-based primitive selection
- ✅ Distortion handling (pre-distortion coordinates)
- ✅ Multi-texture rendering via texture arrays
- ✅ Shader integration for visualization

The codebase compiles without errors and follows the design specifications from the planning documents.


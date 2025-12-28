# Primitive UV Mapping - Quick Reference

## 📚 Documentation Overview

This implementation adds per-primitive UV mapping to the CudaSDF marching cubes system. Three documents guide the implementation:

### 1. **PRIMITIVE_UV_MAPPING_DESIGN.md** (Full Design Document)
- Complete technical architecture
- Data structure definitions
- Kernel modifications
- Atlas packing system
- ~50 pages of detailed specifications

### 2. **PRIMITIVE_UV_MAPPING_SUMMARY.md** (Executive Summary)
- Key design decisions explained
- Identified issues with solutions
- Risk assessment
- Success criteria
- Implementation priority

### 3. **PRIMITIVE_UV_MAPPING_IMPLEMENTATION.md** (Step-by-Step Guide)
- Phase-by-phase implementation
- Code snippets for each step
- Testing checklist
- Debugging tips

---

## 🎯 Core Concept

**Instead of post-process UV unwrapping**, each SDF primitive defines its own UV coordinates that are generated **during marching cubes extraction**.

### How It Works

```
Sphere Primitive (with spherical UVs)
    ↓
SDF Evaluation (tracks which primitive)
    ↓
Marching Cubes (generates vertices)
    ↓
Per-Vertex: Compute UV from primitive type + local position
    ↓
Result: Mesh with predictable per-primitive UVs
```

---

## 🚀 Quick Start

### Minimum Viable Implementation (MVP)

**Time**: 2-3 weeks

**Steps**:
1. Add UV fields to `SDFPrimitive` struct (4 floats: uvScale, uvOffset, uvRotation, uvMode)
2. Add UV mapping functions for each primitive type (sphere, box, cylinder, etc.)
3. Modify `map()` to return primitive ID + local position
4. Update marching cubes kernels to generate UVs
5. Test with single primitives + checkerboard texture

**Result**: Basic per-primitive UV generation working.

---

## 📊 Two Operating Modes

### Direct Mode (Multi-Texture with Texture Array)
- Each primitive's UVs map to [0,1]
- **Multiple textures** via OpenGL `sampler2DArray`
- Primitive ID selects texture layer in shader
- No post-processing required

**Shader Code**:
```glsl
flat in int PrimitiveID;
uniform sampler2DArray textureArray;
vec3 color = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
```

**Setup**:
```cpp
// Create texture array with 8 layers
GLuint textureArray;
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 1024, 1024, 8, ...);

// Load textures to layers
glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layerIndex, width, height, 1, ...);

// Assign layer to primitive
torus.textureID = 0;  // Layer 0 = metal texture
cylinder.textureID = 1;  // Layer 1 = wood texture
```

### Atlas Mode (Single Texture)
- Group primitives into UV charts
- Pack charts into single texture atlas
- Remap UVs to atlas space
- Use standard `sampler2D`

**Shader Code**:
```glsl
uniform sampler2D atlasTexture;
vec3 color = texture(atlasTexture, TexCoord).rgb;  // UVs already remapped
```

**Process**:
```cpp
// Extract charts, pack, and remap
auto charts = ExtractCharts(vertices, primIDs, uvCoords);
auto packed = PackChartsIntoAtlas(charts, 2048, 4);
RemapUVsToAtlas(uvCoords, primIDs, charts, packed);
```

---

## 🔑 Key Technical Decisions

### Decision 1: When to Calculate UVs
**During vertex generation** in marching cubes kernels
- Pros: Knows world position and primitive ID
- Cons: Small performance overhead (~15-25%)

### Decision 2: How to Handle Distortions
**Use pre-distortion local coordinates** for UV calculation
- UVs remain stable when primitive is twisted/bent
- Stored separately from post-distortion position

### Decision 3: How to Handle Operations
- **Union**: Use primitive with smallest SDF value
- **Subtract**: Interior surface uses cutter's UVs
- **Blend**: Use dominant primitive (or interpolate UVs)

### Decision 4: Primitive UV Mappings

| Primitive | UV Strategy |
|-----------|-------------|
| Sphere | Spherical (θ, φ) coordinates |
| Box | Cubic projection (6 faces, select by normal) |
| Cylinder | Cylindrical (angle, height) |
| Torus | Toroidal (major angle, minor angle) |
| Capsule | Cylinder + spherical caps |
| Others | Cylindrical or planar approximation |

---

## ⚠️ Known Issues & Solutions

### Issue 1: UV Seams
**Problem**: Spheres/cylinders have wrap-around seam at θ=0/2π

**Solution**: Detect jumps > 0.5 and adjust by ±1.0
```cuda
if (fabsf(uv1.x - uv0.x) > 0.5f) {
    if (uv1.x > uv0.x) uv1.x -= 1.0f;
    else uv1.x += 1.0f;
}
```

### Issue 2: Blended Operations
**Problem**: Multiple primitives contribute at blend boundaries

**Solution**: Use dominant primitive (smallest SDF contribution)
- May cause UV discontinuity at boundary
- Alternative: Interpolate UVs (experimental)

### Issue 3: Performance
**Problem**: Additional computations per vertex

**Solution**:
- Make UV generation optional (default: disabled)
- Use `__forceinline__` on device functions
- Share computation with color calculation
- Expected overhead: 15-25%

---

## 📝 Implementation Phases

### Phase 1: Core UV Generation (Weeks 1-2) ✅
- Extend data structures
- Implement UV functions
- Modify kernels
- Test single primitives

### Phase 2: Visualization (Week 3) ✅
- Create test textures
- Update shaders
- Visual testing

### Phase 3: Shader Integration for Direct Mode (Week 4) 🔄
- Add primitive ID vertex attribute
- Implement texture array system
- Update shaders for multi-texture support
- Test with multiple textured primitives

### Phase 4: Atlas Mode (Weeks 5-6) 🔄
- Chart extraction
- Rect packing integration
- UV remapping

### Phase 5: Polish (Week 7) 🔄
- Seam fixing
- Edge case handling
- Performance optimization

**Total Estimated Time**: 6-7 weeks

---

## 🧪 Testing Strategy

### Unit Tests
- [ ] Each UV mapping function (sphere, cylinder, etc.)
- [ ] UV transform application
- [ ] Primitive ID propagation

### Integration Tests
- [ ] Single primitive with checkerboard
- [ ] Multi-primitive union
- [ ] Subtraction with interior/exterior differentiation
- [ ] Distorted primitives

### Visual Tests
- [ ] Checkerboard on all primitive types
- [ ] UV debug visualization (RGB = UV coords)
- [ ] Seam detection
- [ ] Atlas packing visualization

### Performance Tests
- [ ] Benchmark with/without UV generation
- [ ] Measure memory overhead
- [ ] Profile atlas packing

---

## 🎨 Shader Integration Example

### Complete Vertex Shader

```glsl
#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in float aPrimitiveID;  // NEW

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec2 TexCoord;
out vec4 VertColor;
flat out int PrimitiveID;  // NEW: flat = no interpolation

void main() {
    vec4 worldPos = model * vec4(aPos.xyz, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = worldPos.xyz;
    TexCoord = aTexCoord;
    VertColor = aColor;
    PrimitiveID = int(aPrimitiveID);
}
```

### Complete Fragment Shader

```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;
in vec4 VertColor;
flat in int PrimitiveID;  // NEW

uniform sampler2DArray textureArray;  // NEW: For direct mode
uniform sampler2D atlasTexture;       // NEW: For atlas mode
uniform int useTexture;  // 0=SDF color, 1=Texture array, 2=Atlas

// ... lighting uniforms ...

void main() {
    // Simple lighting (customize as needed)
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(cross(dFdx(FragPos), dFdy(FragPos)));
    float diff = max(dot(normal, lightDir), 0.0) + 0.2;
    
    vec3 color;
    
    if (useTexture == 1) {
        // DIRECT MODE: Texture array with primitive ID
        vec3 texColor = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
        color = texColor * diff;
    } else if (useTexture == 2) {
        // ATLAS MODE: Single texture with remapped UVs
        vec3 texColor = texture(atlasTexture, TexCoord).rgb;
        color = texColor * diff;
    } else {
        // SDF COLOR MODE: Vertex colors
        color = VertColor.rgb * diff;
    }
    
    FragColor = vec4(color, 1.0);
}
```

### OpenGL Setup Code

```cpp
// 1. Create texture array
GLuint textureArray;
glGenTextures(1, &textureArray);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 1024, 1024, 8, 
             0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

// 2. Load textures to layers
for (int i = 0; i < numTextures; ++i) {
    unsigned char* data = stbi_load(filenames[i], &w, &h, &c, 3);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, w, h, 1, GL_RGB, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
}
glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

// 3. Setup primitive ID vertex attribute
glGenBuffers(1, &primidbo);
glBindBuffer(GL_ARRAY_BUFFER, primidbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);
glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
glEnableVertexAttribArray(3);

// 4. In render loop
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
glUniform1i(glGetUniformLocation(shader, "textureArray"), 0);
glUniform1i(glGetUniformLocation(shader, "useTexture"), 1);  // Direct mode
```

---

## 💡 Usage Examples

### Example 1: Simple Textured Sphere
```cpp
SDFPrimitive sphere = CreateSpherePrim(
    make_float3(0.0f, 0.0f, 0.0f),
    make_float4(0.0f, 0.0f, 0.0f, 1.0f),
    make_float3(1.0f, 1.0f, 1.0f),
    SDF_UNION,
    0.0f,
    0.5f
);
sphere.uvScale = make_float2(1.0f, 1.0f);
sphere.uvMode = UV_PRIMITIVE;

mesh.EnableUVGeneration(true);
mesh.AddPrimitive(sphere);
```

### Example 2: Tiled Cylinder
```cpp
SDFPrimitive cylinder = CreateCylinderPrim(...);
cylinder.uvScale = make_float2(8.0f, 4.0f);  // 8x wrap, 4x vertical
cylinder.uvRotation = 0.785f;  // 45° rotation
cylinder.uvMode = UV_PRIMITIVE;
```

### Example 3: Multi-Primitive Atlas
```cpp
// Create multiple primitives
mesh.AddPrimitive(torus);
mesh.AddPrimitive(sphere);
mesh.AddPrimitive(cylinder);

// Generate with UVs
mesh.EnableUVGeneration(true);
mesh.Update(time, d_verts, d_colors, d_indices, d_uvs, d_primIDs);

// Pack into atlas
auto charts = ExtractCharts(verts, primIDs, uvs);
auto packed = PackChartsIntoAtlas(charts, 2048, 4);
RemapUVsToAtlas(uvs, primIDs, charts, packed);

// Now use for texture projection or baking
```

---

## 🛠️ Key Files to Modify

| File | Changes | Difficulty |
|------|---------|------------|
| `Commons.cuh` | Add UV fields to SDFPrimitive | Easy |
| `MarchingCubesKernels.cu` | Add UV functions, modify map() | Medium |
| `CudaSDFMesh.h/cpp` | Add UV management | Easy |
| `CudaSDFUtil.h` | Update shaders for texture array | Medium |
| `main.cpp` | Texture array setup, atlas packing | Medium |
| `stb_image.h` | Include for texture loading | Easy |

---

## 📈 Expected Results

### Before
- Mesh has no UV coordinates
- UV unwrapping is separate post-process
- Unpredictable UV layouts

### After
- Each primitive has predictable UVs
- UVs generated during mesh extraction
- Support for both direct and atlas modes
- Stable under distortions

---

## 🎓 Learning Resources

### Concepts Used
- **Marching Cubes**: Isosurface extraction from SDF
- **UV Parameterization**: Mapping 3D surfaces to 2D texture space
- **Rect Packing**: Bin packing problem for texture atlases
- **SDF Operations**: Union, subtraction, intersection of implicit surfaces

### Related Techniques
- Triplanar mapping (world-space projection)
- Ptex (per-face texturing without UVs)
- Mesh parameterization (LSCM, ABF++)
- Texture synthesis and baking

---

## 🔗 References

- stb_rect_pack: https://github.com/nothings/stb
- UV mapping theory: https://learnopengl.com/Getting-started/Textures
- Marching cubes: http://paulbourke.net/geometry/polygonise/
- SDF operations: https://iquilezles.org/articles/distfunctions/

---

## ✅ Benefits of This Approach

1. **Predictable**: Each primitive type has well-defined UV layout
2. **Stable**: UVs don't change when primitive transforms or deforms
3. **Flexible**: Supports both simple and advanced workflows
4. **Integrated**: Generated alongside vertices, no separate unwrap pass
5. **Operation-Aware**: Correctly handles union/subtract/intersect

---

## 🚨 Limitations & Future Work

### Current Limitations
- No support for custom UV functions (per primitive)
- Atlas packing is CPU-side (slower for real-time)
- No vertex duplication along seams (may cause filtering artifacts)
- Blended regions may have discontinuous UVs

### Future Enhancements
- GPU-based atlas packing
- Triplanar mapping mode
- Multi-texture support (different texture per primitive)
- Vertex seam splitting for perfect atlas packing
- UV animation (scrolling, rotation)

---

## 📞 Support & Debugging

### Common Problems

**UVs are all (0,0)**
→ Check d_uvCoords buffer is allocated and passed to Update()

**Texture appears black**
→ UVs may be out of range, check with UV debug visualization

**Visible seams on sphere**
→ Enable seam fixing or use atlas mode with vertex duplication

**Wrong primitive UVs appear**
→ Debug map() function, check primitive ID assignment

**Crash in kernel**
→ Check all pointers in SDFGrid are valid, use cuda-memcheck

### Debug Visualization

```glsl
// In fragment shader:
color = vec3(TexCoord.x, TexCoord.y, 0.0);  // Visualize UVs as RG
```

---

## 📦 Deliverables

After complete implementation:

1. **Code**: All modified source files
2. **Tests**: Unit and integration tests
3. **Examples**: Sample scenes demonstrating features
4. **Documentation**: Updated user guide
5. **Performance Report**: Benchmarks and profiling results

---

## 🎉 Conclusion

This primitive-based UV mapping system provides a robust foundation for texturing SDF-based meshes. The design balances simplicity (direct mode) with advanced capabilities (atlas mode), maintains primitive-surface relationships, and handles operations gracefully.

**Ready to implement?** Start with Phase 1 (Core UV Generation) and iterate based on results!



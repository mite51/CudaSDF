# Primitive UV Mapping - Executive Summary

## Overview

This document summarizes the key design decisions and identifies potential issues for implementing per-primitive UV mapping in the CudaSDF marching cubes system.

## Core Concept

Instead of unwrapping the final mesh as a post-process, **each primitive defines its own UV parameterization** which is propagated through the SDF evaluation and marching cubes extraction process. This means:

- A sphere primitive uses spherical UV coordinates
- A box uses cubic projection
- A cylinder uses cylindrical coordinates
- etc.

These primitive UVs are generated **during vertex creation** in the marching cubes kernels, maintaining the relationship between primitives and the final mesh surface.

---

## Key Design Decisions

### ✅ Decision 1: UV Calculation Point
**When**: During marching cubes triangle generation
**Where**: In `generateActiveBlockTriangles` kernel
**Why**: Vertices know their world position and can query which primitive they belong to

### ✅ Decision 2: Primitive Identification
**How**: Extended `map()` function returns primitive ID alongside distance and color
**Storage**: Additional `int* d_primitiveIDs` buffer parallel to vertices
**Benefit**: Enables post-processing (atlas packing) and debugging

### ✅ Decision 3: Two Operating Modes with Shader Integration

#### Direct Mode (Multi-Texture with Texture Array)
- Each primitive's UVs remain in [0, 1] range
- Uses OpenGL `sampler2DArray` for multiple textures
- Primitive ID passed to fragment shader selects texture layer
- No additional processing required
- **Shader**: `texture(textureArray, vec3(UV, primitiveID))`
- **Use case**: Multiple primitives with different materials
- **GPU Support**: OpenGL 3.0+ (widely available)

#### Atlas Mode (Single Texture)
- Post-process groups vertices by primitive ID into "charts"
- Pack charts into unified texture atlas
- Remap all UVs to atlas space
- Uses standard `sampler2D` with single texture
- **Shader**: `texture(atlasTexture, UV)` with remapped UVs
- **Use case**: Texture projection baking, single texture for entire scene

### ✅ Decision 4: Handle Distortions Pre-Application
**Problem**: Twist/bend distortions modify vertex positions
**Solution**: Compute UVs from **pre-distortion** local coordinates
**Implementation**: `map()` stores local position before applying displacement

### ✅ Decision 5: Per-Primitive UV Controls
Each primitive gets:
- `float2 uvScale` - tile/stretch UVs
- `float2 uvOffset` - shift UVs
- `float uvRotation` - rotate UVs
- `int uvMode` - choose mapping type (primitive/triplanar/planar)
- `int textureID` - for multi-texture support (future)

---

## Identified Issues & Solutions

### 🔴 Issue 1: Blended Operations
**Problem**: When using smooth blending (SDF_UNION_BLEND), multiple primitives contribute to a point. Which primitive's UVs should be used?

**Solutions Considered**:
1. **Use dominant primitive** (smallest SDF contribution) ✅ CHOSEN
   - Simple, deterministic
   - May cause sharp UV transitions at blend boundaries
   
2. **Interpolate UVs based on blend factor**
   - Smooth transitions
   - But UVs may not make semantic sense (sphere UV + box UV = ???)
   
3. **Create new "blend zone" chart**
   - Most correct but very complex
   - Requires sophisticated chart extraction

**Recommendation**: Start with option 1, evaluate if transitions are acceptable. Add option 2 as refinement if needed.

---

### 🔴 Issue 2: Subtraction Interior Surfaces
**Problem**: When subtracting (e.g., sphere cuts through box), interior surfaces should use the cutting primitive's UVs, but exterior surfaces use the base primitive.

**Solution**: Track which primitive "won" the SDF comparison:
```cuda
case SDF_SUBTRACT:
    float d_new = max(d, -d_prim);
    if (d_new > d) {  // Subtracted primitive defines surface
        outPrimitiveID = subtractedPrimID;
        outLocalPos = subtractedLocalPos;
    }
    d = d_new;
```

**Complexity**: Medium - requires careful tracking in `map()` function

---

### 🔴 Issue 3: UV Seams (Sphere/Cylinder Wrap-Around)
**Problem**: Spherical UVs have seam at θ = 0/2π. A triangle crossing the seam will have vertices with UVs like (0.01, 0.5), (0.99, 0.5), (0.5, 0.7) causing incorrect interpolation.

**Solutions**:
1. **Detect and fix in kernel** ✅ CHOSEN (for direct mode)
   - Check if UV difference > 0.5, adjust by ±1.0
   - Fast, inline with generation
   
2. **Duplicate vertices along seams** (for atlas mode)
   - Create separate vertices for each side of seam
   - Required for proper atlas packing
   - More complex, post-process step

**Recommendation**: Use solution 1 for direct mode, solution 2 for atlas mode.

---

### 🟡 Issue 4: Performance Overhead
**Problem**: Additional computations per vertex (primitive lookup, UV calculation)

**Estimated Impact**:
- Primitive ID tracking: ~5-10% overhead
- UV calculation: ~10-15% overhead
- Total: ~15-25% slower marching cubes generation

**Mitigations**:
1. Make UV generation optional (disabled by default)
2. Use `__forceinline__` on UV functions
3. Share computation with color calculation (both need primitive ID)
4. Consider shared memory caching for primitive data

**Assessment**: Acceptable tradeoff given the functionality gained. UV generation is typically a one-time or infrequent operation.

---

### 🟡 Issue 5: Memory Overhead
**Per-Vertex Data**:
- Current: 32 bytes (position + color)
- With UVs: 44 bytes (position + color + UV + primitive ID)
- Increase: +37.5%

**For 20M vertices**: +240 MB (~880 MB total)

**Mitigation**: 
- UV generation is optional
- Can free primitive ID buffer after atlas packing
- Modern GPUs have sufficient VRAM for this

**Assessment**: Manageable, especially for one-time unwrap operations.

---

### 🟡 Issue 6: Box UV Face Selection
**Problem**: Box has 6 faces. Need to determine which face a vertex belongs to for proper UV mapping.

**Solution**: Use surface normal to select dominant axis:
```cuda
float3 normal = computeNormal(localPos);  // Approximate
float3 absN = abs(normal);

if (absN.x > absN.y && absN.x > absN.z) {
    // X-face: use Y and Z for UV
    uv = make_float2(localPos.z, localPos.y);
} else if (absN.y > absN.z) {
    // Y-face: use X and Z for UV
    uv = make_float2(localPos.x, localPos.z);
} else {
    // Z-face: use X and Y for UV
    uv = make_float2(localPos.x, localPos.y);
}
```

**Complexity**: Low - normal approximation is already done for color calculations.

---

### 🟢 Issue 7: Torus UV Ambiguity
**Problem**: Torus has two angular dimensions (major ring angle, minor tube angle). Need to ensure consistent unwrap.

**Solution**: Well-defined mathematical mapping:
- Major angle θ (around center): `atan2(z, x)`
- Minor angle φ (around tube): `atan2(tubeY, tubeX)`

**Assessment**: Straightforward, no significant issues.

---

### 🟢 Issue 8: Scale/Rotation/Transform Interaction
**Problem**: If a primitive is scaled non-uniformly or rotated, how do UVs behave?

**Solution**: UVs are computed in **local space** (after inverse transform), so they're independent of world-space transforms. UV scale/rotation/offset are applied as a final step.

**Benefit**: Clean separation of concerns. Transform primitive in world space without affecting UV layout.

---

### 🔴 Issue 9: Hex Prism and Complex Shapes
**Problem**: Some primitives (hex prism, octahedron) don't have obvious UV parameterizations.

**Solutions**:
1. **Planar projection** - simple but may have distortion
2. **Cylindrical approximation** - works for prisms
3. **Face-based mapping** - treat like a box with more faces

**Recommendation**: Start with cylindrical for prisms, planar for others. Can be refined later.

---

### 🟡 Issue 10: Atlas Packing Complexity
**Problem**: Packing arbitrary UV islands efficiently is NP-hard.

**Solutions**:
1. **Use existing library** (stb_rect_pack) ✅ RECOMMENDED
   - Proven, fast, simple shelf-packing
   - Good enough for most cases
   
2. **Custom optimal packing**
   - Better results but complex
   - Overkill for initial implementation

**Recommendation**: Use stb_rect_pack initially. Can be swapped later if needed.

---

### 🟡 Issue 11: Disconnected Primitive Regions
**Problem**: A primitive might have multiple disconnected surface regions (e.g., a box with a hole cut through it appears on multiple sides).

**Current Design**: Treats all surfaces of a primitive as one chart.

**Potential Issue**: May cause UV stretching or overlapping triangles in atlas mode.

**Future Solution**: Implement connected-component analysis to split into sub-charts. Not critical for initial implementation.

---

### 🔴 Issue 12: Real-Time Updates with Atlas Mode
**Problem**: If primitives animate, primitive IDs don't change, but the mesh topology does. Atlas packing becomes invalid.

**Solutions**:
1. **Disable animation in atlas mode** ✅ CURRENT APPROACH
   - Pack once, freeze mesh
   - Acceptable for texture projection workflow
   
2. **Repack every frame**
   - Computationally expensive
   - UVs would "swim" during animation
   
3. **Persistent atlas regions**
   - Pre-allocate atlas space per primitive
   - Wastes space but maintains stability

**Recommendation**: Use solution 1 initially (one-time unwrap, then static). This matches the current behavior when pressing 'U' key.

---

### 🟢 Issue 13: Multiple Materials Per Primitive
**Problem**: User might want different textures for different parts of a primitive (e.g., different texture on each face of a box).

**Current Design**: One UV layout per primitive.

**Future Enhancement**: 
- Add `textureID` field (already in design)
- Support texture arrays or atlases per primitive
- Not blocking for initial implementation

---

### 🟡 Issue 13: Shader Integration for Multiple Textures
**Problem**: In direct mode, each primitive can have a different texture. How does the shader handle this?

**Solution**: Use OpenGL Texture Arrays (`sampler2DArray`)

**Implementation**:
1. **Data Flow**:
   - Each primitive has `textureID` field (which texture layer)
   - Primitive ID written to vertex buffer alongside position/UV
   - Vertex shader passes primitive ID to fragment shader (flat/no interpolation)
   - Fragment shader: `texture(textureArray, vec3(UV.xy, primitiveID))`

2. **Texture Array Setup**:
```cpp
// Create array with N layers
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 1024, 1024, numTextures, ...);

// Load each texture to a layer
for (int i = 0; i < numTextures; ++i) {
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, width, height, 1, ...);
}
```

3. **Vertex Attribute**:
   - Add `layout (location = 3) in float aPrimitiveID;`
   - Pass as `flat out int PrimitiveID;` (no interpolation)

4. **Fragment Shader**:
```glsl
flat in int PrimitiveID;
uniform sampler2DArray textureArray;
// ...
vec3 color = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
```

**Benefits**:
- Efficient: Single draw call for entire mesh
- Simple: No texture binding per primitive
- Flexible: Each primitive can have different texture

**Limitations**:
- All textures must be same resolution
- Limited by GL_MAX_ARRAY_TEXTURE_LAYERS (typically 256-2048)
- Requires OpenGL 3.0+ (not an issue for modern hardware)

**Alternative for Atlas Mode**:
- Use single `sampler2D` with remapped UVs
- All primitives baked into one texture

---

## Missing Details Identified

### 1. Normal Calculation Strategy
**Need**: Surface normal in local space for box face selection and other UV mappings.

**Options**:
- A) Numerical differentiation (expensive, 4 SDF evaluations)
- B) Approximate from vertex positions (what we're already doing)
- C) Analytical normals for simple primitives (best but requires per-primitive implementation)

**Decision Needed**: Should we add analytical normal computation for primitives?

**Recommendation**: Use option B (approximate from vertices) initially. Good enough and zero cost.

---

### 2. Handling Empty Charts
**Need**: What if a primitive has no surface in the final mesh (fully subtracted)?

**Solution**: Chart extraction naturally handles this (no triangles = no chart). No special case needed.

---

### 3. UV Coordinate System Convention
**Need**: Clarify UV origin and orientation.

**Standard**: 
- U: 0 (left) to 1 (right)
- V: 0 (bottom) to 1 (top)
- Origin at bottom-left (OpenGL convention)

**Note**: Some primitives (sphere) have natural discontinuities. Document these clearly.

---

### 4. Multi-Threading for Atlas Packing
**Need**: Should atlas packing be multi-threaded?

**Assessment**: 
- Packing is typically < 100 charts
- CPU-side operation
- Not a bottleneck

**Decision**: Single-threaded is fine. Can parallelize later if needed.

---

### 5. Undo/Redo for UV Operations
**Need**: If user doesn't like atlas packing result, can they revert?

**Current**: No undo mechanism.

**Future**: Store previous UV state. Not critical for initial implementation.

---

## Implementation Priority

### Phase 1: Must-Have (Core Functionality)
- ✅ Extended `SDFPrimitive` with UV parameters
- ✅ UV mapping functions for all primitive types
- ✅ Modified `map()` to return primitive ID and local position
- ✅ Updated marching cubes kernels to output UVs
- ✅ Basic operation handling (union/subtract/intersect)

### Phase 2: Should-Have (Practical Usage)
- ✅ UV transforms (scale/offset/rotation)
- ✅ Distortion handling (pre-distortion coordinates)
- ✅ UV seam fixing
- ✅ Atlas mode with chart extraction and packing

### Phase 3: Nice-to-Have (Polish)
- ⚠️ Smooth UV blending for blended operations
- ⚠️ Connected-component analysis for disconnected regions
- ⚠️ Analytical normals for complex primitives
- ⚠️ GPU-based atlas packing
- ⚠️ Multi-texture support

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Performance degradation | Medium | High | Optional feature, profiling |
| UV seam artifacts | Medium | Medium | Seam detection and fixing |
| Atlas packing failures | Low | Medium | Use proven library (stb_rect_pack) |
| Blended region UV issues | Medium | Low | Start with dominant primitive approach |
| Complex primitive UVs | Low | Low | Use simple planar fallback |
| Memory exhaustion | Low | Low | UV generation is optional |

**Overall Risk**: **LOW-MEDIUM** - Well-understood problem domain with clear solutions.

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Single primitive generates correct UVs
- ✅ Multiple primitives with union operation have correct UVs
- ✅ Subtraction correctly assigns interior/exterior UVs
- ✅ Checkerboard texture displays correctly on sphere, box, cylinder
- ✅ No crashes or memory leaks

### Full Success
- ✅ All primitive types have correct UV mappings
- ✅ Operations (union/subtract/intersect) work correctly
- ✅ Distortions (twist/bend) maintain stable UVs
- ✅ Atlas mode successfully packs and remaps UVs
- ✅ Performance overhead < 30%
- ✅ Integration with existing projection baking workflow
- ✅ No visible seams or artifacts in typical use cases

---

## Conclusion

### Strengths of This Design
1. **Maintains primitive-surface relationship** - knows which part came from which primitive
2. **Predictable UV layouts** - each primitive type has well-defined mapping
3. **Handles operations gracefully** - clear rules for union/subtract/intersect
4. **Two-mode flexibility** - direct for simplicity, atlas for advanced workflows
5. **Optional feature** - no impact on users who don't need UVs

### Remaining Concerns
1. **Blended operations** - may need refinement after testing
2. **Performance overhead** - needs profiling to confirm acceptability
3. **Atlas packing robustness** - depends on library choice
4. **UV seam handling** - may need iteration to get right

### Recommendation
**PROCEED WITH IMPLEMENTATION** - Design is solid with clear solutions to identified issues. Risks are manageable. Start with Phase 1 (core functionality) and iterate based on testing.

---

## Next Steps

1. **Review this design** with stakeholders/users
2. **Create prototype** of core UV generation (Phase 1)
3. **Test with simple scenes** (single primitives)
4. **Iterate on issues** discovered during testing
5. **Expand to full implementation** (Phases 2-3)
6. **Document and create examples**

**Estimated Total Time**: 5-7 weeks for full implementation
**Estimated MVP Time**: 2-3 weeks


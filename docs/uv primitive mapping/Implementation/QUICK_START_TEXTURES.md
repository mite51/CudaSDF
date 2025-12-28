# Quick Start Guide - Texture Array Mode

## What You Have Now

✅ Complete UV generation system integrated
✅ Texture array with 5 debug textures loaded
✅ All 5 primitives configured with UV scales and texture assignments
✅ Keyboard controls for toggling modes
✅ Real-time texture display on animated SDF primitives

---

## Building and Running

```bash
# Build the project
cd build
cmake ..
cmake --build . --config Release

# Run (from build directory)
cd Release
./CudaSDF.exe
```

---

## Controls

| Key | Function |
|-----|----------|
| **T** | Toggle texture array ON/OFF |
| **V** | Toggle UV generation ON/OFF |
| **SPACE** | Pause/resume animation |
| **W** | Wireframe mode |
| **U** | UV unwrap (legacy feature) |
| **P** | Projection baking (legacy feature) |
| **ESC** | Exit |

---

## What You Should See

### With Textures Enabled (Default)

**Torus (top-left, green)**:
- Numbered checkerboard pattern
- Tiles 4 times horizontally, 2 times vertically
- Shows how UVs wrap around toroidal geometry

**Hex Prism (top-right, red)**:
- Colorful debug pattern
- Twisting animation
- Wraps 2x around the hexagonal prism

**Cone (bottom-left, blue)**:
- UV grid visualization
- Bobbing up/down animation
- Standard 1:1 UV mapping

**Cylinder (bottom-right, yellow)**:
- Orientation/directional arrows
- Wraps 3x around cylinder
- Stretched 1.5x vertically

**Sphere (center, purple)**:
- Omni-directional test pattern
- Subtracted from other shapes (creates holes)
- Tiles 2x2

### With Textures Disabled (Press T)

All primitives show their procedural SDF colors:
- Torus: Green
- Hex Prism: Red
- Cone: Blue
- Cylinder: Yellow
- Sphere: Purple (visible in cutouts)

---

## Verification Checklist

On startup, console should show:

```
Selected CUDA Device: [Your GPU Name]
Loaded texture to layer 0: assets/T_checkerNumbered.PNG
Loaded texture to layer 1: assets/T_debug_color_01.PNG
Loaded texture to layer 2: assets/T_debug_uv_01.PNG
Loaded texture to layer 3: assets/T_debug_orientation_01.PNG
Loaded texture to layer 4: assets/T_OmniDebugTexture_COL.png
Created texture array with 5 layers
```

---

## Testing Scenarios

### Test 1: Verify UV Generation

1. Run application (textures should be visible by default)
2. Press **V** to disable UV generation
   - Console: "UV generation: OFF"
   - Primitives should switch to SDF colors (no textures)
3. Press **V** again to re-enable
   - Console: "UV generation: ON"
   - Textures should reappear

### Test 2: Texture Array Toggle

1. Press **T** to disable texture array
   - Console: "Texture array mode: OFF"
   - Primitives show SDF colors
2. Press **T** to re-enable
   - Console: "Texture array mode: ON"
   - Textures reappear

### Test 3: Animation Control

1. Press **SPACE** to pause
   - Console: "Animation: OFF"
   - All primitives freeze
2. Press **SPACE** to resume
   - Console: "Animation: ON"
   - Animation continues

### Test 4: Wireframe View

1. Press **W** to enable wireframe
   - See triangle mesh structure
   - Textures still applied per-triangle
2. Press **W** to disable
   - Return to filled triangles

---

## Expected Performance

- **Frame Rate**: 100+ FPS (on modern GPUs)
- **Vertex Count**: ~500K-2M vertices (varies with cell size)
- **GPU Memory**: Console prints every second:
  ```
  GPU Mem: Used XXX MB, Free XXX MB | Blocks: XX/XXXX
  ```

---

## Troubleshooting

### Problem: No textures visible

**Check 1**: Are texture files present?
```bash
ls assets/T_*.PNG
ls assets/T_*.png
```

**Check 2**: Is UV generation enabled?
- Press **V** and watch console
- Should say "UV generation: ON"

**Check 3**: Is texture array mode enabled?
- Press **T** and watch console
- Should say "Texture array mode: ON"

### Problem: Wrong textures on primitives

**Cause**: Primitive `textureID` doesn't match loaded layer.

**Fix**: Check primitive setup in `main.cpp`:
```cpp
torus.textureID = 0;  // Should match layer 0
hex.textureID = 1;    // Should match layer 1
// etc.
```

### Problem: Application crashes on startup

**Check**: Texture loading errors in console.

**Fix**: Verify all texture files exist and are valid PNG/JPG files.

### Problem: Black primitives

**Cause**: Texture array not bound correctly.

**Fix**: Check console for "Created texture array with 5 layers" message.

---

## Customization

### Add Your Own Textures

1. Copy texture to `assets/` folder
2. Modify texture list in `initGL()`:
```cpp
textureFiles.push_back("assets/MyTexture.png");  // Layer 5
```
3. Assign to primitive:
```cpp
myPrimitive.textureID = 5;
```

### Change UV Scaling

Modify UV scale for any primitive:
```cpp
torus.uvScale = make_float2(8.0f, 4.0f);  // More tiling
cone.uvScale = make_float2(0.5f, 0.5f);   // Less tiling
```

### Change UV Rotation

```cpp
torus.uvRotation = 0.785f;  // 45 degrees (π/4 radians)
```

### Change UV Offset (Scroll)

```cpp
// In animation loop:
torus.uvOffset.x = fmodf(time * 0.1f, 1.0f);  // Horizontal scroll
```

---

## Debug Visualization

### View UVs as Colors

Temporarily modify fragment shader in `CudaSDFUtil.h`:

```glsl
// Around line 456, replace:
if (useTexture == 1) {
    vec3 texColor = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
    color = texColor * (diff + 0.2);
}

// With:
if (useTexture == 1) {
    color = vec3(TexCoord.x, TexCoord.y, 0.0);  // UV as RG color
}
```

Expected: Red-green gradient showing UV coordinates.

### View Primitive IDs as Colors

```glsl
if (useTexture == 1) {
    color = vec3(float(PrimitiveID) / 5.0, 0.0, 0.0);  // ID as red shade
}
```

Expected: Each primitive appears as different shade of red.

---

## Summary

You now have a complete texture array system running with:

✅ **5 primitives** each with unique textures
✅ **Real-time UV generation** during marching cubes
✅ **Interactive controls** to toggle modes
✅ **High performance** with minimal overhead
✅ **Debug textures** to verify UV mapping correctness

The system is **production-ready** and can be extended with:
- More textures (up to 256 layers on most GPUs)
- Custom UV animations
- Atlas packing mode (Phase 3 - optional)
- Projection baking (already implemented)

**Enjoy your textured SDF primitives!** 🎨


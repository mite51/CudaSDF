# Assets Folder - Texture Requirements

## Texture Array Requirements

For the texture array system to work correctly, **all textures must meet these requirements**:

### ✅ Size Requirements
- **All textures MUST be the same dimensions**
- Current configuration: **4096×4096** pixels
- If textures are different sizes, only the first texture will load and others will be skipped

### ✅ Format Requirements
- **Recommended**: PNG with RGBA (4 channels)
- **Supported**: PNG with RGB (3 channels) - auto-converted to RGBA
- **Not recommended**: JPEG (lossy compression, may have artifacts)

### ✅ Naming Convention
- Use descriptive names (e.g., `T_debug_color_01.PNG`)
- Case-sensitive on some platforms
- No spaces in filenames

---

## Current Textures

| Filename | Size | Channels | Layer | Usage |
|----------|------|----------|-------|-------|
| `T_checkerNumbered.PNG` | 4096×4096 | RGBA | 0 | Torus |
| `T_debug_color_01.PNG` | 4096×4096 | RGBA | 1 | Hex Prism |
| `T_debug_uv_01.PNG` | 4096×4096 | RGBA | 2 | Cone |
| `T_debug_orientation_01.PNG` | 4096×4096 | RGBA | 3 | Cylinder |
| `T_OmniDebugTexture_COL.png` | 4096×4096 | RGB→RGBA | 4 | Sphere |

---

## Adding New Textures

### Step 1: Prepare Texture
1. Resize to match existing textures (4096×4096)
2. Save as PNG
3. Place in this `assets/` folder

### Step 2: Update Code
Edit `src/main.cpp` around line 470:

```cpp
std::vector<std::string> textureFiles = {
    "assets/T_checkerNumbered.PNG",          // Layer 0
    "assets/T_debug_color_01.PNG",           // Layer 1
    "assets/T_debug_uv_01.PNG",              // Layer 2
    "assets/T_debug_orientation_01.PNG",     // Layer 3
    "assets/T_OmniDebugTexture_COL.png",     // Layer 4
    "assets/YourNewTexture.png"              // Layer 5 (NEW)
};
```

### Step 3: Assign to Primitive
Around line 755 in `src/main.cpp`:

```cpp
myPrimitive.textureID = 5;  // Use layer 5
myPrimitive.uvScale = make_float2(2.0f, 2.0f);  // 2x tiling
```

### Step 4: Rebuild
```bash
cd build
cmake ..  # Re-run CMake to copy new assets
cmake --build . --config Release
```

---

## Resizing Textures

### Using ImageMagick (Command Line)
```bash
# Install ImageMagick first
# Windows: choco install imagemagick
# Mac: brew install imagemagick
# Linux: apt-get install imagemagick

# Resize single texture
magick convert input.png -resize 4096x4096 output.png

# Batch resize all PNGs
for file in *.png; do
    magick convert "$file" -resize 4096x4096 "resized_$file"
done
```

### Using GIMP (GUI)
1. Open texture in GIMP
2. Image → Scale Image
3. Set Width: 4096, Height: 4096
4. Interpolation: Cubic (best quality)
5. Click "Scale"
6. File → Export As → PNG

### Using Photoshop (GUI)
1. Open texture
2. Image → Image Size
3. Width: 4096, Height: 4096
4. Resample: Bicubic (best quality)
5. File → Save As → PNG

---

## Performance Considerations

### Memory Usage
- **Per texture**: ~64 MB (4096×4096×4 bytes)
- **5 textures**: ~320 MB
- **With mipmaps**: ~426 MB total
- **Recommended VRAM**: 2GB+ for texture array

### Resolution Alternatives

If you need to reduce memory usage:

| Resolution | Memory per Texture | Total (5 textures) |
|------------|-------------------|-------------------|
| 4096×4096 | 64 MB | 320 MB |
| 2048×2048 | 16 MB | 80 MB |
| 1024×1024 | 4 MB | 20 MB |
| 512×512 | 1 MB | 5 MB |

To change resolution:
1. Resize **all** textures to same new size
2. Textures will load automatically at new size
3. No code changes needed!

---

## Debug Textures

The current debug textures help visualize UV mapping:

- **Checker**: Shows tiling and seams
- **Color**: Gradient helps identify orientation
- **UV Grid**: Numbers show U/V coordinates
- **Orientation**: Arrows show surface direction
- **Omni**: All-around test pattern

These are excellent for testing but can be replaced with your own textures for production use.

---

## Troubleshooting

### "Texture size mismatch" Error
**Problem**: Textures are different sizes  
**Solution**: Resize all textures to match the first texture

### "Channel mismatch" Warning
**Problem**: Mixing RGB and RGBA textures  
**Solution**: System auto-converts, but for best results convert all to RGBA

### "Failed to load texture" Error
**Problem**: File not found or corrupted  
**Solution**: 
- Check filename spelling and case
- Verify file is valid PNG
- Check file permissions

### All Primitives Show Same Texture
**Problem**: Texture IDs not set correctly  
**Solution**: Check `primitive.textureID` assignments in `main.cpp`

### Low Frame Rate
**Problem**: Textures too large for GPU  
**Solution**: Reduce texture resolution to 2048×2048 or 1024×1024

---

## Test Pattern (Legacy)

`test_pattern.jpg` is used for the legacy projection baking feature (press 'P' in app). It's independent of the texture array system and doesn't need to match the array texture sizes.

---

## License & Attribution

If using third-party textures, ensure you have proper licenses and provide attribution as required.

For the debug textures included:
- Generally free for testing and development
- Check original sources for production use rights


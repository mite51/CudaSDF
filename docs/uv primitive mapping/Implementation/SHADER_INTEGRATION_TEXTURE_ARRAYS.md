# Shader Integration for Multi-Texture Support

## Overview

This document details how to integrate OpenGL texture arrays to support multiple textures in **Direct Mode** for the primitive UV mapping system.

## Problem Statement

In direct mode, each primitive maintains its own [0,1] UV coordinates. When multiple primitives are present in a scene, each may need a different texture (e.g., torus=metal, cylinder=wood, sphere=stone). 

**Challenge**: How does a single shader handle multiple textures without separate draw calls?

**Solution**: OpenGL 2D Texture Arrays (`sampler2DArray`)

---

## Texture Array Approach

### Concept

A 2D texture array is essentially a stack of 2D textures, all with the same resolution, accessed by a layer index:

```
Layer 0: Metal texture (1024x1024)
Layer 1: Wood texture (1024x1024)
Layer 2: Stone texture (1024x1024)
Layer 3: Fabric texture (1024x1024)
...
```

Each primitive has a `textureID` field that specifies which layer to use. This ID is passed to the fragment shader, which samples: `texture(textureArray, vec3(uv.xy, layerIndex))`.

### Benefits

1. **Single draw call**: Entire mesh rendered in one pass
2. **Per-primitive textures**: Each primitive can have unique material
3. **Efficient**: No texture binding overhead
4. **Simple shader**: Straightforward texture lookup

### Limitations

1. **Same resolution**: All textures must be same size (typically 1024x1024 or 2048x2048)
2. **Layer limit**: GL_MAX_ARRAY_TEXTURE_LAYERS (typically 256-2048 on modern GPUs)
3. **Memory**: All textures resident in VRAM simultaneously
4. **OpenGL 3.0+**: Requires relatively modern OpenGL (not an issue for desktop)

---

## Implementation Details

### 1. Data Flow

```
Primitive Definition (CPU)
    ├─ textureID = 2 (stone texture)
    ├─ uvScale, uvOffset, uvRotation
    └─ ... other primitive data

Marching Cubes Kernel (GPU)
    ├─ Generates vertex position
    ├─ Computes UV from primitive type
    ├─ Identifies which primitive (primitiveID)
    └─ Writes: position, color, UV, primitiveID

Vertex Shader (GPU)
    ├─ Receives: position, color, UV, primitiveID
    ├─ Transforms position to screen space
    └─ Passes to fragment shader (primitiveID as flat/no interpolation)

Fragment Shader (GPU)
    ├─ Receives: UV, primitiveID
    ├─ Samples: texture(textureArray, vec3(UV.xy, float(primitiveID)))
    └─ Outputs: final color
```

**Key Point**: `primitiveID` is the index of the primitive in the scene, which maps directly to the texture layer (since each primitive's `textureID` corresponds to its layer).

---

## Shader Code

### Vertex Shader

```glsl
#version 330 core

// Vertex attributes
layout (location = 0) in vec4 aPos;         // Position (xyz) + w
layout (location = 1) in vec4 aColor;       // Vertex color
layout (location = 2) in vec2 aTexCoord;    // UV coordinates
layout (location = 3) in float aPrimitiveID;  // NEW: Primitive ID

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 FragPosWorld;
out vec2 TexCoord;
out vec4 VertColor;
flat out int PrimitiveID;  // NEW: 'flat' = no interpolation (integer)

void main() {
    vec4 worldPos = model * vec4(aPos.xyz, 1.0);
    gl_Position = projection * view * worldPos;
    
    FragPos = aPos.xyz;
    FragPosWorld = worldPos.xyz;
    TexCoord = aTexCoord;
    VertColor = aColor;
    PrimitiveID = int(aPrimitiveID);  // Cast to int
}
```

**Important**: 
- `aPrimitiveID` is a float (OpenGL vertex attributes are floats)
- We cast to `int` and pass with `flat` interpolation
- `flat` means no interpolation across triangle - all fragments get same value

---

### Fragment Shader

```glsl
#version 330 core

out vec4 FragColor;

// Inputs from vertex shader
in vec3 FragPos;
in vec3 FragPosWorld;
in vec2 TexCoord;
in vec4 VertColor;
flat in int PrimitiveID;  // NEW: Primitive ID (flat = no interpolation)

// Uniforms
uniform float time;
uniform sampler2DArray textureArray;  // NEW: 2D texture array
uniform sampler2D atlasTexture;       // For atlas mode
uniform int useTexture;  // 0=SDF color, 1=Texture array, 2=Atlas
uniform vec3 lightDir = vec3(1.0, 1.0, 1.0);

// ... other uniforms (camera, SDF data) ...

void main() {
    // Calculate lighting
    vec3 normal = normalize(cross(dFdx(FragPosWorld), dFdy(FragPosWorld)));
    float diffuse = max(dot(normalize(normal), normalize(lightDir)), 0.0) + 0.2;
    
    vec3 color;
    
    if (useTexture == 1) {
        // DIRECT MODE: Sample texture array using primitive ID as layer
        vec3 texColor = texture(textureArray, vec3(TexCoord.xy, float(PrimitiveID))).rgb;
        color = texColor * diffuse;
        
    } else if (useTexture == 2) {
        // ATLAS MODE: Sample single texture (UVs already remapped)
        vec3 texColor = texture(atlasTexture, TexCoord).rgb;
        color = texColor * diffuse;
        
    } else {
        // SDF COLOR MODE: Use vertex colors
        color = VertColor.rgb * diffuse;
    }
    
    FragColor = vec4(color, 1.0);
}
```

**Note**: `texture(textureArray, vec3(u, v, layer))` - layer is a float, automatically rounded to nearest integer.

---

## OpenGL Setup

### Create Texture Array

```cpp
GLuint textureArray;
int numLayers = 8;      // Max number of different textures
int texResolution = 1024;  // All textures must be this size

// Generate and bind
glGenTextures(1, &textureArray);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);

// Allocate storage for all layers
glTexImage3D(
    GL_TEXTURE_2D_ARRAY,     // Target
    0,                        // Mipmap level
    GL_RGB8,                  // Internal format
    texResolution,            // Width
    texResolution,            // Height
    numLayers,                // Number of layers (depth)
    0,                        // Border (must be 0)
    GL_RGB,                   // Format
    GL_UNSIGNED_BYTE,         // Type
    nullptr                   // Data (null = allocate only)
);

// Set texture parameters
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
```

### Load Textures to Layers

```cpp
#include "stb_image.h"

for (int layer = 0; layer < numLayers; ++layer) {
    const char* filename = textureFilenames[layer];
    
    // Load image
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);
    
    if (!data) {
        std::cerr << "Failed to load: " << filename << std::endl;
        continue;
    }
    
    // Upload to layer
    glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
    glTexSubImage3D(
        GL_TEXTURE_2D_ARRAY,     // Target
        0,                        // Mipmap level
        0, 0, layer,              // x, y, z offset (layer is z)
        width, height, 1,         // Width, height, depth (1 = single layer)
        GL_RGB,                   // Format
        GL_UNSIGNED_BYTE,         // Type
        data                      // Image data
    );
    
    stbi_image_free(data);
    
    std::cout << "Loaded '" << filename << "' to layer " << layer << std::endl;
}

// Generate mipmaps after all textures loaded
glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
```

---

## Primitive ID Vertex Attribute

### Create Buffer

```cpp
// In initGL():

GLuint primidbo;
glGenBuffers(1, &primidbo);

glBindVertexArray(vao);  // Bind your VAO first

glBindBuffer(GL_ARRAY_BUFFER, primidbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);

// Setup vertex attribute (location 3)
glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
glEnableVertexAttribArray(3);

glBindVertexArray(0);
```

### Register with CUDA

```cpp
struct cudaGraphicsResource* cudaPrimIDBO;

cudaError_t err = cudaGraphicsGLRegisterBuffer(
    &cudaPrimIDBO, 
    primidbo, 
    cudaGraphicsRegisterFlagsWriteDiscard
);

if (err != cudaSuccess) {
    std::cerr << "CUDA Error Register PrimID: " << cudaGetErrorString(err) << std::endl;
    exit(-1);
}
```

### Write from CUDA Kernel

The marching cubes kernel already writes primitive IDs to `grid.d_primitiveIDs`. Ensure this buffer is mapped to `primidbo`:

```cpp
// In main loop, map buffer:
float* d_primIDPtr;
cudaGraphicsMapResources(1, &cudaPrimIDBO, 0);
cudaGraphicsResourceGetMappedPointer((void**)&d_primIDPtr, &size, cudaPrimIDBO);

// Call update with primitive ID buffer
mesh.Update(time, d_vbo, d_cbo, d_ibo, d_uvbo, (int*)d_primIDPtr);  // Cast is ok, writes ints

cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);
```

**Note**: Kernel writes `int`, but OpenGL reads as `float`. The bit pattern is preserved and we cast back to `int` in the vertex shader.

---

## Texture Management

### TextureArrayManager Class

```cpp
class TextureArrayManager {
public:
    TextureArrayManager(int maxLayers = 8, int resolution = 1024) 
        : m_maxLayers(maxLayers), m_resolution(resolution) {
        
        glGenTextures(1, &m_textureArray);
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 
                     m_resolution, m_resolution, m_maxLayers,
                     0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    
    ~TextureArrayManager() {
        if (m_textureArray) glDeleteTextures(1, &m_textureArray);
    }
    
    int LoadTextureToLayer(const std::string& filename, int layer) {
        if (layer >= m_maxLayers) return -1;
        
        int w, h, c;
        unsigned char* data = stbi_load(filename.c_str(), &w, &h, &c, 3);
        if (!data) {
            std::cerr << "Failed to load: " << filename << std::endl;
            return -1;
        }
        
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, w, h, 1, 
                        GL_RGB, GL_UNSIGNED_BYTE, data);
        
        stbi_image_free(data);
        m_layerFiles[layer] = filename;
        return layer;
    }
    
    void GenerateMipmaps() {
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }
    
    void Bind(int unit = 0) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
    }
    
    GLuint GetID() const { return m_textureArray; }
    
private:
    GLuint m_textureArray = 0;
    int m_maxLayers;
    int m_resolution;
    std::map<int, std::string> m_layerFiles;
};
```

### Usage in main()

```cpp
// Global
TextureArrayManager* g_texMgr = nullptr;

// In main(), after initGL():
g_texMgr = new TextureArrayManager(8, 1024);

// Load textures
g_texMgr->LoadTextureToLayer("assets/metal.jpg", 0);
g_texMgr->LoadTextureToLayer("assets/wood.jpg", 1);
g_texMgr->LoadTextureToLayer("assets/stone.jpg", 2);
g_texMgr->LoadTextureToLayer("assets/fabric.jpg", 3);
g_texMgr->GenerateMipmaps();

// Assign to primitives
scenePrimitives[0].textureID = 0;  // Torus = metal
scenePrimitives[1].textureID = 1;  // Hex = wood
scenePrimitives[2].textureID = 2;  // Cone = stone
scenePrimitives[3].textureID = 3;  // Cylinder = fabric

// In render loop:
g_texMgr->Bind(0);
glUniform1i(glGetUniformLocation(shader, "textureArray"), 0);
glUniform1i(glGetUniformLocation(shader, "useTexture"), 1);  // Direct mode
```

---

## Rendering Modes

The system supports three modes via `useTexture` uniform:

### Mode 0: SDF Color
- Uses vertex colors from SDF evaluation
- No textures needed
- Default mode

```cpp
glUniform1i(glGetUniformLocation(shader, "useTexture"), 0);
```

### Mode 1: Direct Mode (Texture Array)
- Each primitive has its own texture layer
- UVs in [0,1] per primitive
- Primitive ID selects texture layer

```cpp
g_texMgr->Bind(0);
glUniform1i(glGetUniformLocation(shader, "textureArray"), 0);
glUniform1i(glGetUniformLocation(shader, "useTexture"), 1);
```

### Mode 2: Atlas Mode
- Single texture with packed UV islands
- UVs remapped to atlas space
- Standard 2D texture sampler

```cpp
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, atlasTextureID);
glUniform1i(glGetUniformLocation(shader, "atlasTexture"), 0);
glUniform1i(glGetUniformLocation(shader, "useTexture"), 2);
```

---

## Key Binding Example

```cpp
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_T) {
            // Toggle texture array mode
            g_useTextureArray = !g_useTextureArray;
            std::cout << "Texture array: " << (g_useTextureArray ? "ON" : "OFF") << std::endl;
        }
        if (key == GLFW_KEY_U) {
            // Trigger UV generation
            g_triggerUnwrap = true;
        }
        if (key == GLFW_KEY_A) {
            // Trigger atlas packing
            g_triggerAtlasMode = true;
        }
    }
}
```

---

## Debugging

### Visualize Primitive IDs

In fragment shader:

```glsl
// Debug: Show primitive ID as color
FragColor = vec4(float(PrimitiveID) / 8.0, 0.0, 0.0, 1.0);
```

Each primitive will appear as a different shade of red.

### Visualize UVs

```glsl
// Debug: Show UVs as colors
FragColor = vec4(TexCoord.x, TexCoord.y, 0.0, 1.0);
```

UVs in [0,1] will appear as a red-green gradient.

### Visualize Texture Layers

```glsl
// Debug: Sample specific layer
vec3 texColor = texture(textureArray, vec3(TexCoord.xy, 0.0)).rgb;  // Force layer 0
FragColor = vec4(texColor, 1.0);
```

---

## Performance Considerations

### Memory Usage

```
Texture Array Memory = width × height × layers × bytesPerPixel

Example:
1024×1024 × 8 layers × 3 bytes (RGB) = 24 MB
+ mipmaps (~33% extra) = 32 MB total
```

This is very reasonable for modern GPUs.

### Texture Switching

- **Without texture array**: N draw calls for N primitives (or manual texture binding per primitive)
- **With texture array**: 1 draw call for entire mesh
- **Performance gain**: Significant for complex scenes

### Mipmap Generation

- Call `glGenerateMipmap()` once after loading all layers
- Enables proper filtering at distance
- Minimal overhead

---

## Comparison: Texture Array vs Atlas

| Feature | Texture Array | Atlas |
|---------|---------------|-------|
| **Draw calls** | 1 | 1 |
| **Texture binding** | 1 | 1 |
| **UV space** | [0,1] per primitive | [0,1] global |
| **Texture resolution** | Must match | Flexible |
| **Memory** | Fixed per layer | Flexible |
| **Complexity** | Simple | Moderate (packing) |
| **Best for** | Multiple materials | Projection baking |

---

## Summary

1. **Texture arrays** solve the multi-texture problem elegantly for direct mode
2. **Primitive ID** passed as vertex attribute enables per-fragment texture selection
3. **Single draw call** maintains performance
4. **OpenGL 3.0+** widely supported, not a compatibility concern
5. **Easy to implement** - main changes are shader and texture loading

This approach provides the missing piece for direct mode multi-primitive texturing!



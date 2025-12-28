# UV Mapping Integration Guide

This guide shows how to integrate the new UV mapping system into your main application.

---

## Step 1: Setup OpenGL Buffers

Add UV and Primitive ID buffers to your OpenGL initialization:

```cpp
// In initGL() or similar:

GLuint uvbo, primidbo;

// Create UV buffer
glGenBuffers(1, &uvbo);
glBindBuffer(GL_ARRAY_BUFFER, uvbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);

// Create Primitive ID buffer
glGenBuffers(1, &primidbo);
glBindBuffer(GL_ARRAY_BUFFER, primidbo);
glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);

// Setup VAO attributes
glBindVertexArray(vao);

// Vertex positions (location 0) - already exists
// Vertex colors (location 1) - already exists

// UV coordinates (location 2)
glBindBuffer(GL_ARRAY_BUFFER, uvbo);
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);
glEnableVertexAttribArray(2);

// Primitive ID (location 3)
glBindBuffer(GL_ARRAY_BUFFER, primidbo);
glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
glEnableVertexAttribArray(3);

glBindVertexArray(0);
```

---

## Step 2: Register Buffers with CUDA

```cpp
struct cudaGraphicsResource* cudaUVBO;
struct cudaGraphicsResource* cudaPrimIDBO;

cudaError_t err;

err = cudaGraphicsGLRegisterBuffer(&cudaUVBO, uvbo, 
                                   cudaGraphicsRegisterFlagsWriteDiscard);
if (err != cudaSuccess) {
    std::cerr << "CUDA Error Register UV BO: " << cudaGetErrorString(err) << std::endl;
    exit(-1);
}

err = cudaGraphicsGLRegisterBuffer(&cudaPrimIDBO, primidbo, 
                                   cudaGraphicsRegisterFlagsWriteDiscard);
if (err != cudaSuccess) {
    std::cerr << "CUDA Error Register PrimID BO: " << cudaGetErrorString(err) << std::endl;
    exit(-1);
}
```

---

## Step 3: Update Main Render Loop

```cpp
// In main loop, before calling mesh.Update():

// Map CUDA resources
float2* d_uvPtr = nullptr;
int* d_primIDPtr = nullptr;
size_t size;

if (g_enableUVGeneration) {  // Flag to enable/disable UV generation
    cudaGraphicsMapResources(1, &cudaUVBO, 0);
    cudaGraphicsMapResources(1, &cudaPrimIDBO, 0);
    
    cudaGraphicsResourceGetMappedPointer((void**)&d_uvPtr, &size, cudaUVBO);
    cudaGraphicsResourceGetMappedPointer((void**)&d_primIDPtr, &size, cudaPrimIDBO);
}

// Call Update with UV pointers
mesh.Update(time, d_vboPtr, d_cboPtr, d_iboPtr, d_uvPtr, d_primIDPtr);

if (g_enableUVGeneration) {
    cudaGraphicsUnmapResources(1, &cudaUVBO, 0);
    cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);
}
```

---

## Step 4: Setup Texture Array (Optional)

For Direct Mode with multiple textures:

```cpp
// Create texture array manager
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
    
    int LoadTextureToLayer(const std::string& filename, int layer) {
        if (layer >= m_maxLayers) return -1;
        
        int w, h, c;
        unsigned char* data = stbi_load(filename.c_str(), &w, &h, &c, 3);
        if (!data) return -1;
        
        glBindTexture(GL_TEXTURE_2D_ARRAY, m_textureArray);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, w, h, 1, 
                        GL_RGB, GL_UNSIGNED_BYTE, data);
        
        stbi_image_free(data);
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
    
private:
    GLuint m_textureArray = 0;
    int m_maxLayers;
    int m_resolution;
};

// In initialization:
TextureArrayManager* g_textureMgr = new TextureArrayManager(8, 1024);

g_textureMgr->LoadTextureToLayer("assets/metal.jpg", 0);
g_textureMgr->LoadTextureToLayer("assets/wood.jpg", 1);
g_textureMgr->LoadTextureToLayer("assets/stone.jpg", 2);
g_textureMgr->GenerateMipmaps();
```

---

## Step 5: Assign Textures to Primitives

```cpp
// When creating primitives:
SDFPrimitive torus = CreateTorusPrim(...);
torus.uvScale = make_float2(4.0f, 2.0f);  // Tile 4x2
torus.textureID = 0;  // Use layer 0 (metal)

SDFPrimitive cylinder = CreateCylinderPrim(...);
cylinder.uvScale = make_float2(2.0f, 1.0f);
cylinder.textureID = 1;  // Use layer 1 (wood)

mesh.AddPrimitive(torus);
mesh.AddPrimitive(cylinder);
```

---

## Step 6: Update Rendering Code

```cpp
// In render loop:

glUseProgram(shaderProgram);

// ... existing uniforms (model, view, projection) ...

// Bind texture array and set mode
if (g_useTextureArray) {
    g_textureMgr->Bind(0);
    glUniform1i(glGetUniformLocation(shaderProgram, "textureArray"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);  // Direct mode
} else {
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);  // SDF color mode
}

// Draw
glBindVertexArray(vao);
glDrawArrays(GL_TRIANGLES, 0, mesh.GetTotalVertexCount());
```

---

## Step 7: Add Keyboard Controls

```cpp
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_U) {
            g_enableUVGeneration = !g_enableUVGeneration;
            std::cout << "UV generation: " << (g_enableUVGeneration ? "ON" : "OFF") << std::endl;
        }
        if (key == GLFW_KEY_T) {
            g_useTextureArray = !g_useTextureArray;
            std::cout << "Texture array: " << (g_useTextureArray ? "ON" : "OFF") << std::endl;
        }
    }
}
```

---

## Complete Example

```cpp
// Global state
bool g_enableUVGeneration = true;
bool g_useTextureArray = true;
TextureArrayManager* g_textureMgr = nullptr;
GLuint uvbo, primidbo;
cudaGraphicsResource* cudaUVBO = nullptr;
cudaGraphicsResource* cudaPrimIDBO = nullptr;

// Initialization
void init() {
    // ... existing GL setup ...
    
    // Create UV and PrimID buffers
    glGenBuffers(1, &uvbo);
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &primidbo);
    glBindBuffer(GL_ARRAY_BUFFER, primidbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    
    // Setup VAO
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);
    glEnableVertexAttribArray(2);
    
    glBindBuffer(GL_ARRAY_BUFFER, primidbo);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);
    
    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaUVBO, uvbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cudaPrimIDBO, primidbo, cudaGraphicsRegisterFlagsWriteDiscard);
    
    // Setup texture array
    g_textureMgr = new TextureArrayManager(8, 1024);
    g_textureMgr->LoadTextureToLayer("assets/metal.jpg", 0);
    g_textureMgr->LoadTextureToLayer("assets/wood.jpg", 1);
    g_textureMgr->GenerateMipmaps();
    
    // Create primitives with UVs
    SDFPrimitive torus = CreateTorusPrim(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(1.0f, 0.5f, 0.2f),
        SDF_UNION, 0.0f, 0.4f, 0.15f
    );
    torus.uvScale = make_float2(4.0f, 2.0f);
    torus.textureID = 0;
    mesh.AddPrimitive(torus);
}

// Render loop
void render() {
    // Map CUDA resources
    float2* d_uvPtr = nullptr;
    int* d_primIDPtr = nullptr;
    size_t size;
    
    if (g_enableUVGeneration) {
        cudaGraphicsMapResources(1, &cudaUVBO, 0);
        cudaGraphicsMapResources(1, &cudaPrimIDBO, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_uvPtr, &size, cudaUVBO);
        cudaGraphicsResourceGetMappedPointer((void**)&d_primIDPtr, &size, cudaPrimIDBO);
    }
    
    // Update mesh
    mesh.Update(time, d_vboPtr, d_cboPtr, d_iboPtr, d_uvPtr, d_primIDPtr);
    
    if (g_enableUVGeneration) {
        cudaGraphicsUnmapResources(1, &cudaUVBO, 0);
        cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);
    }
    
    // Render
    glUseProgram(shaderProgram);
    
    if (g_useTextureArray && g_enableUVGeneration) {
        g_textureMgr->Bind(0);
        glUniform1i(glGetUniformLocation(shaderProgram, "textureArray"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
    } else {
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
    }
    
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, mesh.GetTotalVertexCount());
}
```

---

## Testing

### 1. UV Visualization Test
Set `useTexture = 0` and modify fragment shader temporarily:
```glsl
color = vec3(TexCoord.x, TexCoord.y, 0.0);
```
You should see smooth UV gradients (red-green) across each primitive.

### 2. Primitive ID Test
```glsl
color = vec3(float(PrimitiveID) / 8.0, 0.0, 0.0);
```
Each primitive should appear as different shade of red.

### 3. Texture Array Test
Load checkerboard textures and verify each primitive displays correct pattern.

---

## Troubleshooting

### Problem: UVs are all (0, 0)
- Check that `d_uvPtr` is not nullptr
- Verify buffer allocation size is sufficient
- Check that `g_enableUVGeneration` is true

### Problem: Black textures
- Verify textures loaded successfully to texture array
- Check texture layer indices match primitive textureID
- Ensure `useTexture = 1` is set correctly

### Problem: Crash in kernel
- Check all grid pointers are valid
- Verify primitive count is correct
- Use cuda-memcheck for detailed error

---

## Performance Tips

1. **UV generation is optional** - Set pointers to nullptr to disable
2. **Texture array size** - Use power-of-2 resolutions (512, 1024, 2048)
3. **Mipmap generation** - Generate once after loading all textures
4. **Buffer reuse** - Don't recreate buffers every frame

---

## Summary

The integration requires:
- ✅ 2 new OpenGL buffers (UV, Primitive ID)
- ✅ CUDA registration of buffers
- ✅ Optional texture array setup
- ✅ Minimal changes to render loop
- ✅ ~50 lines of code total

The system is **backward compatible** - passing nullptr for UV pointers disables UV generation entirely.


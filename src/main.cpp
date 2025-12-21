#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <map>
#include <fstream>

#include "SDFMesh/Commons.cuh"
#include "SDFMesh/CudaSDFMesh.h"
#include "SDFMesh/CudaSDFUtil.h" 
#include "uv_unwrap_harmonic_parameterization/unwrap/unwrap_pipeline.h"
#include "uv_unwrap_harmonic_parameterization/unwrap/seam_splitter.h"
#include "uv_unwrap_harmonic_parameterization/common/mesh.h"

#include "SDFMesh/TextureLoader.h"

// Global variables
const float GRID_SIZE = 128.0f; 
const float CELL_SIZE = 1.0f/GRID_SIZE;
const unsigned int WINDOW_WIDTH = 1024;
const unsigned int WINDOW_HEIGHT = 768;

GLuint shaderProgram;
GLuint projectionShaderProgram; // Shader for projection baking
GLuint vao, vbo, ibo, cbo, uvbo; // Added uvbo for texture coordinates
GLuint g_textureID = 0; // Source texture for projection
GLuint g_bakedTextureID = 0; // Baked texture from projection
GLuint g_fbo = 0; // Framebuffer for projection baking
GLuint g_rbo = 0; // Renderbuffer for depth

struct cudaGraphicsResource* cudaVBO;
struct cudaGraphicsResource* cudaIBO; // Unused but kept for structure compatibility if needed
struct cudaGraphicsResource* cudaCBO;

GLuint uboPrimitives; // UBO Handle
GLuint tboBVH, texBVH; // BVH Texture Buffer

CudaSDFMesh mesh;
bool g_triggerUnwrap = false;
bool g_triggerProjection = false; // Flag to trigger projection baking
bool g_AnimateMesh = true; // Flag to control animation/update
bool g_isUnwrapped = false; // Track if mesh has been unwrapped
bool g_hasProjectedTexture = false; // Track if texture has been projection baked
uint32_t g_indexCount = 0; // Number of indices for indexed drawing

void checkGLErrors(const char* label) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL Error at " << label << ": " << err << std::endl;
    }
}

// Simple projection shader sources
const char* projectionVertexShader = R"(
    #version 330 core
    layout (location = 0) in vec4 aPos;
    layout (location = 2) in vec2 aTexCoord;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec4 worldPos;
    out vec3 worldNormal;
    
    void main() {
        // Render in UV space - UV coords become the position
        gl_Position = vec4(aTexCoord.x * 2.0 - 1.0, aTexCoord.y * 2.0 - 1.0, 0.0, 1.0);
        
        // Pass world position for projection sampling
        worldPos = model * vec4(aPos.xyz, 1.0);
        
        // Calculate normal from derivatives (will be done per-triangle in fragment shader)
        worldNormal = vec3(0.0); // Placeholder, will compute in fragment shader
    }
)";

const char* projectionFragmentShader = R"(
    #version 330 core
    in vec4 worldPos;
    in vec3 worldNormal;
    
    uniform sampler2D sourceTexture;
    uniform mat4 projectionMatrix;
    uniform mat4 viewMatrix;
    uniform vec3 cameraPos;
    
    out vec4 FragColor;
    
    void main() {
        // Calculate surface normal from derivatives
        vec3 dFdxPos = dFdx(worldPos.xyz);
        vec3 dFdyPos = dFdy(worldPos.xyz);
        vec3 normal = normalize(cross(dFdxPos, dFdyPos));
        
        // Calculate view direction (from surface to camera)
        vec3 viewDir = normalize(cameraPos - worldPos.xyz);
        
        // Check if surface is facing the camera (backface culling)
        float facing = dot(normal, viewDir);
        
        // Project world position to screen space of source camera
        vec4 projPos = projectionMatrix * viewMatrix * worldPos;
        
        // Perspective divide
        vec3 ndcPos = projPos.xyz / projPos.w;
        
        // Convert to texture coordinates [0,1]
        vec2 screenUV = ndcPos.xy * 0.5 + 0.5;
        
        // Check if within valid projection range and facing camera
        if (facing < 0.0 ||
            screenUV.x < 0.0 || screenUV.x > 1.0 || 
            screenUV.y < 0.0 || screenUV.y > 1.0 ||
            ndcPos.z < -1.0 || ndcPos.z > 1.0) {
            // Backface or outside projection, use a default color
            FragColor = vec4(0.5, 0.5, 0.5, 1.0);
        } else {
            // Sample the source texture using projected UV
            FragColor = texture(sourceTexture, screenUV);
        }
    }
)";


void checkShaderErrors(GLuint shader, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
    }
}

void checkProgramErrors(GLuint program, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 1024, NULL, infoLog);
        std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
    }
}

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
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, true);
        }
    }
}

void initGL() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA Marching Cubes (Sparse)", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Disable VSync
    glfwSetKeyCallback(window, key_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }

    // Initialize CUDA Device from OpenGL Context
    unsigned int cudaDeviceCount = 0;
    int cudaDevices[16];
    cudaError_t cudaErr = cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 16, cudaGLDeviceListAll);
    
    if (cudaErr == cudaSuccess && cudaDeviceCount > 0) {
        std::cout << "Found " << cudaDeviceCount << " CUDA-GL capable devices." << std::endl;
        // Use the first device
        cudaSetDevice(cudaDevices[0]);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cudaDevices[0]);
        std::cout << "Selected CUDA Device: " << prop.name << std::endl;
        
        // Check if the driver version is sufficient
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        std::cout << "CUDA Driver Version: " << driverVersion << std::endl;
        
        int runtimeVersion = 0;
        cudaRuntimeGetVersion(&runtimeVersion);
        std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;
        
        if (driverVersion < runtimeVersion) {
            std::cerr << "Warning: CUDA driver version (" << driverVersion << ") is less than runtime version (" << runtimeVersion << "). This may cause issues." << std::endl;
        }

    } else {
        std::cout << "Warning: No CUDA-GL device found via cudaGLGetDevices. Falling back to default device 0." << std::endl;
        if (cudaErr != cudaSuccess) {
             std::cout << "cudaGLGetDevices error: " << cudaGetErrorString(cudaErr) << std::endl;
        }
        cudaSetDevice(0);
    }
    
    // Build shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkShaderErrors(vertexShader, "VERTEX");
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderErrors(fragmentShader, "FRAGMENT");
    
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkProgramErrors(shaderProgram, "PROGRAM");
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // Build projection shader
    GLuint projVertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(projVertShader, 1, &projectionVertexShader, NULL);
    glCompileShader(projVertShader);
    checkShaderErrors(projVertShader, "PROJECTION_VERTEX");
    
    GLuint projFragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(projFragShader, 1, &projectionFragmentShader, NULL);
    glCompileShader(projFragShader);
    checkShaderErrors(projFragShader, "PROJECTION_FRAGMENT");
    
    projectionShaderProgram = glCreateProgram();
    glAttachShader(projectionShaderProgram, projVertShader);
    glAttachShader(projectionShaderProgram, projFragShader);
    glLinkProgram(projectionShaderProgram);
    checkProgramErrors(projectionShaderProgram, "PROJECTION_PROGRAM");
    
    glDeleteShader(projVertShader);
    glDeleteShader(projFragShader);
    
    // Buffers
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &cbo);
    glGenBuffers(1, &ibo);
    glGenBuffers(1, &uvbo); // Buffer for texture coordinates

    // Load Texture
    g_textureID = LoadTexture("test_pattern.jpg");
    
    glBindVertexArray(vao);
    
    // VBO (float4 positions)
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    unsigned int maxVerts = 20000000; // 20M vertices buffer for high res surface
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 4, NULL, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // CBO (float4 colors)
    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 4, NULL, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    // UVBO (vec2 texture coordinates, location 2)
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    // Don't enable yet - will enable after unwrapping
    // glEnableVertexAttribArray(2);
    
    // IBO (Unused in Sparse Mode, but allocated to keep interop happy if strictly required, or just small)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    unsigned int maxIndices = 20000000 * 3; // Enough for indexed mesh
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, maxIndices * sizeof(unsigned int), NULL, GL_DYNAMIC_DRAW);
    
    // Register with CUDA
    cudaError_t errVBO = cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (errVBO != cudaSuccess) { 
        std::cerr << "CUDA Error Register VBO: " << cudaGetErrorString(errVBO) << std::endl; 
        exit(-1); 
    }
    cudaError_t errCBO = cudaGraphicsGLRegisterBuffer(&cudaCBO, cbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (errCBO != cudaSuccess) { 
        std::cerr << "CUDA Error Register CBO: " << cudaGetErrorString(errCBO) << std::endl; 
        exit(-1); 
    }
    
    glBindVertexArray(0);
    
    // Create UBO
    glGenBuffers(1, &uboPrimitives);
    glBindBuffer(GL_UNIFORM_BUFFER, uboPrimitives);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GPUPrimitive) * 64, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Create BVH TBO
    glGenBuffers(1, &tboBVH);
    glGenTextures(1, &texBVH);
    glBindBuffer(GL_TEXTURE_BUFFER, tboBVH);
    // Allocate initial size
    glBufferData(GL_TEXTURE_BUFFER, 1024 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, texBVH);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tboBVH);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    
    GLuint blockIndex = glGetUniformBlockIndex(shaderProgram, "Primitives");
    if(blockIndex != GL_INVALID_INDEX) {
        glUniformBlockBinding(shaderProgram, blockIndex, 0);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboPrimitives);
    } else {
        std::cout << "Warning: Uniform block 'Primitives' not found" << std::endl;
    }
    
    checkGLErrors("initGL");
}

// Helpers for unwrap
struct VertexKey {
    int x, y, z;
    bool operator<(const VertexKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

uv::Mesh WeldMesh(const std::vector<float4>& rawVerts) {
    uv::Mesh mesh;
    std::map<VertexKey, uint32_t> uniqueMap;
    // Quantize factor for welding
    float q = 10000.0f; 
    
    for(size_t i=0; i<rawVerts.size(); ++i) {
        uv::vec3 v = {rawVerts[i].x, rawVerts[i].y, rawVerts[i].z};
        VertexKey key = { (int)(v.x * q), (int)(v.y * q), (int)(v.z * q) };
        
        if (uniqueMap.find(key) == uniqueMap.end()) {
            uniqueMap[key] = (uint32_t)mesh.V.size();
            mesh.V.push_back(v);
        }
        uint32_t idx = uniqueMap[key];
        
        // Add to triangles
        // We know input is triangle soup, so every 3 verts = 1 tri
        if (i % 3 == 0) mesh.F.push_back({0,0,0});
        
        uint32_t& triIdx = (i % 3 == 0) ? mesh.F.back().x : ((i % 3 == 1) ? mesh.F.back().y : mesh.F.back().z);
        triIdx = idx;
    }
    return mesh;
}

void PerformUnwrap(const std::vector<float4>& rawVerts) {
    if (rawVerts.empty()) {
        std::cout << "No vertices to unwrap!" << std::endl;
        return;
    }
    
    std::cout << "Welding " << rawVerts.size() << " vertices..." << std::endl;
    uv::Mesh mesh = WeldMesh(rawVerts);
    std::cout << "Weld result: " << mesh.V.size() << " vertices, " << mesh.F.size() << " triangles." << std::endl;
    
    uv::UnwrapConfig cfg;
    cfg.atlasMaxSize = 8192; // Increased to reduce packing failure chance
    cfg.paddingPx = 2;       // Reduced padding to save space
    
    uv::UnwrapPipeline pipeline;
    uv::UnwrapResult res = pipeline.Run(mesh, cfg);
    
    // Split vertices to handle hard UV seams
    uv::Mesh splitMesh;
    uv::UnwrapResult splitRes;
    uv::SplitVerticesByChart(mesh, res, splitMesh, splitRes);
    
    // Prepare CPU buffers
    std::vector<float4> unwrappedVerts(splitMesh.V.size());
    std::vector<float2> unwrappedUVs(splitMesh.V.size());
    for(size_t i=0; i<splitMesh.V.size(); ++i) {
        unwrappedVerts[i] = make_float4(splitMesh.V[i].x, splitMesh.V[i].y, splitMesh.V[i].z, 1.0f);
        unwrappedUVs[i] = make_float2(splitRes.uvAtlas[i].x, splitRes.uvAtlas[i].y);
    }
    
    // Flatten Indices (for IBO)
    std::vector<uint32_t> unwrappedIndices(splitMesh.F.size() * 3);
    if (!splitMesh.F.empty()) {
        memcpy(unwrappedIndices.data(), splitMesh.F.data(), splitMesh.F.size() * sizeof(uv::uvec3));
    }
    
    // Update main VAO buffers in-place
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, unwrappedVerts.size() * sizeof(float4), unwrappedVerts.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    glBufferData(GL_ARRAY_BUFFER, unwrappedUVs.size() * sizeof(float2), unwrappedUVs.data(), GL_STATIC_DRAW);
    
    // Enable UV attribute array now that we have valid data
    glBindVertexArray(vao);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, unwrappedIndices.size() * sizeof(uint32_t), unwrappedIndices.data(), GL_STATIC_DRAW);
    
    // Store count for indexed rendering
    g_indexCount = (uint32_t)unwrappedIndices.size();
    g_isUnwrapped = true;
    
    std::cout << "Mesh unwrapped and updated in-place." << std::endl;
}

void PerformProjectionBaking(const float* model, const float* view, const float* projection, int textureSize = 2048) {
    if (!g_isUnwrapped) {
        std::cout << "Mesh must be unwrapped first before projection baking!" << std::endl;
        return;
    }
    
    std::cout << "Starting projection baking to " << textureSize << "x" << textureSize << " texture..." << std::endl;
    
    // Calculate camera position from view matrix
    // View matrix is inverse of camera transform, so we need to extract camera position
    // For a view matrix that translates by (0, 0, -3), camera is at (0, 0, 3)
    float cameraPos[3] = {0.0f, 0.0f, 3.0f}; // This matches the view matrix in main
    
    // Create baked texture if not exists
    if (g_bakedTextureID == 0) {
        glGenTextures(1, &g_bakedTextureID);
        glBindTexture(GL_TEXTURE_2D, g_bakedTextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureSize, textureSize, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    
    // Create FBO if not exists
    if (g_fbo == 0) {
        glGenFramebuffers(1, &g_fbo);
        glGenRenderbuffers(1, &g_rbo);
        
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
        
        // Attach texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_bakedTextureID, 0);
        
        // Setup depth renderbuffer
        glBindRenderbuffer(GL_RENDERBUFFER, g_rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, textureSize, textureSize);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_rbo);
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer is not complete!" << std::endl;
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return;
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    // Render to texture using UV coordinates
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, textureSize, textureSize);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    
    glUseProgram(projectionShaderProgram);
    
    // Pass the current camera matrices for projection
    glUniformMatrix4fv(glGetUniformLocation(projectionShaderProgram, "model"), 1, GL_FALSE, model);
    glUniformMatrix4fv(glGetUniformLocation(projectionShaderProgram, "view"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(projectionShaderProgram, "projection"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(projectionShaderProgram, "projectionMatrix"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(projectionShaderProgram, "viewMatrix"), 1, GL_FALSE, view);
    glUniform3fv(glGetUniformLocation(projectionShaderProgram, "cameraPos"), 1, cameraPos);
    
    // Bind source texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_textureID);
    glUniform1i(glGetUniformLocation(projectionShaderProgram, "sourceTexture"), 0);
    
    // Draw the mesh
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, (GLsizei)g_indexCount, GL_UNSIGNED_INT, 0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    g_hasProjectedTexture = true;
    std::cout << "Projection baking complete!" << std::endl;
    
    checkGLErrors("PerformProjectionBaking");
}


int main() {
    initGL();
    mesh.Initialize(CELL_SIZE, make_float3(-1.1f, -1.1f, -1.1f), make_float3(1.1f, 1.1f, 1.1f));
    
    GLFWwindow* window = glfwGetCurrentContext();

    // Create Scene (Primitives Showcase)
    std::vector<SDFPrimitive> scenePrimitives;
    
    // 1. Torus (green)
    SDFPrimitive torus = CreateTorusPrim(
        make_float3(-0.5f, 0.5f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(0.2f, 0.8f, 0.2f),
        SDF_UNION,
        0.0f,
        0.4f,
        0.15f
    );
    scenePrimitives.push_back(torus);

    // 2. Hex Prism (Red)(Twisted)
    SDFPrimitive hex = CreateHexPrismPrim(
        make_float3(0.5f, 0.5f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(1.0f, 0.2f, 0.2f),
        SDF_UNION,
        0.0f,
        0.3f,
        0.4f
    );
    hex.displacement = DISP_TWIST;
    hex.dispParams[0] = 2.0f; // Initial Twist
    scenePrimitives.push_back(hex);

    // 3. Rounded Cone (Blue)
    SDFPrimitive cone = CreateRoundedConePrim(
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(0.2f, 0.2f, 1.0f),
        SDF_UNION,
        0.0f,
        0.5f,
        0.3f,
        0.1f
    );
    scenePrimitives.push_back(cone);

    // 4. Rounded Cylinder (Yellow)
    SDFPrimitive cyl = CreateRoundedCylinderPrim(
        make_float3(0.5f, -0.5f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(1.0f, 1.0f, 0.2f),
        SDF_UNION,
        0.0f,
        0.4f,
        0.2f,
        0.1f
    );
    scenePrimitives.push_back(cyl);

    // 5. Sphere (Purple, Subtracted)
    SDFPrimitive sphere = CreateSpherePrim(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(0.5f, 0.0f, 0.5f),
        SDF_SUBTRACT,
        0.0f,
        0.6f
    );
    scenePrimitives.push_back(sphere);

    // Upload UBO once (initial)
    std::vector<GPUPrimitive> gpuPrimitives;
    packPrimitives(scenePrimitives, gpuPrimitives);
    glBindBuffer(GL_UNIFORM_BUFFER, uboPrimitives);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(GPUPrimitive) * gpuPrimitives.size(), gpuPrimitives.data());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // Matrices
    float model[16] = {
        1,0,0,0, 
        0,1,0,0, 
        0,0,1,0, 
        0,0,0,1
    };
    
    float view[16] = {
        1,0,0,0, 
        0,1,0,0, 
        0,0,1,0, 
        0,0,-3,1 
    };

    // Standard Perspective Projection
    float fov = 45.0f * (3.14159f / 180.0f);
    float aspect = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;
    float zNear = 0.1f;
    float zFar = 100.0f;
    float f = 1.0f / tan(fov / 2.0f);
    
    float projection[16] = {0};
    projection[0] = f / aspect;
    projection[5] = f;
    projection[10] = (zFar + zNear) / (zNear - zFar);
    projection[11] = -1.0f;
    projection[14] = (2.0f * zFar * zNear) / (zNear - zFar);

    glDisable(GL_CULL_FACE); // Disable culling to see both sides
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    double lastFPSTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastFPSTime >= 1.0) {
            std::string title = "CUDA Marching Cubes (Sparse) - " + std::to_string(frameCount) + " FPS";
            glfwSetWindowTitle(window, title.c_str());
            frameCount = 0;
            lastFPSTime = currentTime;
        }

        float time = (float)currentTime;
        
        // Print stats every 1 second
        static float lastPrintTime = 0.0f;
        if (time - lastPrintTime > 1.0f) {
            CudaSDFMesh::MemStats stats = mesh.GetMemoryStats();
            std::cout << "GPU Mem: Used " << (stats.usedGPU / 1024 / 1024) << " MB, Free " << (stats.freeGPU / 1024 / 1024) << " MB | Blocks: " << stats.activeBlocks << "/" << stats.allocatedBlocks << std::endl;
            lastPrintTime = time;
        }
        
        // Animate Primitives
        if (g_AnimateMesh) {
            scenePrimitives[0].dispParams[1] = 5.0f * sin(time*0.1f);
            float c_rot = cos(time), s_rot = sin(time);
            scenePrimitives[1].rotation = make_float4(0.0f, s_rot, 0.0f, c_rot);
            scenePrimitives[2].position.y = -0.5f + 0.2f * sin(time * 2.0f);
            scenePrimitives[3].params[2] = 0.1f + 0.05f * sin(time * 3.0f);

            // Update Mesh
            mesh.ClearPrimitives();
            for(const auto& p : scenePrimitives) {
                mesh.AddPrimitive(p);
            }
            
            // Map GL buffers
            if (cudaGraphicsMapResources(1, &cudaVBO, 0) != cudaSuccess) break;
            if (cudaGraphicsMapResources(1, &cudaCBO, 0) != cudaSuccess) break;
            // if (cudaGraphicsMapResources(1, &cudaIBO, 0) != cudaSuccess) break;
            
            float4* d_vboPtr;
            float4* d_cboPtr;
            unsigned int* d_iboPtr = nullptr;
            size_t size;
            
            cudaGraphicsResourceGetMappedPointer((void**)&d_vboPtr, &size, cudaVBO);
            cudaGraphicsResourceGetMappedPointer((void**)&d_cboPtr, &size, cudaCBO);
            // cudaGraphicsResourceGetMappedPointer((void**)&d_iboPtr, &size, cudaIBO);
            
            mesh.Update(time, d_vboPtr, d_cboPtr, d_iboPtr);

            cudaGraphicsUnmapResources(1, &cudaVBO, 0);
            cudaGraphicsUnmapResources(1, &cudaCBO, 0);
            // cudaGraphicsUnmapResources(1, &cudaIBO, 0);
        }

        // --- UV Unwrap Trigger ---
        if (g_triggerUnwrap) {
            g_triggerUnwrap = false;
            
            // Map the *current* buffers to read the current mesh state for unwrap
            // We assume the mesh was updated at least once before this.
            if (cudaGraphicsMapResources(1, &cudaVBO, 0) != cudaSuccess) break;
            
            float4* d_vboPtr;
            size_t size;
            cudaGraphicsResourceGetMappedPointer((void**)&d_vboPtr, &size, cudaVBO);
            
            unsigned int count = mesh.GetTotalVertexCount();
            if (count > 0) {
                std::vector<float4> hostVerts(count);
                cudaMemcpy(hostVerts.data(), d_vboPtr, count * sizeof(float4), cudaMemcpyDeviceToHost);
                
                // Unmap before doing heavy CPU work
                cudaGraphicsUnmapResources(1, &cudaVBO, 0); 
                
                PerformUnwrap(hostVerts);
                
                // After unwrap, stop updating the mesh so we can see the result
                g_AnimateMesh = false;
            } else {
                std::cout << "Mesh is empty, cannot unwrap." << std::endl;
                cudaGraphicsUnmapResources(1, &cudaVBO, 0);
            }
        }
        // -------------------------
        
        // --- Projection Baking Trigger ---
        if (g_triggerProjection) {
            g_triggerProjection = false;
            PerformProjectionBaking(model, view, projection);
            // Restore viewport after baking
            glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        // ---------------------------------

        // Update UBO from Mesh (it might have sorted them)
        const auto& currentGPUPrimitives = mesh.GetGPUPrimitives();
        glBindBuffer(GL_UNIFORM_BUFFER, uboPrimitives);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(GPUPrimitive) * currentGPUPrimitives.size(), currentGPUPrimitives.data());
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        
        // Update TBO from Mesh
        const auto& bvhNodes = mesh.GetBVHNodes();
        glBindBuffer(GL_TEXTURE_BUFFER, tboBVH);
        std::vector<float> bvhData;
        bvhData.resize(bvhNodes.size() * 8); 
        for(size_t i=0; i<bvhNodes.size(); ++i) {
            bvhData[i*8+0] = bvhNodes[i].min[0];
            bvhData[i*8+1] = bvhNodes[i].min[1];
            bvhData[i*8+2] = bvhNodes[i].min[2];
            bvhData[i*8+3] = bvhNodes[i].max[0];
            
            bvhData[i*8+4] = bvhNodes[i].max[1];
            bvhData[i*8+5] = bvhNodes[i].max[2];
            
            int left = bvhNodes[i].left;
            int right = bvhNodes[i].right;
            bvhData[i*8+6] = *(float*)&left;
            bvhData[i*8+7] = *(float*)&right;
        }
        glBufferData(GL_TEXTURE_BUFFER, bvhData.size() * sizeof(float), bvhData.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
        
        // Render
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        
        glUseProgram(shaderProgram);
        
        // Set time
        glUniform1f(glGetUniformLocation(shaderProgram, "time"), time);
        glUniform1i(glGetUniformLocation(shaderProgram, "uNumBVHPrimitives"), mesh.GetNumBVHPrimitives());
        glUniform1i(glGetUniformLocation(shaderProgram, "uNumTotalPrimitives"), (int)currentGPUPrimitives.size());
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, texBVH);
        glUniform1i(glGetUniformLocation(shaderProgram, "bvhNodes"), 0);
        
        // Set matrices
        float c = cos(time), s = sin(time);
        model[0] = c; model[2] = -s;
        model[8] = s; model[10] = c;
        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, model);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, view);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, projection);
        
        // Always bind texture1 sampler to avoid GL_INVALID_OPERATION
        glActiveTexture(GL_TEXTURE1);
        if (g_hasProjectedTexture) {
            glBindTexture(GL_TEXTURE_2D, g_bakedTextureID);
        } else {
            glBindTexture(GL_TEXTURE_2D, g_textureID); // Bind source texture (won't be used unless useTexture=1)
        }
        glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 1);

        glBindVertexArray(vao);
        
        if (g_isUnwrapped) {
            // After unwrap, still use vertex colors unless projection baked
            if (g_hasProjectedTexture) {
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
            } else {
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
            }
            glDrawElements(GL_TRIANGLES, (GLsizei)g_indexCount, GL_UNSIGNED_INT, 0);
        } else {
            glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
            // Switch to DrawArrays for Triangle Soup
            glDrawArrays(GL_TRIANGLES, 0, mesh.GetTotalVertexCount());
        }
        
        checkGLErrors("draw");

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glfwTerminate();
    return 0;
}

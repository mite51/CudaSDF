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

// ImGui
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "SDFMesh/Commons.cuh"
#include "SDFMesh/CudaSDFMesh.h"
#include "SDFMesh/CudaSDFUtil.h" 
#include "SDFMesh/GridUVPacker.cuh"
#include "uv_unwrap_harmonic_parameterization/unwrap/unwrap_pipeline.h"
#include "uv_unwrap_harmonic_parameterization/unwrap/seam_splitter.h"
#include "uv_unwrap_harmonic_parameterization/common/mesh.h"

#include "SDFMesh/TextureLoader.h"
#include "Camera.h"

// Global variables
float g_gridSize = 32.0f;  // Mutable grid size (adjustable in UI)
float g_cellSize = 1.0f / g_gridSize;  // Computed from grid size
const unsigned int WINDOW_WIDTH = 1024;
const unsigned int WINDOW_HEIGHT = 768;

GLuint shaderProgram;
GLuint projectionShaderProgram; // Shader for projection baking
GLuint vao, vbo, ibo, cbo, uvbo; // Added uvbo for texture coordinates
GLuint primidbo; // NEW: Primitive ID buffer
GLuint normalbo; // NEW: Normal buffer
GLuint g_textureID = 0; // Source texture for projection
GLuint g_bakedTextureID = 0; // Baked texture from projection
GLuint g_textureArray = 0; // NEW: Texture array for multi-texture support
GLuint g_fbo = 0; // Framebuffer for projection baking
GLuint g_rbo = 0; // Renderbuffer for depth

struct cudaGraphicsResource* cudaVBO;
struct cudaGraphicsResource* cudaIBO; // Unused but kept for structure compatibility if needed
struct cudaGraphicsResource* cudaCBO;
struct cudaGraphicsResource* cudaUVBO; // NEW: For UV coordinates
struct cudaGraphicsResource* cudaPrimIDBO; // NEW: For primitive IDs
struct cudaGraphicsResource* cudaNormalBO; // NEW: For normals

GLuint uboPrimitives; // UBO Handle
GLuint tboBVH, texBVH; // BVH Texture Buffer

CudaSDFMesh mesh;
bool g_triggerUnwrap = false;
bool g_triggerAtlasPack = false; // NEW: CUDA atlas packing trigger
bool g_triggerProjection = false; // Flag to trigger projection baking
bool g_triggerObjExport = false; // Flag to trigger OBJ export
bool g_AnimateMesh = true; // Flag to control animation/update
bool g_meshDirty = true;   // Force a mesh rebuild even when animation is paused
bool g_rotateView = true; // NEW: Toggle camera/view rotation
double g_cameraAngle = 0.0; // View orbit angle (advances with real dt when view rotation is ON)
double g_simTime = 0.0;    // Simulation time (advances only when animation is ON)
double g_lastFrameTime = 0.0;
bool g_isUnwrapped = false; // Track if mesh has been unwrapped
bool g_isAtlasPacked = false; // NEW: Track if UVs have been repacked into a single atlas (triangle soup)
bool g_hasProjectedTexture = false; // Track if texture has been projection baked
bool g_enableUVGeneration = false; // DISABLE to test basic rendering
bool g_useTextureArray = false; // DISABLE to test basic rendering
bool g_textureArrayValid = false; // NEW: Track if texture array loaded successfully
bool g_useVertexNormals = true; // NEW: Use precomputed vertex normals (better for displacements)
uint32_t g_indexCount = 0; // Number of indices for indexed drawing
MeshExtractionTechnique g_meshTechnique = MeshExtractionTechnique::MarchingCubes;
float g_dcNormalSmoothAngleDeg = 30.0f;
float g_dcQefBlend = 1.0f; // QEF blend: 0 = blocky (cell center), 1 = full QEF solve
bool g_wireframeMode = false; // Wireframe rendering toggle
bool g_triggerGridRebuild = false; // Trigger to rebuild grid with new size

// Camera
Camera g_camera;
double g_lastMouseX = 0.0;
double g_lastMouseY = 0.0;
bool g_mouseInitialized = false;

void checkGLErrors(const char* label) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL Error at " << label << ": " << err << std::endl;
    }
}

struct Vec3 {
    float x, y, z;
};

static inline Vec3 vsub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline float vdot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 vcross(const Vec3& a, const Vec3& b) {
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
static inline Vec3 vnormalize(const Vec3& v) {
    float len2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len2 <= 1e-20f) return {0,0,0};
    float inv = 1.0f / sqrtf(len2);
    return { v.x*inv, v.y*inv, v.z*inv };
}

// Column-major OpenGL view matrix (like gluLookAt)
static inline void BuildLookAt(float out[16], const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = vnormalize(vsub(center, eye));
    Vec3 s = vnormalize(vcross(f, up));
    Vec3 u = vcross(s, f);

    out[0] = s.x;  out[4] = s.y;  out[8]  = s.z;  out[12] = -vdot(s, eye);
    out[1] = u.x;  out[5] = u.y;  out[9]  = u.z;  out[13] = -vdot(u, eye);
    out[2] = -f.x; out[6] = -f.y; out[10] = -f.z; out[14] = vdot(f, eye);
    out[3] = 0.0f; out[7] = 0.0f; out[11] = 0.0f; out[15] = 1.0f;
}

static inline void ExtractCameraPosFromView(const float view[16], float outPos[3]) {
    // For rigid view matrix V = [R t; 0 1], eye = -R^T * t
    const float tx = view[12], ty = view[13], tz = view[14];
    const float r00 = view[0], r01 = view[4], r02 = view[8];
    const float r10 = view[1], r11 = view[5], r12 = view[9];
    const float r20 = view[2], r21 = view[6], r22 = view[10];
    outPos[0] = -(r00*tx + r10*ty + r20*tz);
    outPos[1] = -(r01*tx + r11*ty + r21*tz);
    outPos[2] = -(r02*tx + r12*ty + r22*tz);
}

// Load texture array with multiple textures (all must be same size)
GLuint LoadTextureArray(const std::vector<std::string>& filenames) {
    if (filenames.empty()) {
        std::cerr << "ERROR: No texture files provided!" << std::endl;
        return 0;
    }
    
    // Load first texture to determine size and format
    int refWidth, refHeight, refChannels;
    unsigned char* firstData = stbi_load(filenames[0].c_str(), &refWidth, &refHeight, &refChannels, 0);
    if (!firstData) {
        std::cerr << "ERROR: Failed to load first texture: " << filenames[0] << std::endl;
        return 0;
    }
    
    std::cout << "Reference texture: " << filenames[0] << " (" << refWidth << "x" << refHeight 
              << ", " << refChannels << " channels)" << std::endl;
    
    GLenum internalFormat = (refChannels == 4) ? GL_RGBA8 : GL_RGB8;
    GLenum format = (refChannels == 4) ? GL_RGBA : GL_RGB;
    
    GLuint textureArray;
    glGenTextures(1, &textureArray);
    glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);
    
    int numLayers = (int)filenames.size();
    
    // Allocate storage for all layers
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, internalFormat, refWidth, refHeight, numLayers,
                 0, format, GL_UNSIGNED_BYTE, nullptr);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    // Upload first texture
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 
                  refWidth, refHeight, 1, 
                  format, GL_UNSIGNED_BYTE, firstData);
    stbi_image_free(firstData);
    std::cout << "Loaded layer 0: " << filenames[0] << std::endl;
    
    // Load and verify remaining textures
    bool allMatch = true;
    for (int layer = 1; layer < numLayers; ++layer) {
        int width, height, channels;
        unsigned char* data = stbi_load(filenames[layer].c_str(), &width, &height, &channels, 0);
        
        if (!data) {
            std::cerr << "ERROR: Failed to load texture: " << filenames[layer] << std::endl;
            allMatch = false;
            break;
        }
        
        // Check if size matches reference
        if (width != refWidth || height != refHeight) {
            std::cerr << "ERROR: Texture size mismatch!" << std::endl;
            std::cerr << "  Expected: " << refWidth << "x" << refHeight << std::endl;
            std::cerr << "  Got:      " << width << "x" << height << " (" << filenames[layer] << ")" << std::endl;
            std::cerr << "  All textures in a texture array must be the same size!" << std::endl;
            stbi_image_free(data);
            allMatch = false;
            break;
        }
        
        // Check if channel count matches (important for format compatibility)
        if (channels != refChannels) {
            std::cerr << "WARNING: Texture channel mismatch!" << std::endl;
            std::cerr << "  Expected: " << refChannels << " channels" << std::endl;
            std::cerr << "  Got:      " << channels << " channels (" << filenames[layer] << ")" << std::endl;
            std::cerr << "  Attempting to convert..." << std::endl;
            
            // Reload with forced channel count
            stbi_image_free(data);
            data = stbi_load(filenames[layer].c_str(), &width, &height, &channels, refChannels);
            if (!data) {
                std::cerr << "ERROR: Failed to convert texture channels!" << std::endl;
                allMatch = false;
                break;
            }
        }
        
        // Upload to layer
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, 
                      width, height, 1, 
                      format, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
        std::cout << "Loaded layer " << layer << ": " << filenames[layer] << std::endl;
    }
    
    if (!allMatch) {
        std::cerr << "ERROR: Texture array creation failed due to size mismatch!" << std::endl;
        glDeleteTextures(1, &textureArray);
        return 0;
    }
    
    // Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    std::cout << "Successfully created texture array with " << numLayers << " layers (" 
              << refWidth << "x" << refHeight << ")" << std::endl;
    
    return textureArray;
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

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!g_mouseInitialized) {
        g_lastMouseX = xpos;
        g_lastMouseY = ypos;
        g_mouseInitialized = true;
        return;
    }
    
    double xOffset = xpos - g_lastMouseX;
    double yOffset = ypos - g_lastMouseY;
    g_lastMouseX = xpos;
    g_lastMouseY = ypos;
    
    // Only rotate when Alt is held and in free camera mode
    if (g_camera.IsFreeCameraMode() &&
        (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
         glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS)) {
        g_camera.ProcessMouseMovement((float)xOffset, (float)yOffset);
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        bool freeMode = g_camera.IsFreeCameraMode();

        if (key == GLFW_KEY_Q) {  // Changed from W to Q for wireframe toggle
            g_wireframeMode = !g_wireframeMode;
            glPolygonMode(GL_FRONT_AND_BACK, g_wireframeMode ? GL_LINE : GL_FILL);
        }
        if (key == GLFW_KEY_C) {  // Toggle view rotation (orbit mode only)
            if (!freeMode) {
                g_rotateView = !g_rotateView;
                std::cout << "View rotation: " << (g_rotateView ? "ON" : "OFF") << std::endl;
            }
        }
        if (key == GLFW_KEY_F) {  // Toggle free camera mode
            g_camera.SetFreeCameraMode(!freeMode);
            if (freeMode) {
                g_rotateView = false;  // Disable orbit when entering free camera
                std::cout << "Free camera mode: ON (WASD to move, Alt+Mouse to look, Shift=fast, +/-=speed)" << std::endl;
            } else {
                std::cout << "Free camera mode: OFF (orbit mode)" << std::endl;
            }
        }
        if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {  // + key
            g_camera.AdjustSpeed(0.5f);
        }
        if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {  // - key
            g_camera.AdjustSpeed(-0.5f);
        }
        if (key == GLFW_KEY_U) {
            g_triggerUnwrap = true;
        }
        if (key == GLFW_KEY_A) {
            if(!freeMode)
            {
                g_triggerAtlasPack = true;
            }
        }
        if (key == GLFW_KEY_P) {
            g_triggerProjection = true;
        }
        if (key == GLFW_KEY_T) {  // NEW: Toggle texture array mode
            g_useTextureArray = !g_useTextureArray;
            if (g_useTextureArray && !g_textureArrayValid) {
                std::cout << "Texture array mode: ON (but no textures loaded - will show SDF colors)" << std::endl;
            } else {
                std::cout << "Texture array mode: " << (g_useTextureArray ? "ON" : "OFF") << std::endl;
            }
        }
        if (key == GLFW_KEY_V) {  // NEW: Toggle UV generation
            g_enableUVGeneration = !g_enableUVGeneration;
            std::cout << "UV generation: " << (g_enableUVGeneration ? "ON" : "OFF") << std::endl;
            // UV generation affects mesh output buffers
            g_meshDirty = true;
        }
        if (key == GLFW_KEY_N) {  // NEW: Toggle vertex normals
            g_useVertexNormals = !g_useVertexNormals;
            std::cout << "Vertex normals: " << (g_useVertexNormals ? "ON (accurate)" : "OFF (SDF gradient)") << std::endl;
        }
        if (key == GLFW_KEY_O) {  // NEW: Export mesh to OBJ
            g_triggerObjExport = true;
        }
        if (key == GLFW_KEY_SPACE) {  // NEW: Toggle animation
            g_AnimateMesh = !g_AnimateMesh;
            std::cout << "Animation: " << (g_AnimateMesh ? "ON" : "OFF") << std::endl;
        }
        if (key == GLFW_KEY_M) {  // Mesh extraction technique: Marching Cubes
            g_meshTechnique = MeshExtractionTechnique::MarchingCubes;
            std::cout << "Mesh technique: Marching Cubes" << std::endl;
            g_meshDirty = true;
        }
        if (key == GLFW_KEY_X) {  // Mesh extraction technique: Dual Contouring
            g_meshTechnique = MeshExtractionTechnique::DualContouring;
            std::cout << "Mesh technique: Dual Contouring" << std::endl;
            g_meshDirty = true;
        }
        if (key == GLFW_KEY_LEFT_BRACKET) { // DC normal smoothing angle down
            g_dcNormalSmoothAngleDeg = std::max(0.0f, g_dcNormalSmoothAngleDeg - 5.0f);
            std::cout << "DC normal smooth angle: " << g_dcNormalSmoothAngleDeg << " deg" << std::endl;
            g_meshDirty = true;
        }
        if (key == GLFW_KEY_RIGHT_BRACKET) { // DC normal smoothing angle up
            g_dcNormalSmoothAngleDeg = std::min(89.0f, g_dcNormalSmoothAngleDeg + 5.0f);
            std::cout << "DC normal smooth angle: " << g_dcNormalSmoothAngleDeg << " deg" << std::endl;
            g_meshDirty = true;
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
    glfwSetCursorPosCallback(window, mouse_callback);

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
    
    // Verify the program is valid
    GLint isLinked = 0;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &isLinked);
    if (isLinked == GL_FALSE) {
        std::cerr << "ERROR: Shader program failed to link!" << std::endl;
        exit(-1);
    }
    
    std::cout << "Shader program linked successfully" << std::endl;
    
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
    glGenBuffers(1, &normalbo); // NEW: Buffer for normals

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
    // DON'T enable - shader doesn't use aColor
    // glEnableVertexAttribArray(1);
    
    // NEW: Normal Buffer (float4 normals, location 1 - REPLACING the color buffer slot)
    glBindBuffer(GL_ARRAY_BUFFER, normalbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 4, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);  // Enable for normals
    
    // UVBO (vec2 texture coordinates, location 2)
    glBindBuffer(GL_ARRAY_BUFFER, uvbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);  // Enable for UV generation
    
    // NEW: Primitive ID Buffer (float to avoid compatibility issues)
    glGenBuffers(1, &primidbo);
    glBindBuffer(GL_ARRAY_BUFFER, primidbo);
    glBufferData(GL_ARRAY_BUFFER, maxVerts * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(3);  // Enable for primitive ID
    
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
    
    // NEW: Register UV and Primitive ID buffers with CUDA
    cudaError_t errUVBO = cudaGraphicsGLRegisterBuffer(&cudaUVBO, uvbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (errUVBO != cudaSuccess) {
        std::cerr << "CUDA Error Register UVBO: " << cudaGetErrorString(errUVBO) << std::endl;
        exit(-1);
    }
    
    cudaError_t errPrimIDBO = cudaGraphicsGLRegisterBuffer(&cudaPrimIDBO, primidbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (errPrimIDBO != cudaSuccess) {
        std::cerr << "CUDA Error Register PrimIDBO: " << cudaGetErrorString(errPrimIDBO) << std::endl;
        exit(-1);
    }
    
    // NEW: Register Normal buffer with CUDA
    cudaError_t errNormalBO = cudaGraphicsGLRegisterBuffer(&cudaNormalBO, normalbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (errNormalBO != cudaSuccess) {
        std::cerr << "CUDA Error Register NormalBO: " << cudaGetErrorString(errNormalBO) << std::endl;
        exit(-1);
    }
    
    glBindVertexArray(0);
    
    // NEW: Load Texture Array
    std::vector<std::string> textureFiles = {
        "assets/T_checkerNumbered.PNG",          // Layer 0
        "assets/T_debug_color_01.PNG",           // Layer 1
        "assets/T_debug_uv_01.PNG",              // Layer 2
        "assets/T_debug_orientation_01.PNG",     // Layer 3
        "assets/T_OmniDebugTexture_COL.png"      // Layer 4
    };
    g_textureArray = LoadTextureArray(textureFiles);  // Auto-detect size
    g_textureArrayValid = (g_textureArray != 0);
    
    if (g_textureArrayValid) {
        g_useTextureArray = true;  // Enable texture array mode
        std::cout << "Texture array mode ENABLED" << std::endl;
    } else {
        std::cout << "Texture array mode DISABLED (textures not found)" << std::endl;
        std::cout << "Press 'T' to toggle texture array mode when textures are available" << std::endl;
    }
    
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
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup ImGui style - dark theme with custom accent
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.12f, 0.94f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.35f, 1.00f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.45f, 0.65f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.18f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.26f, 0.32f, 1.00f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.46f, 0.69f, 1.00f, 1.00f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.98f, 0.59f, 1.00f);
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    std::cout << "ImGui initialized successfully" << std::endl;
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
    if (!g_isUnwrapped && !g_isAtlasPacked) {
        std::cout << "Mesh must be unwrapped or atlas-packed before projection baking!" << std::endl;
        return;
    }
    
    std::cout << "Starting projection baking to " << textureSize << "x" << textureSize << " texture..." << std::endl;
    
    // Calculate camera position from view matrix
    float cameraPos[3] = {0.0f, 0.0f, 3.0f};
    ExtractCameraPosFromView(view, cameraPos);
    
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
    if (g_isUnwrapped) {
        glDrawElements(GL_TRIANGLES, (GLsizei)g_indexCount, GL_UNSIGNED_INT, 0);
    } else {
        // Atlas-packed triangle soup path
        glDrawArrays(GL_TRIANGLES, 0, mesh.GetTotalVertexCount());
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    g_hasProjectedTexture = true;
    std::cout << "Projection baking complete!" << std::endl;
    
    checkGLErrors("PerformProjectionBaking");
}

// ImGui Control Panel
void RenderUI() {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // --- Rendering Section ---
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Checkbox("Wireframe", &g_wireframeMode)) {
            glPolygonMode(GL_FRONT_AND_BACK, g_wireframeMode ? GL_LINE : GL_FILL);
        }
        
        if (ImGui::Checkbox("UV Generation", &g_enableUVGeneration)) {
            std::cout << "UV generation: " << (g_enableUVGeneration ? "ON" : "OFF") << std::endl;
            g_meshDirty = true;
        }
        
        bool canUseTextureArray = g_textureArrayValid && !g_isUnwrapped && !g_isAtlasPacked;
        if (!canUseTextureArray) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Checkbox("Texture Array Mode", &g_useTextureArray)) {
            if (g_useTextureArray && !g_textureArrayValid) {
                std::cout << "Texture array mode: ON (but no textures loaded)" << std::endl;
            } else {
                std::cout << "Texture array mode: " << (g_useTextureArray ? "ON" : "OFF") << std::endl;
            }
        }
        if (!canUseTextureArray) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Requires valid texture array and no unwrap/atlas");
            }
        }
        
        if (ImGui::Checkbox("Vertex Normals", &g_useVertexNormals)) {
            std::cout << "Vertex normals: " << (g_useVertexNormals ? "ON (accurate)" : "OFF (SDF gradient)") << std::endl;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("ON = accurate for displacements\nOFF = SDF gradient");
        }
    }
    
    // --- Mesh Extraction Section ---
    if (ImGui::CollapsingHeader("Mesh Extraction", ImGuiTreeNodeFlags_DefaultOpen)) {
        /* Disabled until changing grid size is implemented
        // Grid Size
        ImGui::SetNextItemWidth(120);
        int gridSizeInt = (int)g_gridSize;
        if (ImGui::SliderInt("Grid Size", &gridSizeInt, 8, 128)) {
            g_gridSize = (float)gridSizeInt;
            g_cellSize = 1.0f / g_gridSize;
        }
        ImGui::SameLine();
        if (ImGui::Button("Rebuild")) {
            g_triggerGridRebuild = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rebuild mesh with new grid size");
        }
        
        ImGui::Separator();
        */
        
        bool isMC = (g_meshTechnique == MeshExtractionTechnique::MarchingCubes);
        bool isDC = (g_meshTechnique == MeshExtractionTechnique::DualContouring);
        
        if (ImGui::RadioButton("Marching Cubes", isMC)) {
            g_meshTechnique = MeshExtractionTechnique::MarchingCubes;
            std::cout << "Mesh technique: Marching Cubes" << std::endl;
            g_meshDirty = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Dual Contouring", isDC)) {
            g_meshTechnique = MeshExtractionTechnique::DualContouring;
            std::cout << "Mesh technique: Dual Contouring" << std::endl;
            g_meshDirty = true;
        }
        
        if (isDC) {
            ImGui::SetNextItemWidth(150);
            if (ImGui::SliderFloat("Normal Smooth Angle", &g_dcNormalSmoothAngleDeg, 0.0f, 89.0f, "%.0f deg")) {
                std::cout << "DC normal smooth angle: " << g_dcNormalSmoothAngleDeg << " deg" << std::endl;
                g_meshDirty = true;
            }
            
            ImGui::SetNextItemWidth(150);
            if (ImGui::SliderFloat("QEF Blend", &g_dcQefBlend, 0.0f, 1.0f, "%.2f")) {
                g_meshDirty = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("0 = Blocky (cell center)\n1 = Full QEF solve\nBlends vertex positions");
            }
        }
    }
    
    // --- UV / Export Section ---
    if (ImGui::CollapsingHeader("UV / Export", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool meshLocked = g_isUnwrapped || g_isAtlasPacked;
        
        if (meshLocked) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Unwrap Mesh (U)", ImVec2(-1, 0))) {
            g_triggerUnwrap = true;
        }
        if (meshLocked) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Mesh already unwrapped or atlas-packed");
            }
        }
        
        bool canAtlasPack = g_enableUVGeneration && !meshLocked;
        if (!canAtlasPack) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("CUDA Atlas Pack (A)", ImVec2(-1, 0))) {
            g_triggerAtlasPack = true;
        }
        if (!canAtlasPack) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Requires UV generation enabled");
            }
        }
        
        bool canProject = g_isUnwrapped || g_isAtlasPacked;
        if (!canProject) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Projection Baking (P)", ImVec2(-1, 0))) {
            g_triggerProjection = true;
        }
        if (!canProject) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Mesh must be unwrapped or atlas-packed first");
            }
        }
        
        ImGui::Separator();
        
        if (ImGui::Button("Export to OBJ (O)", ImVec2(-1, 0))) {
            g_triggerObjExport = true;
        }
    }
    
    // --- Animation Section ---
    if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Checkbox("Animate", &g_AnimateMesh)) {
            std::cout << "Animation: " << (g_AnimateMesh ? "ON" : "OFF") << std::endl;
        }
        
        // Status indicators
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Status:");
        
        if (g_isUnwrapped) {
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "  Unwrapped");
        }
        if (g_isAtlasPacked) {
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "  Atlas Packed");
        }
        if (g_hasProjectedTexture) {
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.4f, 1.0f), "  Texture Baked");
        }
        
        ImGui::Text("Vertices: %u", mesh.GetTotalVertexCount());
        if (g_isUnwrapped) {
            ImGui::Text("Indices: %u", g_indexCount);
        }
    }
    
    // --- Camera Section ---
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool freeMode = g_camera.IsFreeCameraMode();
        
        if (ImGui::Checkbox("Free Camera (F)", &freeMode)) {
            g_camera.SetFreeCameraMode(freeMode);
            if (freeMode) {
                g_rotateView = false;
                std::cout << "Free camera mode: ON" << std::endl;
            } else {
                std::cout << "Free camera mode: OFF (orbit mode)" << std::endl;
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("WASD to move, Alt+Mouse to look\nShift for speed, +/- to adjust");
        }
        
        if (!freeMode) {
            if (ImGui::Checkbox("Orbit Rotation (C)", &g_rotateView)) {
                std::cout << "View rotation: " << (g_rotateView ? "ON" : "OFF") << std::endl;
            }
        } else {
            ImGui::BeginDisabled();
            bool dummy = false;
            ImGui::Checkbox("Orbit Rotation (C)", &dummy);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Only available in orbit mode");
            }
        }
        
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Mode: %s", freeMode ? "Free" : "Orbit");
        if (!freeMode) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Orbit: %s", g_rotateView ? "Rotating" : "Stopped");
        }
    }
    
    ImGui::End();
}

void ExportMeshToOBJ() {
    std::cout << "Exporting mesh to OBJ..." << std::endl;
    
    // Generate timestamped filename
    time_t now = time(nullptr);
    struct tm timeinfo;
    localtime_s(&timeinfo, &now);
    char filename[256];
    strftime(filename, sizeof(filename), "mesh_export_%Y%m%d_%H%M%S.obj", &timeinfo);
    
    std::ofstream objFile(filename);
    if (!objFile.is_open()) {
        std::cerr << "ERROR: Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    objFile << "# Exported from CudaSDF" << std::endl;
    objFile << "# Generated: " << filename << std::endl;
    objFile << std::endl;
    
    if (g_isUnwrapped) {
        // Export unwrapped mesh with proper indexing
        std::cout << "Exporting unwrapped mesh with " << g_indexCount << " indices..." << std::endl;
        
        // Read vertex positions from VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        size_t vertexCount = g_indexCount; // Maximum possible vertices
        std::vector<float4> vertices(vertexCount);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * sizeof(float4), vertices.data());
        
        // Read UVs from UVBO
        glBindBuffer(GL_ARRAY_BUFFER, uvbo);
        std::vector<float2> uvs(vertexCount);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * sizeof(float2), uvs.data());
        
        // Read indices from IBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        std::vector<uint32_t> indices(g_indexCount);
        glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, g_indexCount * sizeof(uint32_t), indices.data());
        
        // Find actual unique vertex count
        uint32_t maxIndex = 0;
        for (uint32_t idx : indices) {
            maxIndex = std::max(maxIndex, idx);
        }
        uint32_t actualVertexCount = maxIndex + 1;
        
        std::cout << "Unique vertices: " << actualVertexCount << ", Triangles: " << (g_indexCount / 3) << std::endl;
        
        // Write vertices
        for (uint32_t i = 0; i < actualVertexCount; ++i) {
            objFile << "v " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;
        }
        objFile << std::endl;
        
        // Write UVs
        for (uint32_t i = 0; i < actualVertexCount; ++i) {
            objFile << "vt " << uvs[i].x << " " << uvs[i].y << std::endl;
        }
        objFile << std::endl;
        
        // Write faces (OBJ uses 1-based indexing)
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i] + 1;
            uint32_t i1 = indices[i + 1] + 1;
            uint32_t i2 = indices[i + 2] + 1;
            objFile << "f " << i0 << "/" << i0 << " " 
                    << i1 << "/" << i1 << " " 
                    << i2 << "/" << i2 << std::endl;
        }
        
    } else {
        // Export triangle soup (non-unwrapped mesh)
        uint32_t vertexCount = mesh.GetTotalVertexCount();
        std::cout << "Exporting triangle soup with " << vertexCount << " vertices..." << std::endl;
        
        if (vertexCount == 0) {
            std::cout << "ERROR: No vertices to export!" << std::endl;
            objFile.close();
            return;
        }
        
        // Read vertex positions from VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<float4> vertices(vertexCount);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * sizeof(float4), vertices.data());
        
        // Read UVs from UVBO (if available)
        std::vector<float2> uvs(vertexCount);
        bool hasUVs = g_enableUVGeneration;
        if (hasUVs) {
            glBindBuffer(GL_ARRAY_BUFFER, uvbo);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * sizeof(float2), uvs.data());
        }
        
        // Write vertices
        for (uint32_t i = 0; i < vertexCount; ++i) {
            objFile << "v " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;
        }
        objFile << std::endl;
        
        // Write UVs if available
        if (hasUVs) {
            for (uint32_t i = 0; i < vertexCount; ++i) {
                objFile << "vt " << uvs[i].x << " " << uvs[i].y << std::endl;
            }
            objFile << std::endl;
        }
        
        // Write faces (triangle soup - each vertex is unique)
        for (uint32_t i = 0; i < vertexCount; i += 3) {
            uint32_t i0 = i + 1;
            uint32_t i1 = i + 2;
            uint32_t i2 = i + 3;
            
            if (hasUVs) {
                objFile << "f " << i0 << "/" << i0 << " " 
                        << i1 << "/" << i1 << " " 
                        << i2 << "/" << i2 << std::endl;
            } else {
                objFile << "f " << i0 << " " << i1 << " " << i2 << std::endl;
            }
        }
    }
    
    objFile.close();
    std::cout << "Mesh exported successfully to: " << filename << std::endl;
}



int main() {
    initGL();
    mesh.Initialize(g_cellSize, make_float3(-1.1f, -1.1f, -1.1f), make_float3(1.1f, 1.1f, 1.1f));
    // UV generation is controlled by passing non-null pointers to Update()
    g_enableUVGeneration = true;  // Enable flag
    
    // Initialize camera at orbit position
    g_camera.Initialize(0.0f, 0.0f, 3.0f);
    g_camera.SetSpeed(2.0f);
    
    GLFWwindow* window = glfwGetCurrentContext();

    // Create Scene (Primitives Showcase)
    std::vector<SDFPrimitive> scenePrimitives;

    // 1. Torus (green) - Checker pattern
    SDFPrimitive torus = CreateTorusPrim(
        make_float3(-0.5f, 0.5f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(0.2f, 0.8f, 0.2f),
        SDF_UNION,
        0.0f,
        0.4f,
        0.15f
    );
    torus.uvScale = make_float2(4.0f, 2.0f);  // Tile 4x horizontally, 2x vertically
    torus.textureID = 0;  // Checker numbered texture
    scenePrimitives.push_back(torus);

    // 2. Hex Prism (Red)(Twisted) - Color debug texture
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
    hex.uvScale = make_float2(2.0f, 1.0f);  // 2x horizontal wrap
    hex.textureID = 1;  // Color debug 01
    scenePrimitives.push_back(hex);

    // 3. Rounded Cone (Blue) - UV debug texture
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
    cone.uvScale = make_float2(1.0f, 1.0f);  // Standard mapping
    cone.textureID = 2;  // UV debug 01
    scenePrimitives.push_back(cone);

    // 4. Rounded Cylinder (Yellow) - Orientation debug
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
    cyl.uvScale = make_float2(3.0f, 1.5f);  // 3x wrap, 1.5x vertical
    cyl.textureID = 3;  // Orientation debug
    scenePrimitives.push_back(cyl);

    // 5. Sphere (Purple, Subtracted) - Omni debug texture
    SDFPrimitive sphere = CreateSpherePrim(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float3(0.5f, 0.0f, 0.5f),
        SDF_SUBTRACT,
        0.0f,
        0.6f
    );
    sphere.uvScale = make_float2(2.0f, 2.0f);  // 2x tiling
    sphere.textureID = 4;  // Omni debug texture
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
    float baseView[16];
    memcpy(baseView, view, sizeof(view));

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

    // Print startup info
    std::cout << "\n=== CudaSDF Started ===" << std::endl;
    std::cout << "Controls available in the ImGui panel on the left." << std::endl;
    std::cout << "Camera: F=free mode, WASD=move, Alt+Mouse=look, C=orbit toggle" << std::endl;
    std::cout << "ESC to exit" << std::endl;
    std::cout << "========================\n" << std::endl;

    double lastFPSTime = glfwGetTime();
    int frameCount = 0;
    g_lastFrameTime = lastFPSTime;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastFPSTime >= 1.0) {
            std::string title = "CUDA Marching Cubes (Sparse) - " + std::to_string(frameCount) + " FPS";
            glfwSetWindowTitle(window, title.c_str());
            frameCount = 0;
            lastFPSTime = currentTime;
        }

        // Advance simulation time only when animation is enabled.
        double dt = currentTime - g_lastFrameTime;
        g_lastFrameTime = currentTime;
        if (dt < 0.0) dt = 0.0;
        if (dt > 0.25) dt = 0.25; // clamp huge hitches (breakpoints, window drag, etc.)
        if (g_AnimateMesh) {
            g_simTime += dt;
        }
        float simTime = (float)g_simTime;
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Camera update
        if (g_camera.IsFreeCameraMode()) {
            // Free camera mode - process WASD movement
            g_camera.ProcessKeyboard(window, (float)dt);
            g_camera.GetViewMatrix(view);
        } else {
            // Orbit mode - rotate camera around origin
            if (g_rotateView) {
                g_cameraAngle -= dt * 1.5;
                // keep angle bounded to avoid precision issues over long runs
                if (g_cameraAngle > 100000.0) g_cameraAngle = fmod(g_cameraAngle, 6.283185307179586);
            }
            float angle = (float)g_cameraAngle;
            Vec3 eye = {sinf(angle) * 3.0f, 0.0f, cosf(angle) * 3.0f};
            BuildLookAt(view, eye, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        }

        // Update mesh if animating OR if something toggled that requires a rebuild.
        // NOTEF: If the mesh has been unwrapped or atlas-packed, we treat it as "locked" and do not overwrite buffers.
        const bool meshLocked = (g_isUnwrapped || g_isAtlasPacked);
        const bool shouldUpdateMesh = (!meshLocked) && (g_AnimateMesh || g_meshDirty);
        if (shouldUpdateMesh) {
            if (g_AnimateMesh) {
                // Animate primitives only when animation is enabled
                scenePrimitives[0].dispParams[1] = 5.0f * sinf(simTime*0.1f);
                float c_rot = cosf(simTime), s_rot = sinf(simTime);
                scenePrimitives[1].rotation = make_float4(0.0f, s_rot, 0.0f, c_rot);
                scenePrimitives[2].position.y = -0.5f + 0.2f * sinf(simTime * 2.0f);
                scenePrimitives[3].params[2] = 0.1f + 0.05f * sinf(simTime * 3.0f);
            }

            // Update Mesh
            mesh.ClearPrimitives();
            for(const auto& p : scenePrimitives) {
                mesh.AddPrimitive(p);
            }
            
            // Map GL buffers
            if (cudaGraphicsMapResources(1, &cudaVBO, 0) != cudaSuccess) break;
            if (cudaGraphicsMapResources(1, &cudaCBO, 0) != cudaSuccess) break;
            
            // NEW: Map Normal buffer (always enabled for better lighting)
            cudaError_t normalErr = cudaGraphicsMapResources(1, &cudaNormalBO, 0);
            if (normalErr != cudaSuccess) {
                std::cerr << "ERROR mapping NormalBO: " << cudaGetErrorString(normalErr) << std::endl;
                break;
            }
            
            // NEW: Map UV and Primitive ID buffers if UV generation is enabled
            if (g_enableUVGeneration) {
                cudaError_t uvErr = cudaGraphicsMapResources(1, &cudaUVBO, 0);
                cudaError_t primErr = cudaGraphicsMapResources(1, &cudaPrimIDBO, 0);
                
                if (uvErr != cudaSuccess) {
                    std::cerr << "ERROR mapping UVBO: " << cudaGetErrorString(uvErr) << std::endl;
                    break;
                }
                if (primErr != cudaSuccess) {
                    std::cerr << "ERROR mapping PrimIDBO: " << cudaGetErrorString(primErr) << std::endl;
                    break;
                }
            }
            
            float4* d_vboPtr;
            float4* d_cboPtr;
            float4* d_normalPtr = nullptr;
            unsigned int* d_iboPtr = nullptr;
            float2* d_uvPtr = nullptr;
            int* d_primIDPtr = nullptr;
            size_t size;
            
            cudaGraphicsResourceGetMappedPointer((void**)&d_vboPtr, &size, cudaVBO);
            cudaGraphicsResourceGetMappedPointer((void**)&d_cboPtr, &size, cudaCBO);
            cudaGraphicsResourceGetMappedPointer((void**)&d_normalPtr, &size, cudaNormalBO);
            
            // NEW: Get UV and Primitive ID pointers
            if (g_enableUVGeneration) {
                cudaError_t uvPtrErr = cudaGraphicsResourceGetMappedPointer((void**)&d_uvPtr, &size, cudaUVBO);
                cudaError_t primPtrErr = cudaGraphicsResourceGetMappedPointer((void**)&d_primIDPtr, &size, cudaPrimIDBO);
                
                if (uvPtrErr != cudaSuccess) {
                    std::cerr << "ERROR getting UV pointer: " << cudaGetErrorString(uvPtrErr) << std::endl;
                }
                if (primPtrErr != cudaSuccess) {
                    std::cerr << "ERROR getting PrimID pointer: " << cudaGetErrorString(primPtrErr) << std::endl;
                }
            }
            
            // Update mesh with UV generation and normals
            mesh.Update(simTime, d_vboPtr, d_cboPtr, d_iboPtr, d_uvPtr, d_primIDPtr, d_normalPtr, g_meshTechnique, g_dcNormalSmoothAngleDeg, g_dcQefBlend);
            
            cudaGraphicsUnmapResources(1, &cudaVBO, 0);
            cudaGraphicsUnmapResources(1, &cudaCBO, 0);
            cudaGraphicsUnmapResources(1, &cudaNormalBO, 0);
            // NEW: Unmap UV and Primitive ID buffers
            if (g_enableUVGeneration) {
                cudaGraphicsUnmapResources(1, &cudaUVBO, 0);
                cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);
            }
            // cudaGraphicsUnmapResources(1, &cudaIBO, 0);

            g_meshDirty = false;
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
                g_meshDirty = false;
            } else {
                std::cout << "Mesh is empty, cannot unwrap." << std::endl;
                cudaGraphicsUnmapResources(1, &cudaVBO, 0);
            }
        }
        // -------------------------

        // --- CUDA Atlas Pack Trigger (primitive UVs -> single atlas) ---
        if (g_triggerAtlasPack) {
            g_triggerAtlasPack = false;

            if (!g_enableUVGeneration) {
                std::cout << "UV generation is OFF. Press 'V' to enable UV generation before packing." << std::endl;
            } else {
                unsigned int count = mesh.GetTotalVertexCount();
                if (count == 0) {
                    std::cout << "Mesh is empty, cannot pack." << std::endl;
                } else {
                    // Map current buffers to run packer directly on CUDA/GL interop memory
                    if (cudaGraphicsMapResources(1, &cudaVBO, 0) != cudaSuccess) break;
                    if (cudaGraphicsMapResources(1, &cudaUVBO, 0) != cudaSuccess) break;
                    if (cudaGraphicsMapResources(1, &cudaPrimIDBO, 0) != cudaSuccess) break;

                    float4* d_vboPtr = nullptr;
                    float2* d_uvPtr = nullptr;
                    int* d_primIDPtr = nullptr;
                    size_t size = 0;

                    cudaGraphicsResourceGetMappedPointer((void**)&d_vboPtr, &size, cudaVBO);
                    cudaGraphicsResourceGetMappedPointer((void**)&d_uvPtr, &size, cudaUVBO);
                    cudaGraphicsResourceGetMappedPointer((void**)&d_primIDPtr, &size, cudaPrimIDBO);

                    std::cout << "Running CUDA atlas packer on " << count << " vertices..." << std::endl;

                    GridUVPacker::PackingConfig cfg;
                    cfg.gridSize = 2048;         // board size in cells (v1)
                    cfg.gridResolution = 64;     // quantization: cells per UV unit (v1 simplification)
                    cfg.marginCells = 4;         // gutter
                    cfg.enableRotation = true;   // 0° / 90°
                    cfg.maxIterations = 256;     // number of attempts (capped internally)
                    cfg.xAlignment = 8;          // aligned-X simplification
                    cfg.maxCandidatesPerIsland = 16; // perf/quality balance (was 32)
                    cfg.maxDropSteps = 256;          // cap downward probing to avoid worst-case stalls
                    cfg.shuffleOrderPerAttempt = true;
                    cfg.randomSeed = 1337;

                    auto charts = GridUVPacker::ExtractCharts(d_vboPtr, d_uvPtr, d_primIDPtr, (int)count);
                    auto islands = GridUVPacker::CreateIslands(charts, d_uvPtr, cfg.gridResolution, cfg.marginCells);
                    auto atlas = GridUVPacker::PackIslands(islands, cfg);
                    if (atlas.success) {
                        GridUVPacker::RemapUVsToAtlas(d_uvPtr, d_primIDPtr, (int)count, charts, islands, atlas, cfg.gridResolution);
                        g_isAtlasPacked = true;
                        g_useTextureArray = false; // atlas mode uses a single texture
                        g_AnimateMesh = false;     // freeze mesh so packed UVs remain stable for projection tests
                        g_meshDirty = false;
                        std::cout << "Atlas packing complete. UVs remapped into [0,1] atlas." << std::endl;
                    } else {
                        std::cout << "Atlas packing failed (partial/empty result). Try increasing gridSize or reducing margin/attempts." << std::endl;
                    }
                    GridUVPacker::FreeIslands(islands);

                    cudaGraphicsUnmapResources(1, &cudaVBO, 0);
                    cudaGraphicsUnmapResources(1, &cudaUVBO, 0);
                    cudaGraphicsUnmapResources(1, &cudaPrimIDBO, 0);
                }
            }
        }
        // ------------------------------------------------------------
        
        // --- Projection Baking Trigger ---
        if (g_triggerProjection) {
            g_triggerProjection = false;
            PerformProjectionBaking(model, view, projection);
            // Restore viewport after baking
            glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        // ---------------------------------
        
        // --- OBJ Export Trigger ---
        if (g_triggerObjExport) {
            g_triggerObjExport = false;
            ExportMeshToOBJ();
        }
        // --------------------------
        
        // --- Grid Rebuild Trigger ---
        if (g_triggerGridRebuild) {
            g_triggerGridRebuild = false;
            std::cout << "Rebuilding grid with size " << g_gridSize << "..." << std::endl;
            
            // Reset mesh state flags
            g_isUnwrapped = false;
            g_isAtlasPacked = false;
            g_hasProjectedTexture = false;
            
            // Reinitialize mesh with new cell size
            mesh.ClearPrimitives();
            mesh.Initialize(g_cellSize, make_float3(-1.1f, -1.1f, -1.1f), make_float3(1.1f, 1.1f, 1.1f));
            
            // Re-add primitives (they will be re-added in the update loop)
            g_meshDirty = true;
            
            std::cout << "Grid rebuilt. Cell size: " << g_cellSize << std::endl;
        }
        // ----------------------------

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
        
        // Validate the program with current GL state
        glValidateProgram(shaderProgram);
        GLint isValid = 0;
        glGetProgramiv(shaderProgram, GL_VALIDATE_STATUS, &isValid);
        if (isValid == GL_FALSE) {
            GLchar infoLog[1024];
            glGetProgramInfoLog(shaderProgram, 1024, NULL, infoLog);
            std::cerr << "ERROR: Shader program validation failed: " << infoLog << std::endl;
            std::cerr << "Press any key to continue..." << std::endl;
            std::cin.get();
        }
        
        // Set time
        glUniform1f(glGetUniformLocation(shaderProgram, "time"), simTime);
        glUniform1i(glGetUniformLocation(shaderProgram, "uNumBVHPrimitives"), mesh.GetNumBVHPrimitives());
        glUniform1i(glGetUniformLocation(shaderProgram, "uNumTotalPrimitives"), (int)currentGPUPrimitives.size());
        glUniform1i(glGetUniformLocation(shaderProgram, "useVertexNormals"), g_useVertexNormals ? 1 : 0);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, texBVH);
        glUniform1i(glGetUniformLocation(shaderProgram, "bvhNodes"), 0);
    
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, model);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, view);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, projection);
        
        // Texture array binding for Direct Mode (only if valid and not in atlas/unwrapped mode)
        if (g_useTextureArray && g_textureArrayValid && g_enableUVGeneration && !g_isUnwrapped && !g_isAtlasPacked) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D_ARRAY, g_textureArray);
            glUniform1i(glGetUniformLocation(shaderProgram, "textureArray"), 1);
            
            // Bind dummy texture to atlasTexture sampler (required even if not used)
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, g_textureID);
            glUniform1i(glGetUniformLocation(shaderProgram, "atlasTexture"), 2);
            
            glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);  // Direct mode
        } else {
            // Always bind texture samplers to avoid GL_INVALID_OPERATION
            glActiveTexture(GL_TEXTURE1);
            if (g_hasProjectedTexture) {
                glBindTexture(GL_TEXTURE_2D, g_bakedTextureID);
            } else {
                glBindTexture(GL_TEXTURE_2D, g_textureID); // Bind source texture
            }
            glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 1);
            
            // Bind dummy texture to texture array sampler (required even if not used)
            glActiveTexture(GL_TEXTURE2);
            // IMPORTANT: atlas mode samples `atlasTexture`, so bind the same 2D texture here too.
            // (Projection baking writes to g_bakedTextureID; we want to display that when available.)
            glBindTexture(GL_TEXTURE_2D, g_hasProjectedTexture ? g_bakedTextureID : g_textureID);
            glUniform1i(glGetUniformLocation(shaderProgram, "atlasTexture"), 2);
        }

        glBindVertexArray(vao);
        
        if (g_isUnwrapped) {
            // After unwrap, still use vertex colors unless projection baked
            if (g_hasProjectedTexture) {
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 2);  // Single texture mode
            } else {
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);  // SDF color
            }
            glDrawElements(GL_TRIANGLES, (GLsizei)g_indexCount, GL_UNSIGNED_INT, 0);
        } else {
            // Triangle soup mode - use texture array if enabled and valid
            if (g_isAtlasPacked) {
                // Atlas-packed UVs (single atlas texture)
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), g_hasProjectedTexture ? 2 : 0);
            } else if (!g_useTextureArray || !g_textureArrayValid || !g_enableUVGeneration) {
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);  // SDF color
            }
            glDrawArrays(GL_TRIANGLES, 0, mesh.GetTotalVertexCount());
        }
        
        // Render ImGui
        RenderUI();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // Cleanup GLFW
    glfwTerminate();
    return 0;
}

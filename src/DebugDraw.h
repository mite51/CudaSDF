#pragma once

#include <glad/glad.h>
#include <vector>
#include <cstdint>

// Debug line drawing system for visualization
// Lines are rendered as screen-facing quads with variable width

struct DebugVertex {
    float x, y, z;      // Position
    float r, g, b, a;   // Color
    float u, v;         // Texture coordinates
    float width;        // Line width at this vertex
};

struct DebugLine {
    DebugVertex start;
    DebugVertex end;
};

class DebugDraw {
public:
    DebugDraw();
    ~DebugDraw();

    // Initialize OpenGL resources (call after GL context is ready)
    bool Initialize();

    // Shutdown and cleanup
    void Shutdown();

    // Add a line to be drawn this frame
    // width is in world units, color is RGBA [0-1]
    void AddLine(
        float x0, float y0, float z0,
        float x1, float y1, float z1,
        float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f,
        float widthStart = 0.01f, float widthEnd = 0.01f
    );

    // Add a line with full vertex control (color, UV, width per endpoint)
    void AddLine(const DebugVertex& start, const DebugVertex& end);

    // Add a box (12 edges)
    void AddBox(
        float minX, float minY, float minZ,
        float maxX, float maxY, float maxZ,
        float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f,
        float width = 0.01f
    );

    // Add a sphere approximation (wireframe)
    void AddWireSphere(
        float cx, float cy, float cz, float radius,
        float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f,
        float width = 0.01f,
        int segments = 16
    );

    // Add a coordinate axes visualization
    void AddAxes(
        float cx, float cy, float cz, float size,
        float width = 0.01f
    );

    // Render all queued lines, then clear the queue
    // Pass view/projection matrices (column-major, OpenGL convention)
    void Render(const float* viewMatrix, const float* projectionMatrix);

    // Clear all queued lines without rendering
    void Clear();

    // Set optional texture (0 = no texture, uses vertex colors only)
    void SetTexture(GLuint textureID);

    // Get number of lines queued
    size_t GetLineCount() const { return m_lines.size(); }

    // Enable/disable depth testing for debug lines
    void SetDepthTestEnabled(bool enabled) { m_depthTestEnabled = enabled; }
    bool IsDepthTestEnabled() const { return m_depthTestEnabled; }

private:
    void CreateShaders();
    void CreateBuffers();
    void UploadLineData();

    std::vector<DebugLine> m_lines;

    // OpenGL resources
    GLuint m_shaderProgram = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;

    // Uniform locations
    GLint m_locView = -1;
    GLint m_locProjection = -1;
    GLint m_locUseTexture = -1;
    GLint m_locTexture = -1;

    // Optional texture
    GLuint m_textureID = 0;

    // Settings
    bool m_depthTestEnabled = true;
    bool m_initialized = false;

    // Buffer capacity (grows as needed)
    size_t m_bufferCapacity = 0;
};

// Global debug draw instance (optional convenience)
extern DebugDraw* g_debugDraw;

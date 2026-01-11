#include "DebugDraw.h"
#include <iostream>
#include <cmath>

// Global instance (optional)
DebugDraw* g_debugDraw = nullptr;

// Vertex shader - passes line data to geometry shader
static const char* debugVertexShader = R"(
    #version 330 core
    
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec4 aColor;
    layout (location = 2) in vec2 aTexCoord;
    layout (location = 3) in float aWidth;
    
    out VS_OUT {
        vec4 color;
        vec2 texCoord;
        float width;
    } vs_out;
    
    void main() {
        gl_Position = vec4(aPos, 1.0);
        vs_out.color = aColor;
        vs_out.texCoord = aTexCoord;
        vs_out.width = aWidth;
    }
)";

// Geometry shader - expands lines into screen-facing quads
static const char* debugGeometryShader = R"(
    #version 330 core
    
    layout (lines) in;
    layout (triangle_strip, max_vertices = 4) out;
    
    in VS_OUT {
        vec4 color;
        vec2 texCoord;
        float width;
    } gs_in[];
    
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec4 fragColor;
    out vec2 fragTexCoord;
    
    void main() {
        // Transform line endpoints to clip space
        vec4 p0_world = gl_in[0].gl_Position;
        vec4 p1_world = gl_in[1].gl_Position;
        
        vec4 p0_view = view * p0_world;
        vec4 p1_view = view * p1_world;
        
        vec4 p0_clip = projection * p0_view;
        vec4 p1_clip = projection * p1_view;
        
        // Calculate line direction in view space
        vec3 lineDir = normalize(p1_view.xyz - p0_view.xyz);
        
        // Get camera forward (view space Z is the camera direction)
        vec3 camForward = vec3(0.0, 0.0, -1.0);
        
        // Calculate perpendicular direction (screen-space right)
        vec3 perpDir = normalize(cross(lineDir, camForward));
        
        // If line is parallel to view, use up vector instead
        if (length(perpDir) < 0.001) {
            perpDir = vec3(0.0, 1.0, 0.0);
        }
        
        // Scale by width (in world units, but we keep it relative)
        vec3 offset0 = perpDir * gs_in[0].width * 0.5;
        vec3 offset1 = perpDir * gs_in[1].width * 0.5;
        
        // Generate 4 vertices for the quad
        
        // Vertex 0: start - offset
        vec4 v0_view = vec4(p0_view.xyz - offset0, 1.0);
        gl_Position = projection * v0_view;
        fragColor = gs_in[0].color;
        fragTexCoord = vec2(0.0, 0.0);
        EmitVertex();
        
        // Vertex 1: start + offset
        vec4 v1_view = vec4(p0_view.xyz + offset0, 1.0);
        gl_Position = projection * v1_view;
        fragColor = gs_in[0].color;
        fragTexCoord = vec2(0.0, 1.0);
        EmitVertex();
        
        // Vertex 2: end - offset
        vec4 v2_view = vec4(p1_view.xyz - offset1, 1.0);
        gl_Position = projection * v2_view;
        fragColor = gs_in[1].color;
        fragTexCoord = vec2(1.0, 0.0);
        EmitVertex();
        
        // Vertex 3: end + offset
        vec4 v3_view = vec4(p1_view.xyz + offset1, 1.0);
        gl_Position = projection * v3_view;
        fragColor = gs_in[1].color;
        fragTexCoord = vec2(1.0, 1.0);
        EmitVertex();
        
        EndPrimitive();
    }
)";

// Fragment shader
static const char* debugFragmentShader = R"(
    #version 330 core
    
    in vec4 fragColor;
    in vec2 fragTexCoord;
    
    uniform int useTexture;
    uniform sampler2D debugTexture;
    
    out vec4 FragColor;
    
    void main() {
        vec4 color = fragColor;
        
        if (useTexture == 1) {
            vec4 texColor = texture(debugTexture, fragTexCoord);
            color *= texColor;
        }
        
        // Discard fully transparent pixels
        if (color.a < 0.01) {
            discard;
        }
        
        FragColor = color;
    }
)";

DebugDraw::DebugDraw() {
}

DebugDraw::~DebugDraw() {
    Shutdown();
}

bool DebugDraw::Initialize() {
    if (m_initialized) {
        return true;
    }

    CreateShaders();
    CreateBuffers();

    m_initialized = true;
    std::cout << "DebugDraw initialized successfully" << std::endl;
    return true;
}

void DebugDraw::Shutdown() {
    if (!m_initialized) {
        return;
    }

    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    m_initialized = false;
    m_lines.clear();
}

void DebugDraw::CreateShaders() {
    // Compile vertex shader
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &debugVertexShader, nullptr);
    glCompileShader(vs);

    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(vs, 1024, nullptr, infoLog);
        std::cerr << "DebugDraw vertex shader compilation error:\n" << infoLog << std::endl;
    }

    // Compile geometry shader
    GLuint gs = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(gs, 1, &debugGeometryShader, nullptr);
    glCompileShader(gs);

    glGetShaderiv(gs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(gs, 1024, nullptr, infoLog);
        std::cerr << "DebugDraw geometry shader compilation error:\n" << infoLog << std::endl;
    }

    // Compile fragment shader
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &debugFragmentShader, nullptr);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(fs, 1024, nullptr, infoLog);
        std::cerr << "DebugDraw fragment shader compilation error:\n" << infoLog << std::endl;
    }

    // Link program
    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vs);
    glAttachShader(m_shaderProgram, gs);
    glAttachShader(m_shaderProgram, fs);
    glLinkProgram(m_shaderProgram);

    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(m_shaderProgram, 1024, nullptr, infoLog);
        std::cerr << "DebugDraw shader program link error:\n" << infoLog << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(gs);
    glDeleteShader(fs);

    // Cache uniform locations
    m_locView = glGetUniformLocation(m_shaderProgram, "view");
    m_locProjection = glGetUniformLocation(m_shaderProgram, "projection");
    m_locUseTexture = glGetUniformLocation(m_shaderProgram, "useTexture");
    m_locTexture = glGetUniformLocation(m_shaderProgram, "debugTexture");
}

void DebugDraw::CreateBuffers() {
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    // Initial buffer allocation
    m_bufferCapacity = 1000;
    glBufferData(GL_ARRAY_BUFFER, m_bufferCapacity * 2 * sizeof(DebugVertex), nullptr, GL_DYNAMIC_DRAW);

    // Position (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), (void*)offsetof(DebugVertex, x));
    glEnableVertexAttribArray(0);

    // Color (4 floats)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), (void*)offsetof(DebugVertex, r));
    glEnableVertexAttribArray(1);

    // TexCoord (2 floats)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), (void*)offsetof(DebugVertex, u));
    glEnableVertexAttribArray(2);

    // Width (1 float)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), (void*)offsetof(DebugVertex, width));
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
}

void DebugDraw::AddLine(
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    float r, float g, float b, float a,
    float widthStart, float widthEnd
) {
    DebugVertex start = { x0, y0, z0, r, g, b, a, 0.0f, 0.0f, widthStart };
    DebugVertex end = { x1, y1, z1, r, g, b, a, 1.0f, 0.0f, widthEnd };
    m_lines.push_back({ start, end });
}

void DebugDraw::AddLine(const DebugVertex& start, const DebugVertex& end) {
    m_lines.push_back({ start, end });
}

void DebugDraw::AddBox(
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ,
    float r, float g, float b, float a,
    float width
) {
    // Bottom face
    AddLine(minX, minY, minZ, maxX, minY, minZ, r, g, b, a, width, width);
    AddLine(maxX, minY, minZ, maxX, minY, maxZ, r, g, b, a, width, width);
    AddLine(maxX, minY, maxZ, minX, minY, maxZ, r, g, b, a, width, width);
    AddLine(minX, minY, maxZ, minX, minY, minZ, r, g, b, a, width, width);

    // Top face
    AddLine(minX, maxY, minZ, maxX, maxY, minZ, r, g, b, a, width, width);
    AddLine(maxX, maxY, minZ, maxX, maxY, maxZ, r, g, b, a, width, width);
    AddLine(maxX, maxY, maxZ, minX, maxY, maxZ, r, g, b, a, width, width);
    AddLine(minX, maxY, maxZ, minX, maxY, minZ, r, g, b, a, width, width);

    // Vertical edges
    AddLine(minX, minY, minZ, minX, maxY, minZ, r, g, b, a, width, width);
    AddLine(maxX, minY, minZ, maxX, maxY, minZ, r, g, b, a, width, width);
    AddLine(maxX, minY, maxZ, maxX, maxY, maxZ, r, g, b, a, width, width);
    AddLine(minX, minY, maxZ, minX, maxY, maxZ, r, g, b, a, width, width);
}

void DebugDraw::AddWireSphere(
    float cx, float cy, float cz, float radius,
    float r, float g, float b, float a,
    float width,
    int segments
) {
    const float PI = 3.14159265359f;
    
    // Draw 3 circles (XY, XZ, YZ planes)
    for (int i = 0; i < segments; ++i) {
        float angle0 = (float)i / segments * 2.0f * PI;
        float angle1 = (float)(i + 1) / segments * 2.0f * PI;

        float c0 = cosf(angle0);
        float s0 = sinf(angle0);
        float c1 = cosf(angle1);
        float s1 = sinf(angle1);

        // XY circle
        AddLine(cx + radius * c0, cy + radius * s0, cz,
                cx + radius * c1, cy + radius * s1, cz,
                r, g, b, a, width, width);

        // XZ circle
        AddLine(cx + radius * c0, cy, cz + radius * s0,
                cx + radius * c1, cy, cz + radius * s1,
                r, g, b, a, width, width);

        // YZ circle
        AddLine(cx, cy + radius * c0, cz + radius * s0,
                cx, cy + radius * c1, cz + radius * s1,
                r, g, b, a, width, width);
    }
}

void DebugDraw::AddAxes(
    float cx, float cy, float cz, float size,
    float width
) {
    // X axis - Red
    AddLine(cx, cy, cz, cx + size, cy, cz, 1.0f, 0.0f, 0.0f, 1.0f, width, width);
    // Y axis - Green
    AddLine(cx, cy, cz, cx, cy + size, cz, 0.0f, 1.0f, 0.0f, 1.0f, width, width);
    // Z axis - Blue
    AddLine(cx, cy, cz, cx, cy, cz + size, 0.0f, 0.0f, 1.0f, 1.0f, width, width);
}

void DebugDraw::UploadLineData() {
    if (m_lines.empty()) {
        return;
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    // Resize buffer if needed
    if (m_lines.size() > m_bufferCapacity) {
        m_bufferCapacity = m_lines.size() * 2;
        glBufferData(GL_ARRAY_BUFFER, m_bufferCapacity * 2 * sizeof(DebugVertex), nullptr, GL_DYNAMIC_DRAW);
    }

    // Upload line data (each line has 2 vertices)
    std::vector<DebugVertex> vertices;
    vertices.reserve(m_lines.size() * 2);
    for (const auto& line : m_lines) {
        vertices.push_back(line.start);
        vertices.push_back(line.end);
    }

    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(DebugVertex), vertices.data());
}

void DebugDraw::Render(const float* viewMatrix, const float* projectionMatrix) {
    if (!m_initialized || m_lines.empty()) {
        m_lines.clear();
        return;
    }

    UploadLineData();

    // Save GL state
    GLboolean depthTestWasEnabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blendWasEnabled = glIsEnabled(GL_BLEND);
    GLint prevBlendSrcRGB, prevBlendDstRGB, prevBlendSrcAlpha, prevBlendDstAlpha;
    glGetIntegerv(GL_BLEND_SRC_RGB, &prevBlendSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB, &prevBlendDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &prevBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &prevBlendDstAlpha);

    // Setup state
    if (m_depthTestEnabled) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind shader and set uniforms
    glUseProgram(m_shaderProgram);
    glUniformMatrix4fv(m_locView, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(m_locProjection, 1, GL_FALSE, projectionMatrix);

    // Texture setup
    if (m_textureID != 0) {
        glActiveTexture(GL_TEXTURE3);  // Use texture unit 3 to avoid conflicts
        glBindTexture(GL_TEXTURE_2D, m_textureID);
        glUniform1i(m_locTexture, 3);
        glUniform1i(m_locUseTexture, 1);
    } else {
        glUniform1i(m_locUseTexture, 0);
    }

    // Draw
    glBindVertexArray(m_vao);
    glDrawArrays(GL_LINES, 0, (GLsizei)(m_lines.size() * 2));
    glBindVertexArray(0);

    // Restore GL state
    if (depthTestWasEnabled) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    if (blendWasEnabled) {
        glEnable(GL_BLEND);
    } else {
        glDisable(GL_BLEND);
    }
    glBlendFuncSeparate(prevBlendSrcRGB, prevBlendDstRGB, prevBlendSrcAlpha, prevBlendDstAlpha);

    // Clear lines after rendering
    m_lines.clear();
}

void DebugDraw::Clear() {
    m_lines.clear();
}

void DebugDraw::SetTexture(GLuint textureID) {
    m_textureID = textureID;
}

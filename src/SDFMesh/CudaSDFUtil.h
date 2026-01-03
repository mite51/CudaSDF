#pragma once
#include "Commons.cuh"
#include <vector>

// --- Primitive Creation Helpers ---

// Helper to initialize with default 0 and padding
inline SDFPrimitive CreatePrimitive(int type, float3 position, float4 rotation, float3 color, int operation, float blendFactor) {
    SDFPrimitive prim;
    prim.rotation = rotation;
    prim.dispParams[0] = 0.0f; prim.dispParams[1] = 0.0f; prim.dispParams[2] = 0.0f; prim.dispParams[3] = 0.0f;
    
    prim.position = position;
    prim.blendFactor = blendFactor;
    
    prim.scale = make_float3(1.0f, 1.0f, 1.0f);
    prim.rounding = 0.0f;
    
    prim.color = color;
    prim.annular = 0.0f;
    
    for(int i=0; i<6; i++) prim.params[i] = 0.0f;
    
    prim.type = type;
    prim.operation = operation;
    prim.displacement = DISP_NONE;
    prim._pad1 = 0;
    
    // Initialize UV parameters (NEW)
    prim.uvScale = make_float2(1.0f, 1.0f);
    prim.uvOffset = make_float2(0.0f, 0.0f);
    prim.uvRotation = 0.0f;
    prim.uvMode = UV_PRIMITIVE;
    prim.textureID = 0;
    prim.atlasIslandID = -1;
    
    return prim;
}

inline SDFPrimitive CreateSpherePrim(float3 position, float4 rotation, float3 color, int operation, float blendFactor, float radius) {
    SDFPrimitive prim = CreatePrimitive(SDF_SPHERE, position, rotation, color, operation, blendFactor);
    prim.params[0] = radius;
    return prim;
}

inline SDFPrimitive CreateTorusPrim(float3 position, float4 rotation, float3 color, int operation, float blendFactor, float majorRadius, float minorRadius) {
    SDFPrimitive prim = CreatePrimitive(SDF_TORUS, position, rotation, color, operation, blendFactor);
    prim.params[0] = majorRadius;
    prim.params[1] = minorRadius;
    return prim;
}

inline SDFPrimitive CreateHexPrismPrim(float3 position, float4 rotation, float3 color, int operation, float blendFactor, float radius, float height) {
    SDFPrimitive prim = CreatePrimitive(SDF_HEX_PRISM, position, rotation, color, operation, blendFactor);
    prim.params[0] = radius;
    prim.params[1] = height;
    return prim;
}

inline SDFPrimitive CreateRoundedConePrim(float3 position, float4 rotation, float3 color, int operation, float blendFactor, float height, float r1, float r2) {
    SDFPrimitive prim = CreatePrimitive(SDF_ROUNDED_CONE, position, rotation, color, operation, blendFactor);
    prim.params[0] = height;
    prim.params[1] = r1;
    prim.params[2] = r2;
    return prim;
}

inline SDFPrimitive CreateRoundedCylinderPrim(float3 position, float4 rotation, float3 color, int operation, float blendFactor, float height, float r1, float r2) {
    SDFPrimitive prim = CreatePrimitive(SDF_ROUNDED_CYLINDER, position, rotation, color, operation, blendFactor);
    prim.params[0] = height;
    prim.params[1] = r1;
    prim.params[2] = r2;
    return prim;
}

// --- Shader Sources ---
// Moved from main.cpp
inline const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec4 aPos;
    layout (location = 1) in vec4 aNormal;  // NEW: Vertex normal
    layout (location = 2) in vec2 aTexCoord;
    layout (location = 3) in float aPrimitiveID;  // Back to float for compatibility
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 FragPosWorld;
    out vec3 Normal;  // NEW: Pass normal to fragment shader
    out vec2 TexCoord;
    out vec4 VertColor;
    flat out int PrimitiveID;  // NEW: flat = no interpolation

    void main() {
        vec4 worldPos = model * vec4(aPos.xyz, 1.0);
        gl_Position = projection * view * worldPos;
        FragPos = aPos.xyz;
        FragPosWorld = worldPos.xyz;
        
        // Transform normal to world space
        mat3 normalMatrix = transpose(inverse(mat3(model)));
        Normal = normalize(normalMatrix * aNormal.xyz);
        
        TexCoord = aTexCoord;
        VertColor = vec4(1.0);  // Default white since we're not using vertex colors
        PrimitiveID = int(aPrimitiveID);  // Convert float to int
    }
)";

inline const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec3 FragPos;
    in vec3 FragPosWorld;
    in vec3 Normal;  // NEW: Vertex normal from vertex shader
    in vec2 TexCoord;
    in vec4 VertColor;
    flat in int PrimitiveID;  // NEW: Primitive ID (flat = no interpolation)
    
    uniform float time;
    uniform samplerBuffer bvhNodes;
    uniform sampler2D texture1;  // Legacy single texture support
    uniform sampler2D atlasTexture;  // For atlas mode (single packed texture)
    uniform sampler2DArray textureArray;  // NEW: 2D texture array for multiple textures
    uniform int useTexture;  // 0 = SDF Color, 1 = Texture Array, 2 = Single Texture (legacy/atlas)
    uniform bool useVertexNormals;  // NEW: Toggle between vertex normals and recomputed normals
    
    struct Primitive {
        vec4 info; // x=type, y=op, z=disp, w=blend
        vec4 pos;  // xyz=pos, w=annular
        vec4 rot;  // quaternion
        vec4 scale; // xyz=scale, w=rounding
        vec4 color; // xyz=color, w=unused
        vec4 params1; // params[0-3]
        vec4 params2; // params[4-5], dispParams[0-1]
        vec4 params3; // dispParams[2-3], unused, unused
    };
    
    layout (std140) uniform Primitives {
        Primitive primitives[64]; // Max primitives
    };
    
    uniform int uNumBVHPrimitives; // Start of non-BVH primitives
    uniform int uNumTotalPrimitives;

    // --- Math Helpers ---
    float dot2(vec3 v) { return dot(v, v); }
    float dot2(vec2 v) { return dot(v, v); }
    vec3 abs_f3(vec3 v) { return abs(v); }
    vec3 max_f3(vec3 v, float f) { return max(v, vec3(f)); }
    
    vec3 rotateVector(vec3 v, vec4 q) {
        vec3 u = q.xyz;
        float s = q.w;
        return 2.0f * dot(u, v) * u + (s*s - dot(u, u)) * v + 2.0f * s * cross(u, v);
    }

    vec3 invRotateVector(vec3 v, vec4 q) {
        vec4 invQ = vec4(-q.x, -q.y, -q.z, q.w);
        return rotateVector(v, invQ);
    }
    
    // --- Primitives ---
    float sdSphere(vec3 p, float r) { return length(p) - r; }
    
    float sdBox(vec3 p, vec3 b) {
        vec3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    }
    
    float sdTorus(vec3 p, float r1, float r2) {
        vec2 q = vec2(length(p.xz) - r1, p.y);
        return length(q) - r2;
    }
    
    float sdCylinder(vec3 p, float h, float r) {
        vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    }
    
    float sdCapsule(vec3 p, float h, float r) {
        p.y -= clamp(p.y, 0.0, h);
        return length(p) - r;
    }
    
    float sdCone(vec3 p, float h, float angle) {
        // angle in degrees
        float r = radians(angle);
        vec2 c = vec2(sin(r), cos(r));
        vec2 q_u = vec2(h * (c.x/c.y), -h);
        vec2 pXZ = vec2(p.x, p.z);
        vec2 w_u = vec2(length(pXZ), p.y);
        float dot_wq = dot(w_u, q_u);
        float dot_qq = dot(q_u, q_u);
        float t = clamp(dot_wq / dot_qq, 0.0, 1.0);
        vec2 a = w_u - q_u * t;
        float t_b = clamp(w_u.x / q_u.x, 0.0, 1.0);
        vec2 b = w_u - vec2(q_u.x * t_b, q_u.y);
        float k = (q_u.y > 0.0) ? 1.0 : -1.0;
        float d = min(dot(a, a), dot(b, b));
        float s = max(k * (w_u.x * q_u.y - w_u.y * q_u.x), k * (w_u.y - q_u.y));
        return sqrt(d) * (s > 0.0 ? 1.0 : -1.0);
    }
    
    float sdRoundedBox(vec3 p, vec3 b, float r) {
        vec3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }
    
    float sdEllipsoid(vec3 p, vec3 r) {
        float k0 = length(p / r);
        float k1 = length(p / (r * r));
        return k0 * (k0 - 1.0) / k1;
    }
    
    float sdOctahedron(vec3 p, float s) {
        p = abs(p);
        return (p.x + p.y + p.z - s) * 0.57735027;
    }
    
    float sdTriPrism(vec3 p, vec2 h) {
        vec3 q = abs(p);
        return max(q.z - h.y, max(q.x * 0.866025 + p.y * 0.5, -p.y) - h.x * 0.5);
    }

    float sdRoundedCone(vec3 p, float h, float r1, float r2) {
        vec2 q = vec2(length(p.xz), p.y);
        float b = (r1 - r2) / h;
        float a = sqrt(1.0 - b * b);
        float k = dot(q, vec2(-b, a));
        if (k < 0.0) return length(q) - r1;
        if (k > a * h) return length(q - vec2(0.0, h)) - r2;
        return dot(q, vec2(a, b)) - r1;
    }

    float sdRoundedCylinder(vec3 p, float h, float r1, float r2) {
        vec2 d = vec2(length(p.xz) - 2.0 * r1 + r2, abs(p.y) - h);
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r2;
    }

    float sdHexPrism(vec3 p, vec2 h) {
        const vec3 k = vec3(-0.8660254, 0.5, 0.57735027);
        p = abs(p);
        p.xz -= 2.0 * min(dot(k.xy, p.xz), 0.0) * k.xy;
        vec2 d = vec2(
            length(p.xz - vec2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x)) * sign(p.z - h.x),
            p.y - h.y
        );
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    }

    // --- Displacements ---
    vec3 dispTwist(vec3 p, float k) {
        float c = cos(k * p.y);
        float s = sin(k * p.y);
        mat2 m = mat2(c, -s, s, c); // Rotation around Y
        vec2 xz = m * p.xz; 
        return vec3(xz.x, p.y, xz.y);
    }

    vec3 dispBend(vec3 p, float k) {
        float c = cos(k * p.x);
        float s = sin(k * p.x);
        mat2 m = mat2(c, -s, s, c); 
        vec2 xy = m * p.xy;
        return vec3(xy.x, xy.y, p.z);
    }

    // --- Ops ---
    float opSubtract(float d1, float d2) { return max(d1, -d2); }
    
    // --- Map ---
    
    bool intersectAABB(vec3 p, vec3 minB, vec3 maxB, float min_d) {
        vec3 d = max(max(minB - p, p - maxB), vec3(0.0));
        float dist = length(d);
        return dist < min_d;
    }

    void map(vec3 p_world, out float outDist, out vec3 outColor) {
        float d = 1e10;
        vec3 color = vec3(0.0);
        
        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0; // Root
        
        int iter = 0;
        // Limit iterations for safety
        while (stackPtr > 0 && iter < 256) {
            iter++;
            int nodeIdx = stack[--stackPtr];
            
            int texIdx = nodeIdx * 2;
            vec4 data0 = texelFetch(bvhNodes, texIdx);
            vec4 data1 = texelFetch(bvhNodes, texIdx + 1);
            
            vec3 nodeMin = data0.xyz;
            vec3 nodeMax = vec3(data0.w, data1.xy);
            int left = floatBitsToInt(data1.z);
            int right = floatBitsToInt(data1.w);
            
            // Pruning
            if (!intersectAABB(p_world, nodeMin, nodeMax, d)) continue;
            
            if (right == -1) { // Leaf
                int i = left;
                Primitive prim = primitives[i];
                
                int type = int(prim.info.x);
                int op = int(prim.info.y);
                int dispType = int(prim.info.z);
                float blend = prim.info.w;
                
                vec3 p = p_world - prim.pos.xyz;
                p = invRotateVector(p, prim.rot);
                p = p / prim.scale.xyz; 
                
                if (dispType == 2) p = dispTwist(p, prim.params2.z);
                else if (dispType == 3) p = dispBend(p, prim.params2.z);
                else if (dispType == 1) p.y += sin(p.x * prim.params2.z + time) * prim.params2.w;
                
                float d_prim = 1e10;
                if(type == 0) d_prim = sdSphere(p, prim.params1.x);
                else if(type == 1) d_prim = sdBox(p, prim.params1.xyz);
                else if(type == 2) d_prim = sdTorus(p, prim.params1.x, prim.params1.y);
                else if(type == 3) d_prim = sdCylinder(p, prim.params1.x, prim.params1.y);
                else if(type == 4) d_prim = sdCapsule(p, prim.params1.x, prim.params1.y);
                else if(type == 5) d_prim = sdCone(p, prim.params1.x, prim.params1.y);
                else if(type == 6) d_prim = sdRoundedBox(p, prim.params1.xyz, prim.scale.w);
                else if(type == 9) d_prim = sdEllipsoid(p, prim.params1.xyz);
                else if(type == 10) d_prim = sdTriPrism(p, prim.params1.xy);
                else if(type == 11) d_prim = sdOctahedron(p, prim.params1.x);
                else if(type == 7) d_prim = sdRoundedCylinder(p, prim.params1.x, prim.params1.y, prim.params1.z);
                else if(type == 8) d_prim = sdRoundedCone(p, prim.params1.x, prim.params1.y, prim.params1.z);
                else if(type == 12) d_prim = sdHexPrism(p, prim.params1.xy);
                
                float scaleFactor = min(prim.scale.x, min(prim.scale.y, prim.scale.z));
                d_prim *= scaleFactor;
                
                if(prim.pos.w > 0.0) d_prim = abs(d_prim) - prim.pos.w;
                if(prim.scale.w > 0.0 && type != 6) d_prim -= prim.scale.w; 
                
                // Combine
                if(d == 1e10) {
                    d = d_prim;
                    color = prim.color.rgb;
                } else {
                    if(op == 0) { // Union
                        if(d_prim < d) { d = d_prim; color = prim.color.rgb; }
                    } else if (op == 1) { // Subtract
                        float prev_d = d;
                        d = opSubtract(d, d_prim);
                        if (d > prev_d) { color = prim.color.rgb; }
                    } else if (op == 3) { // Union Blend
                        float h = clamp(0.5 + 0.5 * (d - d_prim) / blend, 0.0, 1.0);
                        d = mix(d, d_prim, h) - blend * h * (1.0 - h);
                        color = mix(color, prim.color.rgb, h);
                    } else if (op == 4) { // Subtract Blend
                        float h = clamp(0.5 - 0.5 * (d + d_prim) / blend, 0.0, 1.0);
                        d = mix(d, -d_prim, h) + blend * h * (1.0 - h);
                        color = mix(color, prim.color.rgb, h);
                    } else if (op == 2) { // Intersect
                        if (d_prim > d) { d = d_prim; color = prim.color.rgb; }
                    }
                }
            } else {
                 // Internal
                 stack[stackPtr++] = right;
                 stack[stackPtr++] = left;
            }
        }
        
        // Linear Pass for Non-Union Primitives (Global Modifiers)
        for(int i = uNumBVHPrimitives; i < uNumTotalPrimitives; i++) {
            Primitive prim = primitives[i];
            int type = int(prim.info.x);
            int op = int(prim.info.y);
            int dispType = int(prim.info.z);
            float blend = prim.info.w;
            
            vec3 p = p_world - prim.pos.xyz;
            p = invRotateVector(p, prim.rot);
            p = p / prim.scale.xyz; 
            
            // Displacements
            if (dispType == 2) p = dispTwist(p, prim.params2.z);
            else if (dispType == 3) p = dispBend(p, prim.params2.z);
            else if (dispType == 1) p.y += sin(p.x * prim.params2.z + time) * prim.params2.w;
            
            float d_prim = 1e10;
            if(type == 0) d_prim = sdSphere(p, prim.params1.x);
            else if(type == 1) d_prim = sdBox(p, prim.params1.xyz);
            else if(type == 2) d_prim = sdTorus(p, prim.params1.x, prim.params1.y);
            else if(type == 3) d_prim = sdCylinder(p, prim.params1.x, prim.params1.y);
            else if(type == 4) d_prim = sdCapsule(p, prim.params1.x, prim.params1.y);
            else if(type == 5) d_prim = sdCone(p, prim.params1.x, prim.params1.y);
            else if(type == 6) d_prim = sdRoundedBox(p, prim.params1.xyz, prim.scale.w);
            else if(type == 9) d_prim = sdEllipsoid(p, prim.params1.xyz);
            else if(type == 10) d_prim = sdTriPrism(p, prim.params1.xy);
            else if(type == 11) d_prim = sdOctahedron(p, prim.params1.x);
            else if(type == 7) d_prim = sdRoundedCylinder(p, prim.params1.x, prim.params1.y, prim.params1.z);
            else if(type == 8) d_prim = sdRoundedCone(p, prim.params1.x, prim.params1.y, prim.params1.z);
            else if(type == 12) d_prim = sdHexPrism(p, prim.params1.xy);
            
            float scaleFactor = min(prim.scale.x, min(prim.scale.y, prim.scale.z));
            d_prim *= scaleFactor;
            
            // Annular/Rounding
            if(prim.pos.w > 0.0) d_prim = abs(d_prim) - prim.pos.w;
            if(prim.scale.w > 0.0 && type != 6) d_prim -= prim.scale.w; 
            
            // Combine - Note: No need to check i==0 because this is post-BVH
            // Wait, if BVH was empty, d is 1e10.
            if(d == 1e10) {
                d = d_prim;
                color = prim.color.rgb;
            } else {
                if(op == 0) { // Union
                    if(d_prim < d) { d = d_prim; color = prim.color.rgb; }
                } else if (op == 1) { // Subtract
                    float prev_d = d;
                    d = opSubtract(d, d_prim);
                    if (d > prev_d) { color = prim.color.rgb; }
                } else if (op == 3) { // Union Blend
                    float h = clamp(0.5 + 0.5 * (d - d_prim) / blend, 0.0, 1.0);
                    d = mix(d, d_prim, h) - blend * h * (1.0 - h);
                    color = mix(color, prim.color.rgb, h);
                } else if (op == 4) { // Subtract Blend
                    float h = clamp(0.5 - 0.5 * (d + d_prim) / blend, 0.0, 1.0);
                    d = mix(d, -d_prim, h) + blend * h * (1.0 - h);
                    color = mix(color, prim.color.rgb, h);
                } else if (op == 2) { // Intersect
                    if (d_prim > d) { d = d_prim; color = prim.color.rgb; }
                }
            }
        }
        
        outDist = d;
        outColor = color;
    }

    void main() {
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        
        vec3 normal;
        if (useVertexNormals) {
            // Use precomputed vertex normals (accurate for displaced geometry)
            normal = normalize(Normal);
        } else {
            // Fallback: Recompute normal using SDF gradient (old method)
            vec2 e = vec2(0.001, 0.0);
            float d_center; vec3 c_center;
            map(FragPos, d_center, c_center);
            float d_x; vec3 c_x; map(FragPos + e.xyy, d_x, c_x);
            float d_y; vec3 c_y; map(FragPos + e.yxy, d_y, c_y);
            float d_z; vec3 c_z; map(FragPos + e.yyx, d_z, c_z);
            normal = normalize(vec3(d_x - d_center, d_y - d_center, d_z - d_center));
        }
        
        float d; 
        vec3 sdfColor;
        map(FragPos, d, sdfColor);
        
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 color;
        
        if (useTexture == 1) {
            // DIRECT MODE: Sample texture array using the primitive's textureID as layer
            int layer = int(primitives[PrimitiveID].params3.z + 0.5);
            vec3 texColor = texture(textureArray, vec3(TexCoord.xy, float(layer))).rgb;
            color = texColor * (diff + 0.2);
        } else if (useTexture == 2) {
            // SINGLE TEXTURE / ATLAS MODE: Use atlasTexture sampler
            vec3 texColor = texture(atlasTexture, TexCoord).rgb;
            color = texColor * (diff + 0.2);
        } else {
            // SDF COLOR MODE: Use vertex colors or SDF evaluated color
            color = sdfColor * (diff + 0.2);
        }
        
        FragColor = vec4(color, 1.0);
    }
)";

// --- Structs ---
struct GPUPrimitive {
    float info[4];   // x=type, y=op, z=disp, w=blend
    float pos[4];    // xyz=pos, w=annular
    float rot[4];    // quaternion
    float scale[4];  // xyz=scale, w=rounding
    float color[4];  // xyz=color, w=unused
    float params1[4]; // params[0-3]
    float params2[4]; // params[4-5], dispParams[0-1]
    float params3[4]; // dispParams[2-3], unused, unused
};

// --- BVH Construction Helpers ---
inline float3 transformPoint(float3 p, float3 pos, float4 rot, float3 scale) {
    p = make_float3(p.x * scale.x, p.y * scale.y, p.z * scale.z);
    p = rotateVector(p, rot);
    return p + pos;
}

inline void getPrimitiveAABB(const SDFPrimitive& prim, float3& minBox, float3& maxBox) {
    float3 localMin = make_float3(-1.0f, -1.0f, -1.0f);
    float3 localMax = make_float3(1.0f, 1.0f, 1.0f);
    
    float padding = 0.1f;
    if (prim.displacement != DISP_NONE) padding += 0.5f;
    
    float3 dims = make_float3(1.0f, 1.0f, 1.0f);
    if (prim.type == SDF_SPHERE) dims = make_float3(prim.params[0], prim.params[0], prim.params[0]);
    else if (prim.type == SDF_BOX || prim.type == SDF_ROUNDED_BOX) dims = make_float3(prim.params[0], prim.params[1], prim.params[2]);
    else if (prim.type == SDF_TORUS) dims = make_float3(prim.params[0] + prim.params[1], prim.params[1], prim.params[0] + prim.params[1]);
    else dims = make_float3(1.0f, 1.0f, 1.0f);
    
    dims = dims + make_float3(padding, padding, padding);
    localMin = make_float3(-dims.x, -dims.y, -dims.z);
    localMax = dims;

    float3 corners[8];
    corners[0] = make_float3(localMin.x, localMin.y, localMin.z);
    corners[1] = make_float3(localMax.x, localMin.y, localMin.z);
    corners[2] = make_float3(localMin.x, localMax.y, localMin.z);
    corners[3] = make_float3(localMax.x, localMax.y, localMin.z);
    corners[4] = make_float3(localMin.x, localMin.y, localMax.z);
    corners[5] = make_float3(localMax.x, localMin.y, localMax.z);
    corners[6] = make_float3(localMin.x, localMax.y, localMax.z);
    corners[7] = make_float3(localMax.x, localMax.y, localMax.z);
    
    minBox = make_float3(1e10f, 1e10f, 1e10f);
    maxBox = make_float3(-1e10f, -1e10f, -1e10f);
    
    for(int i=0; i<8; ++i) {
        float3 p = transformPoint(corners[i], prim.position, prim.rotation, prim.scale);
        minBox.x = fminf(minBox.x, p.x); minBox.y = fminf(minBox.y, p.y); minBox.z = fminf(minBox.z, p.z);
        maxBox.x = fmaxf(maxBox.x, p.x); maxBox.y = fmaxf(maxBox.y, p.y); maxBox.z = fmaxf(maxBox.z, p.z);
    }
}

inline int buildBVHRecursive(int* primIndices, int count, const std::vector<SDFPrimitive>& primitives, std::vector<BVHNode>& nodes) {
    BVHNode node;
    
    node.min[0] = 1e10f; node.min[1] = 1e10f; node.min[2] = 1e10f;
    node.max[0] = -1e10f; node.max[1] = -1e10f; node.max[2] = -1e10f;
    
    float3 centerCentroid = make_float3(0,0,0);
    
    for(int i=0; i<count; ++i) {
        float3 pMin, pMax;
        getPrimitiveAABB(primitives[primIndices[i]], pMin, pMax);
        
        node.min[0] = fminf(node.min[0], pMin.x); node.min[1] = fminf(node.min[1], pMin.y); node.min[2] = fminf(node.min[2], pMin.z);
        node.max[0] = fmaxf(node.max[0], pMax.x); node.max[1] = fmaxf(node.max[1], pMax.y); node.max[2] = fmaxf(node.max[2], pMax.z);
        
        centerCentroid = centerCentroid + (pMin + pMax) * 0.5f;
    }
    centerCentroid = centerCentroid * (1.0f / count);
    
    if (count <= 1) {
        node.left = primIndices[0];
        node.right = -1;
        nodes.push_back(node);
        return nodes.size() - 1;
    }
    
    int axis = 0;
    float3 extent = make_float3(node.max[0]-node.min[0], node.max[1]-node.min[1], node.max[2]-node.min[2]);
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.y && extent.z > extent.x) axis = 2;
    
    float splitPos = (axis == 0) ? centerCentroid.x : (axis == 1) ? centerCentroid.y : centerCentroid.z;
    
    int mid = 0;
    for(int i=0; i<count; ++i) {
        float3 pMin, pMax;
        getPrimitiveAABB(primitives[primIndices[i]], pMin, pMax);
        float c = (axis == 0) ? (pMin.x+pMax.x)*0.5f : (axis == 1) ? (pMin.y+pMax.y)*0.5f : (pMin.z+pMax.z)*0.5f;
        
        if (c < splitPos) {
            std::swap(primIndices[i], primIndices[mid]);
            mid++;
        }
    }
    
    if (mid == 0 || mid == count) {
        mid = count / 2;
    }
    
    int currentIndex = nodes.size();
    nodes.push_back(node);
    
    int leftChild = buildBVHRecursive(primIndices, mid, primitives, nodes);
    int rightChild = buildBVHRecursive(primIndices + mid, count - mid, primitives, nodes);
    
    nodes[currentIndex].left = leftChild;
    nodes[currentIndex].right = rightChild;
    
    return currentIndex;
}

inline void buildBVH(const std::vector<SDFPrimitive>& primitives, std::vector<BVHNode>& nodes) {
    nodes.clear();
    if (primitives.empty()) return;
    
    std::vector<int> indices(primitives.size());
    for(size_t i=0; i<primitives.size(); ++i) indices[i] = i;
    
    buildBVHRecursive(indices.data(), indices.size(), primitives, nodes);
}

inline void packPrimitives(const std::vector<SDFPrimitive>& primitives, std::vector<GPUPrimitive>& gpuPrimitives) {
    gpuPrimitives.resize(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i) {
        const SDFPrimitive& src = primitives[i];
        GPUPrimitive& dst = gpuPrimitives[i];
        
        dst.info[0] = (float)src.type;
        dst.info[1] = (float)src.operation;
        dst.info[2] = (float)src.displacement;
        dst.info[3] = src.blendFactor;
        
        dst.pos[0] = src.position.x; dst.pos[1] = src.position.y; dst.pos[2] = src.position.z;
        dst.pos[3] = src.annular;
        
        dst.rot[0] = src.rotation.x; dst.rot[1] = src.rotation.y; dst.rot[2] = src.rotation.z; dst.rot[3] = src.rotation.w;
        
        dst.scale[0] = src.scale.x; dst.scale[1] = src.scale.y; dst.scale[2] = src.scale.z;
        dst.scale[3] = src.rounding;
        
        dst.color[0] = src.color.x; dst.color[1] = src.color.y; dst.color[2] = src.color.z; dst.color[3] = 0.0f;
        
        dst.params1[0] = src.params[0]; dst.params1[1] = src.params[1]; dst.params1[2] = src.params[2]; dst.params1[3] = src.params[3];
        dst.params2[0] = src.params[4]; dst.params2[1] = src.params[5]; 
        dst.params2[2] = src.dispParams[0]; dst.params2[3] = src.dispParams[1];
        dst.params3[0] = src.dispParams[2];
        dst.params3[1] = src.dispParams[3];
        dst.params3[2] = (float)src.textureID;      // expose texture layer to shader
        dst.params3[3] = (float)src.atlasIslandID;  // reserved for atlas tooling/debug
    }
}

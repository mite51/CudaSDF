#include "Commons.cuh"
#include "MarchingCubesTables.cuh"
#include <device_launch_parameters.h>

#define EMPTY_SUBGRID -1

// Math constants for CUDA (not always defined by default)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

__constant__ float isoThreshold = 0.0f;

// --------------------------------------------------------------------------
// Robust helpers for Marching Cubes vertex interpolation / degenerate checks
// --------------------------------------------------------------------------

__device__ inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline bool isFinite3(const float3& v) {
    return !(isnan(v.x) || isnan(v.y) || isnan(v.z) || isinf(v.x) || isinf(v.y) || isinf(v.z));
}

__device__ inline float safeEdgeT(float val1, float val2) {
    // Prevent NaNs/Infs when val2 == val1, and reduce exact-endpoint collapses.
    const float denom = val2 - val1;
    float t = 0.5f;
    if (fabsf(denom) > 1e-12f) {
        t = (isoThreshold - val1) / denom;
    }
    // Avoid exact endpoints: if we hit exactly 0/1 due to equalities, nudge inward.
    t = clamp(t, 0.0f, 1.0f);
    const float endEps = 1e-7f;
    if (t <= 0.0f) t = endEps;
    if (t >= 1.0f) t = 1.0f - endEps;
    return t;
}

__device__ inline float3 lerp3(const float3& a, const float3& b, float t) {
    return make_float3(
        a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t
    );
}

__device__ inline float len2_3(const float3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ inline float3 interpEdgePoint(const SDFGrid& grid, const int4& xyz, const float cubeVal[8], int edgeId) {
    const int c1 = edgeConnections[edgeId][0];
    const int c2 = edgeConnections[edgeId][1];

    const float val1 = cubeVal[c1];
    const float val2 = cubeVal[c2];
    const float t = safeEdgeT(val1, val2);

    const int4 pos1I = make_int4(
        xyz.x + marchingCubeCorners[c1][0],
        xyz.y + marchingCubeCorners[c1][1],
        xyz.z + marchingCubeCorners[c1][2], 0);

    const int4 pos2I = make_int4(
        xyz.x + marchingCubeCorners[c2][0],
        xyz.y + marchingCubeCorners[c2][1],
        xyz.z + marchingCubeCorners[c2][2], 0);

    const float3 p1 = getLocation(grid, pos1I);
    const float3 p2 = getLocation(grid, pos2I);
    return lerp3(p1, p2, t);
}

__device__ inline bool isDegenerateTri(const float3 p[3], float cellSize) {
    // Reject NaNs/Infs
    if (!isFinite3(p[0]) || !isFinite3(p[1]) || !isFinite3(p[2])) return true;

    // Reject collapsed vertices (very short edges)
    //const float minEdge = fmaxf(cellSize * 1e-4f, 1e-7f);
    const float minEdge = cellSize / 1024.0f;
    const float minEdge2 = minEdge * minEdge;
    const float3 e01 = p[1] - p[0];
    const float3 e12 = p[2] - p[1];
    const float3 e20 = p[0] - p[2];
    if (len2_3(e01) < minEdge2 || len2_3(e12) < minEdge2 || len2_3(e20) < minEdge2) return true;

    // Reject near-zero area
    /*
    // JW - removing these cause manifold edges :/
    const float3 n = cross3(e01, p[2] - p[0]);
    if (len2_3(n) < 1e-18f) return true;
    */
    return false;
}

// --------------------------------------------------------------------------
// Math Helpers
// --------------------------------------------------------------------------

__device__ inline float dot2(float2 v) { return dot(make_float3(v.x, v.y, 0), make_float3(v.x, v.y, 0)); }
__device__ inline float dot2(float3 v) { return dot(v, v); }
__device__ inline float length(float3 v) { return sqrtf(dot(v, v)); }
__device__ inline float length(float2 v) { return sqrtf(v.x * v.x + v.y * v.y); }
__device__ inline float3 abs_f3(float3 v) { return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
__device__ inline float2 abs_f2(float2 v) { return make_float2(fabsf(v.x), fabsf(v.y)); }
__device__ inline float2 max_f2(float2 v, float f) { return make_float2(fmaxf(v.x, f), fmaxf(v.y, f)); }
__device__ inline float2 min_f2(float2 v, float f) { return make_float2(fminf(v.x, f), fminf(v.y, f)); }
__device__ inline float3 max_f3(float3 v, float f) { return make_float3(fmaxf(v.x, f), fmaxf(v.y, f), fmaxf(v.z, f)); }
__device__ inline float3 min_f3(float3 v, float f) { return make_float3(fminf(v.x, f), fminf(v.y, f), fminf(v.z, f)); }
__device__ inline float3 max_f3(float3 a, float3 b) { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }

__device__ inline float3 mix(float3 a, float3 b, float t) {
    return make_float3(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t, a.z * (1.0f - t) + b.z * t);
}

// --------------------------------------------------------------------------
// SDF Primitives
// --------------------------------------------------------------------------

__device__ float sdSphere(float3 p, float r) {
    return length(p) - r;
}

__device__ float sdBox(float3 p, float3 b) {
    float3 q = abs_f3(p) - b;
    return length(max_f3(q, 0.0f)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
}

__device__ float sdTorus(float3 p, float r1, float r2) {
    float2 q = make_float2(length(make_float2(p.x, p.z)) - r1, p.y);
    return length(q) - r2;
}

__device__ float sdCylinder(float3 p, float h, float r) {
    float2 d = abs_f2(make_float2(length(make_float2(p.x, p.z)), p.y)) - make_float2(r, h);
    return fminf(fmaxf(d.x, d.y), 0.0f) + length(max_f3(make_float3(d.x, d.y, 0.0f), 0.0f));
}

__device__ float sdCapsule(float3 p, float h, float r) {
    p.y -= clamp(p.y, 0.0f, h);
    return length(p) - r;
}

__device__ float sdCone(float3 p, float h, float angle) {
    float2 c = make_float2(sinf(angle * 0.0174533f), cosf(angle * 0.0174533f));
    float2 q_u = make_float2(h * (c.x/c.y), -h);
    float2 pXZ = make_float2(p.x, p.z);
    float2 w_u = make_float2(length(pXZ), p.y);
    float dot_wq = w_u.x * q_u.x + w_u.y * q_u.y;
    float dot_qq = q_u.x * q_u.x + q_u.y * q_u.y;
    float t = clamp(dot_wq / dot_qq, 0.0f, 1.0f);
    float2 a = w_u - make_float2(q_u.x * t, q_u.y * t);
    float t_b = clamp(w_u.x / q_u.x, 0.0f, 1.0f);
    float2 b = w_u - make_float2(q_u.x * t_b, q_u.y);
    float k = (q_u.y > 0) ? 1.0f : -1.0f;
    float d = fminf(dot2(a), dot2(b));
    float s = fmaxf(k * (w_u.x * q_u.y - w_u.y * q_u.x), k * (w_u.y - q_u.y));
    return sqrtf(d) * (s > 0.0f ? 1.0f : -1.0f);
}

__device__ float sdRoundedBox(float3 p, float3 b, float r) {
    float3 q = abs_f3(p) - b;
    return length(max_f3(q, 0.0f)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f) - r;
}

__device__ float sdEllipsoid(float3 p, float3 r) {
    float k0 = length(make_float3(p.x / r.x, p.y / r.y, p.z / r.z));
    float k1 = length(make_float3(p.x / (r.x * r.x), p.y / (r.y * r.y), p.z / (r.z * r.z)));
    return k0 * (k0 - 1.0f) / k1;
}

__device__ float sdOctahedron(float3 p, float s) {
    p = abs_f3(p);
    return (p.x + p.y + p.z - s) * 0.57735027f;
}

__device__ float sdTriPrism(float3 p, float2 h) {
    float3 q = abs_f3(p);
    return fmaxf(q.z - h.y, fmaxf(q.x * 0.866025f + p.y * 0.5f, -p.y) - h.x * 0.5f);
}

__device__ float sdRoundedCone(float3 p, float h, float r1, float r2) {
    float2 q = make_float2(length(make_float2(p.x, p.z)), p.y);
    float b = (r1 - r2) / h;
    float a = sqrtf(1.0f - b * b);
    float k = q.x * (-b) + q.y * a; 
    if (k < 0.0f) return length(q) - r1;
    if (k > a * h) return length(q - make_float2(0.0f, h)) - r2;
    return (q.x * a + q.y * b) - r1; 
}

__device__ float sdRoundedCylinder(float3 p, float h, float r1, float r2) {
    float2 d = make_float2(length(make_float2(p.x, p.z)) - 2.0f * r1 + r2, fabsf(p.y) - h);
    return fminf(fmaxf(d.x, d.y), 0.0f) + length(max_f2(d, 0.0f)) - r2;
}

__device__ float sdHexPrism(float3 p, float2 h) {
    const float3 k = make_float3(-0.8660254f, 0.5f, 0.57735027f);
    p = abs_f3(p);
    float2 p_xz = make_float2(p.x, p.z);
    float dot_k_p = k.x * p_xz.x + k.y * p_xz.y;
    float2 k_xy = make_float2(k.x, k.y);
    float2 offset = k_xy * 2.0f * fminf(dot_k_p, 0.0f);
    p_xz = p_xz - offset;
    float2 d = make_float2(
        length(p_xz - make_float2(clamp(p_xz.x, -k.z * h.x, k.z * h.x), h.x)) * copysignf(1.0f, p_xz.y - h.x),
        p.y - h.y
    );
    return fminf(fmaxf(d.x, d.y), 0.0f) + length(max_f2(d, 0.0f));
}

// --------------------------------------------------------------------------
// Operations & Displacements
// --------------------------------------------------------------------------

__device__ float opUnion(float d1, float d2) { return fminf(d1, d2); }
__device__ float opSubtract(float d1, float d2) { return fmaxf(d1, -d2); }
__device__ float opIntersect(float d1, float d2) { return fmaxf(d1, d2); }

__device__ float3 dispTwist(float3 p, float k) {
    float c = cosf(k * p.y);
    float s = sinf(k * p.y);
    float px = c * p.x - s * p.z;
    float pz = s * p.x + c * p.z;
    return make_float3(px, p.y, pz);
}

__device__ float3 dispBend(float3 p, float k) {
    float c = cosf(k * p.x);
    float s = sinf(k * p.x);
    float px = c * p.x - s * p.y;
    float py = s * p.x + c * p.y;
    return make_float3(px, py, p.z);
}

// --------------------------------------------------------------------------
// UV Mapping Functions
// --------------------------------------------------------------------------

__device__ float2 uvSphere(float3 p) {
    float len = length(p);
    if (len < 1e-6f) return make_float2(0.5f, 0.5f);
    
    p = p * (1.0f / len);  // Normalize
    float theta = atan2f(p.z, p.x);
    float phi = asinf(clamp(p.y, -1.0f, 1.0f));
    
    return make_float2(
        (theta + M_PI) / (2.0f * M_PI),
        (phi + M_PI_2) / M_PI
    );
}

__device__ float2 uvBox(float3 p, float3 normal) {
    float3 absN = make_float3(fabsf(normal.x), fabsf(normal.y), fabsf(normal.z));
    
    if (absN.x > absN.y && absN.x > absN.z) {
        // X-face
        return make_float2(p.z * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
    } else if (absN.y > absN.z) {
        // Y-face
        return make_float2(p.x * 0.5f + 0.5f, p.z * 0.5f + 0.5f);
    } else {
        // Z-face
        return make_float2(p.x * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
    }
}

__device__ float2 uvCylinder(float3 p, float height) {
    float theta = atan2f(p.z, p.x);
    float u = (theta + M_PI) / (2.0f * M_PI);
    float v = clamp((p.y / height) * 0.5f + 0.5f, 0.0f, 1.0f);
    return make_float2(u, v);
}

__device__ float2 uvTorus(float3 p, float majorRadius, float minorRadius) {
    float theta = atan2f(p.z, p.x);
    
    float2 q = make_float2(length(make_float2(p.x, p.z)) - majorRadius, p.y);
    float phi = atan2f(q.y, q.x);
    
    return make_float2(
        (theta + M_PI) / (2.0f * M_PI),
        (phi + M_PI) / (2.0f * M_PI)
    );
}

__device__ float2 uvCapsule(float3 p, float height, float radius) {
    // Similar to cylinder but handle hemispherical caps
    if (p.y > height) {
        // Top cap (sphere)
        float3 pCap = make_float3(p.x, p.y - height, p.z);
        return uvSphere(pCap);
    } else if (p.y < 0.0f) {
        // Bottom cap (sphere)
        return uvSphere(p);
    } else {
        // Cylindrical body
        return uvCylinder(p, height);
    }
}

__device__ float2 uvCone(float3 p, float height) {
    // Similar to cylinder
    float theta = atan2f(p.z, p.x);
    float u = (theta + M_PI) / (2.0f * M_PI);
    float v = clamp((p.y / height) * 0.5f + 0.5f, 0.0f, 1.0f);
    return make_float2(u, v);
}

__device__ float2 transformUV(float2 uv, float2 scale, float2 offset, float rotation) {
    // Apply scale
    uv = make_float2(uv.x * scale.x, uv.y * scale.y);
    
    // Apply rotation
    if (fabsf(rotation) > 1e-6f) {
        float c = cosf(rotation);
        float s = sinf(rotation);
        float2 rotated = make_float2(
            uv.x * c - uv.y * s,
            uv.x * s + uv.y * c
        );
        uv = rotated;
    }
    
    // Apply offset
    uv = uv + offset;
    
    return uv;
}

// --------------------------------------------------------------------------
// Seam-Aware Triangle UV Computation
// --------------------------------------------------------------------------

// Forward declare approximateLocalNormal (defined after computePrimitiveUV)
__device__ float3 approximateLocalNormal(float3 localPos, const SDFPrimitive& prim);

/**
 * Detects if a UV coordinate wraps across a seam and adjusts it to maintain coherence.
 * @param uv The UV coordinate to potentially adjust
 * @param reference The reference UV coordinate to compare against
 * @param wrapThreshold The threshold for detecting a wrap (typically 0.5)
 * @return The adjusted UV coordinate that is continuous with the reference
 */
__device__ float2 adjustUVForSeam(float2 uv, float2 reference, float wrapThreshold = 0.5f) {
    float2 result = uv;
    
    // Check U coordinate
    float deltaU = fabsf(uv.x - reference.x);
    if (deltaU > wrapThreshold) {
        // Seam detected - adjust to maintain coherence
        if (uv.x < reference.x) {
            result.x += 1.0f;  // Wrap up (0.02 becomes 1.02 to stay near 0.98)
        } else {
            result.x -= 1.0f;  // Wrap down (0.98 becomes -0.02 to stay near 0.02)
        }
    }
    
    // Check V coordinate
    float deltaV = fabsf(uv.y - reference.y);
    if (deltaV > wrapThreshold) {
        // Seam detected - adjust to maintain coherence
        if (uv.y < reference.y) {
            result.y += 1.0f;
        } else {
            result.y -= 1.0f;
        }
    }
    
    return result;
}

/**
 * Computes UVs for a triangle with seam awareness.
 * This function computes all three UVs together and adjusts them to ensure
 * they are continuous, even if that means going outside [0,1] range.
 * 
 * @param localPos Array of 3 local space positions
 * @param prim The primitive being mapped
 * @param time Current animation time
 * @param outUVs Output array of 3 UV coordinates (must be pre-allocated)
 */
__device__ void computeTriangleUVs(
    const float3 localPos[3],
    const SDFPrimitive& prim,
    float time,
    float2 outUVs[3]
) {
    // Compute normal approximations for vertices that need them
    float3 localNormals[3];
    bool needsNormals = (prim.type == SDF_BOX || 
                        prim.type == SDF_ROUNDED_BOX || 
                        prim.type == SDF_OCTOHEDRON);
    
    if (needsNormals) {
        for (int i = 0; i < 3; ++i) {
            localNormals[i] = approximateLocalNormal(localPos[i], prim);
        }
    }
    
    // Step 1: Compute raw UVs for all three vertices
    for (int i = 0; i < 3; ++i) {
        float2 uv = make_float2(0.0f, 0.0f);
        
        switch(prim.type) {
            case SDF_SPHERE:
                uv = uvSphere(localPos[i]);
                break;
            case SDF_BOX:
            case SDF_ROUNDED_BOX:
                uv = uvBox(localPos[i], localNormals[i]);
                break;
            case SDF_CYLINDER:
            case SDF_ROUNDED_CYLINDER:
                uv = uvCylinder(localPos[i], prim.params[0]);
                break;
            case SDF_TORUS:
                uv = uvTorus(localPos[i], prim.params[0], prim.params[1]);
                break;
            case SDF_CAPSULE:
                uv = uvCapsule(localPos[i], prim.params[0], prim.params[1]);
                break;
            case SDF_CONE:
            case SDF_ROUNDED_CONE:
                uv = uvCone(localPos[i], prim.params[0]);
                break;
            case SDF_HEX_PRISM:
            case SDF_TRIANGULAR_PRISM:
                uv = uvCylinder(localPos[i], prim.params[1]);
                break;
            case SDF_ELLIPSOID:
                uv = uvSphere(localPos[i]);
                break;
            case SDF_OCTOHEDRON:
                uv = uvBox(localPos[i], localNormals[i]);
                break;
            default:
                // Planar fallback
                uv = make_float2(localPos[i].x * 0.5f + 0.5f, localPos[i].y * 0.5f + 0.5f);
                break;
        }
        
        outUVs[i] = uv;
    }
    
    // Step 2: Detect and fix seam wrapping
    // Use vertex 0 as the reference, adjust vertices 1 and 2 to be continuous with it
    outUVs[1] = adjustUVForSeam(outUVs[1], outUVs[0]);
    outUVs[2] = adjustUVForSeam(outUVs[2], outUVs[0]);
    
    // Additional check: if vertex 2 wrapped relative to vertex 0,
    // but vertex 1 is actually closer to the wrapped vertex 2,
    // we might need to adjust vertex 0 and 1 instead.
    // This handles cases where the reference vertex is the outlier.
    float dist_01 = fabsf(outUVs[0].x - outUVs[1].x) + fabsf(outUVs[0].y - outUVs[1].y);
    float dist_02 = fabsf(outUVs[0].x - outUVs[2].x) + fabsf(outUVs[0].y - outUVs[2].y);
    float dist_12 = fabsf(outUVs[1].x - outUVs[2].x) + fabsf(outUVs[1].y - outUVs[2].y);
    
    // If vertex 0 is far from both 1 and 2, but 1 and 2 are close,
    // then vertex 0 is the outlier - adjust it to match vertex 1
    if (dist_01 > 0.5f && dist_02 > 0.5f && dist_12 < 0.5f) {
        outUVs[0] = adjustUVForSeam(outUVs[0], outUVs[1]);
    }
    
    // Step 3: Apply UV transforms to all vertices
    for (int i = 0; i < 3; ++i) {
        outUVs[i] = transformUV(outUVs[i], prim.uvScale, prim.uvOffset, prim.uvRotation);
    }
}

__device__ float3 approximateLocalNormal(float3 localPos, const SDFPrimitive& prim) {
    // Simple gradient approximation based on primitive type
    if (prim.type == SDF_SPHERE) {
        float len = length(localPos);
        return (len > 1e-6f) ? (localPos * (1.0f / len)) : make_float3(0.0f, 1.0f, 0.0f);
    } else if (prim.type == SDF_BOX || prim.type == SDF_ROUNDED_BOX) {
        float3 absP = abs_f3(localPos);
        if (absP.x > absP.y && absP.x > absP.z) {
            return make_float3(copysignf(1.0f, localPos.x), 0.0f, 0.0f);
        } else if (absP.y > absP.z) {
            return make_float3(0.0f, copysignf(1.0f, localPos.y), 0.0f);
        } else {
            return make_float3(0.0f, 0.0f, copysignf(1.0f, localPos.z));
        }
    }
    
    // Default: use position direction
    float len = length(localPos);
    return (len > 1e-6f) ? (localPos * (1.0f / len)) : make_float3(0.0f, 1.0f, 0.0f);
}

__device__ float2 computePrimitiveUV(
    const SDFPrimitive& prim,
    float3 localPos,
    float time
) {
    float2 uv = make_float2(0.0f, 0.0f);
    
    // Compute normal approximation
    float3 localNormal = approximateLocalNormal(localPos, prim);
    
    // Select UV mapping based on primitive type
    switch(prim.type) {
        case SDF_SPHERE:
            uv = uvSphere(localPos);
            break;
        case SDF_BOX:
        case SDF_ROUNDED_BOX:
            uv = uvBox(localPos, localNormal);
            break;
        case SDF_CYLINDER:
        case SDF_ROUNDED_CYLINDER:
            uv = uvCylinder(localPos, prim.params[0]);
            break;
        case SDF_TORUS:
            uv = uvTorus(localPos, prim.params[0], prim.params[1]);
            break;
        case SDF_CAPSULE:
            uv = uvCapsule(localPos, prim.params[0], prim.params[1]);
            break;
        case SDF_CONE:
        case SDF_ROUNDED_CONE:
            uv = uvCone(localPos, prim.params[0]);
            break;
        case SDF_HEX_PRISM:
        case SDF_TRIANGULAR_PRISM:
            // Use cylindrical approximation
            uv = uvCylinder(localPos, prim.params[1]);
            break;
        case SDF_ELLIPSOID:
            // Use spherical mapping
            uv = uvSphere(localPos);
            break;
        case SDF_OCTOHEDRON:
            // Use box-like mapping
            uv = uvBox(localPos, localNormal);
            break;
        default:
            // Planar fallback (XY projection)
            uv = make_float2(localPos.x * 0.5f + 0.5f, localPos.y * 0.5f + 0.5f);
            break;
    }
    
    // Apply UV transforms
    uv = transformUV(uv, prim.uvScale, prim.uvOffset, prim.uvRotation);
    
    return uv;
}

// --------------------------------------------------------------------------
// SDF Evaluation
// --------------------------------------------------------------------------

__device__ bool intersectAABB(float3 p, float3 minB, float3 maxB, float min_d) {
    float3 d = max_f3(max_f3(minB - p, p - maxB), 0.0f);
    float dist = length(d);
    return dist < min_d;
}

__device__ void map(float3 p_world, const SDFGrid& grid, float time, float& outDist, float3& outColor, int& outPrimitiveID, float3& outLocalPos) {
    float d = 1e10f;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    int closestPrimID = -1;
    float3 closestLocalPos = make_float3(0.0f, 0.0f, 0.0f);

    // BVH Traversal
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Root
    
    int iter = 0;
    while (stackPtr > 0 && iter < 256) {
        iter++;
        int nodeIdx = stack[--stackPtr];
        BVHNode node = grid.d_bvhNodes[nodeIdx];
        
        float3 nodeMin = make_float3(node.min[0], node.min[1], node.min[2]);
        float3 nodeMax = make_float3(node.max[0], node.max[1], node.max[2]);
        
        if (!intersectAABB(p_world, nodeMin, nodeMax, d)) continue;
        
        if (node.right == -1) { // Leaf
            int i = node.left;
            SDFPrimitive prim = grid.d_primitives[i];
            
            // Transform World -> Local
            float3 p = p_world - prim.position;
            p = invRotateVector(p, prim.rotation);
            p = make_float3(p.x / prim.scale.x, p.y / prim.scale.y, p.z / prim.scale.z);
            
            // STORE PRE-DISTORTION POSITION (for UV calculation)
            float3 p_preDisp = p;
            
            if (prim.displacement == DISP_TWIST) p = dispTwist(p, prim.dispParams[0]);
            else if (prim.displacement == DISP_BEND) p = dispBend(p, prim.dispParams[0]);
            else if (prim.displacement == DISP_SINE) p.y += sinf(p.x * prim.dispParams[0] + time) * prim.dispParams[1];
            
            float d_prim = 1e10f;
            switch(prim.type) {
                case SDF_SPHERE: d_prim = sdSphere(p, prim.params[0]); break;
                case SDF_BOX: d_prim = sdBox(p, make_float3(prim.params[0], prim.params[1], prim.params[2])); break;
                case SDF_TORUS: d_prim = sdTorus(p, prim.params[0], prim.params[1]); break;
                case SDF_CYLINDER: d_prim = sdCylinder(p, prim.params[0], prim.params[1]); break;
                case SDF_CAPSULE: d_prim = sdCapsule(p, prim.params[0], prim.params[1]); break;
                case SDF_CONE: d_prim = sdCone(p, prim.params[0], prim.params[1]); break;
                case SDF_ROUNDED_BOX: d_prim = sdRoundedBox(p, make_float3(prim.params[0], prim.params[1], prim.params[2]), prim.params[3]); break;
                case SDF_ELLIPSOID: d_prim = sdEllipsoid(p, make_float3(prim.params[0], prim.params[1], prim.params[2])); break;
                case SDF_OCTOHEDRON: d_prim = sdOctahedron(p, prim.params[0]); break;
                case SDF_TRIANGULAR_PRISM: d_prim = sdTriPrism(p, make_float2(prim.params[0], prim.params[1])); break;
                case SDF_ROUNDED_CYLINDER: d_prim = sdRoundedCylinder(p, prim.params[0], prim.params[1], prim.params[2]); break;
                case SDF_ROUNDED_CONE: d_prim = sdRoundedCone(p, prim.params[0], prim.params[1], prim.params[2]); break;
                case SDF_HEX_PRISM: d_prim = sdHexPrism(p, make_float2(prim.params[0], prim.params[1])); break;
                default: d_prim = length(p) - 1.0f; break;
            }
            
            d_prim *= fminf(prim.scale.x, fminf(prim.scale.y, prim.scale.z));

            if (prim.annular > 0.0f) d_prim = fabsf(d_prim) - prim.annular;
            if (prim.rounding > 0.0f) d_prim -= prim.rounding;

            if (d == 1e10f) {
                d = d_prim;
                color = prim.color;
                closestPrimID = i;
                closestLocalPos = p_preDisp;
            } else {
                float h = 0.0f;
                switch(prim.operation) {
                    case SDF_UNION:
                        if (d_prim < d) { 
                            d = d_prim; 
                            color = prim.color;
                            closestPrimID = i;
                            closestLocalPos = p_preDisp;
                        }
                        break;
                    case SDF_SUBTRACT:
                        {
                            float d_old = d;
                            d = opSubtract(d, d_prim);
                            if (d > d_old) {  // Subtracted surface is visible
                                color = prim.color;
                                closestPrimID = i;  // Interior uses cutter's UV
                                closestLocalPos = p_preDisp;
                            }
                        }
                        break;
                    case SDF_INTERSECT:
                        if (d_prim > d) { 
                            d = d_prim; 
                            color = prim.color;
                            closestPrimID = i;
                            closestLocalPos = p_preDisp;
                        }
                        break;
                    case SDF_UNION_BLEND:
                        h = clamp(0.5f + 0.5f * (d - d_prim) / prim.blendFactor, 0.0f, 1.0f);
                        d = mix(d, d_prim, h) - prim.blendFactor * h * (1.0f - h);
                        color = mix(color, prim.color, h);
                        // Use dominant primitive for blending
                        if (d_prim < d || h > 0.5f) {
                            closestPrimID = i;
                            closestLocalPos = p_preDisp;
                        }
                        break;
                    case SDF_SUBTRACT_BLEND:
                        h = clamp(0.5f - 0.5f * (d + d_prim) / prim.blendFactor, 0.0f, 1.0f);
                        d = mix(d, -d_prim, h) + prim.blendFactor * h * (1.0f - h);
                        color = mix(color, prim.color, h);
                        if (h > 0.5f) {
                            closestPrimID = i;
                            closestLocalPos = p_preDisp;
                        }
                        break;
                    case SDF_INTERSECT_BLEND:
                        h = clamp(0.5f - 0.5f * (d - d_prim) / prim.blendFactor, 0.0f, 1.0f);
                        d = mix(d, d_prim, h) + prim.blendFactor * h * (1.0f - h);
                        color = mix(color, prim.color, h);
                        if (d_prim > d || h > 0.5f) {
                            closestPrimID = i;
                            closestLocalPos = p_preDisp;
                        }
                        break;
                }
            }
        } else {
            stack[stackPtr++] = node.right;
            stack[stackPtr++] = node.left;
        }
    }
    
    // Global Primitives (Non-Union)
    for (int i = grid.numBVHPrimitives; i < grid.numPrimitives; ++i) {
        SDFPrimitive prim = grid.d_primitives[i];
        
        float3 p = p_world - prim.position;
        p = invRotateVector(p, prim.rotation);
        p = make_float3(p.x / prim.scale.x, p.y / prim.scale.y, p.z / prim.scale.z);
        
        // STORE PRE-DISTORTION POSITION
        float3 p_preDisp = p;
        
        if (prim.displacement == DISP_TWIST) p = dispTwist(p, prim.dispParams[0]);
        else if (prim.displacement == DISP_BEND) p = dispBend(p, prim.dispParams[0]);
        else if (prim.displacement == DISP_SINE) p.y += sinf(p.x * prim.dispParams[0] + time) * prim.dispParams[1];
        
        float d_prim = 1e10f;
        switch(prim.type) {
            case SDF_SPHERE: d_prim = sdSphere(p, prim.params[0]); break;
            case SDF_BOX: d_prim = sdBox(p, make_float3(prim.params[0], prim.params[1], prim.params[2])); break;
            case SDF_TORUS: d_prim = sdTorus(p, prim.params[0], prim.params[1]); break;
            case SDF_CYLINDER: d_prim = sdCylinder(p, prim.params[0], prim.params[1]); break;
            case SDF_CAPSULE: d_prim = sdCapsule(p, prim.params[0], prim.params[1]); break;
            case SDF_CONE: d_prim = sdCone(p, prim.params[0], prim.params[1]); break;
            case SDF_ROUNDED_BOX: d_prim = sdRoundedBox(p, make_float3(prim.params[0], prim.params[1], prim.params[2]), prim.params[3]); break;
            case SDF_ELLIPSOID: d_prim = sdEllipsoid(p, make_float3(prim.params[0], prim.params[1], prim.params[2])); break;
            case SDF_OCTOHEDRON: d_prim = sdOctahedron(p, prim.params[0]); break;
            case SDF_TRIANGULAR_PRISM: d_prim = sdTriPrism(p, make_float2(prim.params[0], prim.params[1])); break;
            case SDF_ROUNDED_CYLINDER: d_prim = sdRoundedCylinder(p, prim.params[0], prim.params[1], prim.params[2]); break;
            case SDF_ROUNDED_CONE: d_prim = sdRoundedCone(p, prim.params[0], prim.params[1], prim.params[2]); break;
            case SDF_HEX_PRISM: d_prim = sdHexPrism(p, make_float2(prim.params[0], prim.params[1])); break;
            default: d_prim = length(p) - 1.0f; break;
        }
        d_prim *= fminf(prim.scale.x, fminf(prim.scale.y, prim.scale.z));
        if (prim.annular > 0.0f) d_prim = fabsf(d_prim) - prim.annular;
        if (prim.rounding > 0.0f) d_prim -= prim.rounding;

        if (d == 1e10f) { 
            d = d_prim; 
            color = prim.color;
            closestPrimID = i;
            closestLocalPos = p_preDisp;
        }
        else {
            float h = 0.0f;
            switch(prim.operation) {
                case SDF_UNION: 
                    if (d_prim < d) { 
                        d = d_prim; 
                        color = prim.color;
                        closestPrimID = i;
                        closestLocalPos = p_preDisp;
                    } 
                    break;
                case SDF_SUBTRACT: 
                    {
                        float d_old = d;
                        d = opSubtract(d, d_prim); 
                        if (d > d_old) { 
                            color = prim.color;
                            closestPrimID = i;
                            closestLocalPos = p_preDisp;
                        }
                    }
                    break;
                case SDF_INTERSECT: 
                    if (d_prim > d) { 
                        d = d_prim; 
                        color = prim.color;
                        closestPrimID = i;
                        closestLocalPos = p_preDisp;
                    } 
                    break;
                case SDF_UNION_BLEND:
                    h = clamp(0.5f + 0.5f * (d - d_prim) / prim.blendFactor, 0.0f, 1.0f);
                    d = mix(d, d_prim, h) - prim.blendFactor * h * (1.0f - h);
                    color = mix(color, prim.color, h);
                    if (d_prim < d || h > 0.5f) {
                        closestPrimID = i;
                        closestLocalPos = p_preDisp;
                    }
                    break;
                case SDF_SUBTRACT_BLEND:
                    h = clamp(0.5f - 0.5f * (d + d_prim) / prim.blendFactor, 0.0f, 1.0f);
                    d = mix(d, -d_prim, h) + prim.blendFactor * h * (1.0f - h);
                    color = mix(color, prim.color, h);
                    if (h > 0.5f) {
                        closestPrimID = i;
                        closestLocalPos = p_preDisp;
                    }
                    break;
                case SDF_INTERSECT_BLEND:
                    h = clamp(0.5f - 0.5f * (d - d_prim) / prim.blendFactor, 0.0f, 1.0f);
                    d = mix(d, d_prim, h) + prim.blendFactor * h * (1.0f - h);
                    color = mix(color, prim.color, h);
                    if (d_prim > d || h > 0.5f) {
                        closestPrimID = i;
                        closestLocalPos = p_preDisp;
                    }
                    break;
            }
        }
    }
    outDist = d;
    outColor = color;
    outPrimitiveID = closestPrimID;
    outLocalPos = closestLocalPos;
}

// --------------------------------------------------------------------------
// Normal Computation Helper (must be after map() is defined)
// --------------------------------------------------------------------------

__device__ inline float3 computeNormal(float3 p, const SDFGrid& grid, float time) {
    const float h = 0.001f;  // Small epsilon for finite differences
    
    float dist_c, dist_x, dist_y, dist_z;
    float3 color_temp;
    int primID_temp;
    float3 localPos_temp;
    
    // Sample SDF at center and offset positions
    map(p, grid, time, dist_c, color_temp, primID_temp, localPos_temp);
    map(make_float3(p.x + h, p.y, p.z), grid, time, dist_x, color_temp, primID_temp, localPos_temp);
    map(make_float3(p.x, p.y + h, p.z), grid, time, dist_y, color_temp, primID_temp, localPos_temp);
    map(make_float3(p.x, p.y, p.z + h), grid, time, dist_z, color_temp, primID_temp, localPos_temp);
    
    // Compute gradient
    float3 grad = make_float3(
        dist_x - dist_c,
        dist_y - dist_c,
        dist_z - dist_c
    );
    
    // Normalize
    float len = sqrtf(grad.x * grad.x + grad.y * grad.y + grad.z * grad.z);
    if (len > 1e-6f) {
        grad = make_float3(grad.x / len, grad.y / len, grad.z / len);
    } else {
        grad = make_float3(0.0f, 1.0f, 0.0f);  // Default up vector
    }
    
    return grad;
}

// --------------------------------------------------------------------------
// Scout Kernel
// --------------------------------------------------------------------------

__global__ void scoutActiveBlocks(SDFGrid grid, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalBlocks = grid.blocksDim.x * grid.blocksDim.y * grid.blocksDim.z;
    
    if (idx >= totalBlocks) return;
    
    // Block Coord
    int bx = idx % grid.blocksDim.x;
    int temp = idx / grid.blocksDim.x;
    int by = temp % grid.blocksDim.y;
    int bz = temp / grid.blocksDim.y;
    
    // Center of the block
    // The block covers range [b * size, (b+1)*size] in voxel coords
    // Center voxel coord ~ b*size + size/2
    float3 centerVoxel = make_float3(
        (bx * SDF_BLOCK_SIZE + SDF_BLOCK_SIZE * 0.5f) * grid.cellSize + grid.origin.x,
        (by * SDF_BLOCK_SIZE + SDF_BLOCK_SIZE * 0.5f) * grid.cellSize + grid.origin.y,
        (bz * SDF_BLOCK_SIZE + SDF_BLOCK_SIZE * 0.5f) * grid.cellSize + grid.origin.z
    );
    
    // Bounding Radius
    // Radius of cube = half_diag = (size/2) * sqrt(3)
    // Add buffer for interpolation (1-2 voxels)
    float r = (SDF_BLOCK_SIZE * 0.5f * 1.73205f + 2.0f) * grid.cellSize;
    
    float d;
    float3 c;
    int primID;       // Unused here
    float3 localPos;  // Unused here
    
    map(centerVoxel, grid, time, d, c, primID, localPos);
    
    if (fabsf(d) <= r) {
        int id = atomicAdd(grid.d_activeBlockCount, 1);
        if (id < grid.maxBlocks) {
            grid.d_activeBlocks[id] = idx;
        }
    }
}

// --------------------------------------------------------------------------
// Active Block Processing
// --------------------------------------------------------------------------

__device__ int4 getGlobalCoordsFromBlock(int blockIdx, int tid, const SDFGrid& grid) {
    // blockIdx is the ID in d_activeBlocks list, not the linear index
    // Wait, no, we pass the linear block ID to this function usually?
    // Let's say blockID is the linear index (bx + by*w + ...)
    
    int bx = blockIdx % grid.blocksDim.x;
    int temp = blockIdx / grid.blocksDim.x;
    int by = temp % grid.blocksDim.y;
    int bz = temp / grid.blocksDim.y;
    
    // Local ID in block
    int lz = tid / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
    int trem = tid % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
    int ly = trem / SDF_BLOCK_SIZE;
    int lx = trem % SDF_BLOCK_SIZE;
    
    return make_int4(bx * SDF_BLOCK_SIZE + lx, by * SDF_BLOCK_SIZE + ly, bz * SDF_BLOCK_SIZE + lz, 0);
}

__global__ void countActiveBlockTriangles(SDFGrid grid, float time) {
    int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;
    
    int blockIndex = grid.d_activeBlocks[activeBlockId];
    int tid = threadIdx.x; // 0 to 511
    
    int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    
    // Check boundaries
    if (outOfBounds(grid, xyz)) {
        grid.d_packetVertexCounts[activeBlockId * SDF_BLOCK_SIZE_CUBED + tid] = 0;
        return;
    }

    // Evaluate 8 corners
    // Optimization: We could share memory or use warp shuffle, but for now brute force 8 eval
    // Since we don't have a global SDF grid anymore.
    
    float cubeVal[8];
    int code = 0;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int dx = marchingCubeCorners[i][0];
        int dy = marchingCubeCorners[i][1];
        int dz = marchingCubeCorners[i][2];
        
        float3 p = getLocation(grid, make_int4(xyz.x + dx, xyz.y + dy, xyz.z + dz, 0));
        float d; float3 c;
        int primID; float3 localPos;
        map(p, grid, time, d, c, primID, localPos);
        
        cubeVal[i] = d;
        // Treat exact-equality as "inside" to reduce corner-collapses.
        if (d > isoThreshold) code |= (1 << i);
    }
    
    int firstIn = firstMarchingCubesId[code];
    int num = firstMarchingCubesId[code + 1] - firstIn;

    // Count only NON-degenerate triangles so offsets match what we actually write later.
    unsigned int validCount = 0;
    for (int i = 0; i < num; i += 3) {
        float3 pTri[3];
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int eid = marchingCubesIds[firstIn + i + j];
            pTri[j] = interpEdgePoint(grid, xyz, cubeVal, eid);
        }
        if (!isDegenerateTri(pTri, grid.cellSize)) {
            validCount += 3;
        }
    }

    grid.d_packetVertexCounts[activeBlockId * SDF_BLOCK_SIZE_CUBED + tid] = validCount;
}

__global__ void generateActiveBlockTriangles(SDFGrid grid, float time) {
    int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;
    
    int blockIndex = grid.d_activeBlocks[activeBlockId];
    int tid = threadIdx.x;
    
    unsigned int first = grid.d_packetVertexOffsets[activeBlockId * SDF_BLOCK_SIZE_CUBED + tid];
    
    int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    if (outOfBounds(grid, xyz)) return;
    
    float cubeVal[8];
    int code = 0;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int dx = marchingCubeCorners[i][0];
        int dy = marchingCubeCorners[i][1];
        int dz = marchingCubeCorners[i][2];
        
        float3 p = getLocation(grid, make_int4(xyz.x + dx, xyz.y + dy, xyz.z + dz, 0));
        float d; float3 c;
        int primID; float3 localPos;
        map(p, grid, time, d, c, primID, localPos);
        
        cubeVal[i] = d;
        // Treat exact-equality as "inside" to reduce corner-collapses.
        if (d > isoThreshold) code |= (1 << i);
    }
    
    int firstIn = firstMarchingCubesId[code];
    int num = firstMarchingCubesId[code + 1] - firstIn;
    
    // Generate Triangles
    unsigned int write = first;
    for (int i = 0; i < num; i += 3) {
        float3 pTri[3];
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int eid = marchingCubesIds[firstIn + i + j];
            pTri[j] = interpEdgePoint(grid, xyz, cubeVal, eid);
        }

        if (isDegenerateTri(pTri, grid.cellSize)) {
            continue; // counts/offsets already excluded these in countActiveBlockTriangles
        }

        if (write + 2 >= grid.maxVertices) break;
/*        
        // STEP 1: Evaluate all 3 vertices to get their individual data
        float dist_v[3];
        float3 color_v[3];
        int primitiveID_v[3];
        float3 localPos_v[3];
        
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            map(pTri[j], grid, time, dist_v[j], color_v[j], primitiveID_v[j], localPos_v[j]);
        }
   
*/
        // STEP 2: Choose DOMINANT primitive for the entire triangle
        // Use the primitive at the triangle centroid for consistency
        float3 triCenter = make_float3(
            (pTri[0].x + pTri[1].x + pTri[2].x) / 3.0f,
            (pTri[0].y + pTri[1].y + pTri[2].y) / 3.0f,
            (pTri[0].z + pTri[1].z + pTri[2].z) / 3.0f
        );
        
        float dist_center;
        float3 color_center;
        int dominantPrimID;
        float3 localPos_center;
        map(triCenter, grid, time, dist_center, color_center, dominantPrimID, localPos_center);
        
        // STEP 3: Transform all 3 vertex positions to dominant primitive's local space
        float3 localPositions[3];
        if (dominantPrimID >= 0 && dominantPrimID < grid.numPrimitives) {
            SDFPrimitive prim = grid.d_primitives[dominantPrimID];
            
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                float3 p_local = pTri[j] - prim.position;
                p_local = invRotateVector(p_local, prim.rotation);
                p_local = make_float3(p_local.x / prim.scale.x, p_local.y / prim.scale.y, p_local.z / prim.scale.z);
                localPositions[j] = p_local;
            }
        }
        
        // STEP 4: Compute triangle-aware UVs with seam handling
        float2 triangleUVs[3];
        if (grid.d_uvCoords && dominantPrimID >= 0 && dominantPrimID < grid.numPrimitives) {
            SDFPrimitive prim = grid.d_primitives[dominantPrimID];
            computeTriangleUVs(localPositions, prim, time, triangleUVs);
        } else {
            // Fallback
            triangleUVs[0] = triangleUVs[1] = triangleUVs[2] = make_float2(0.0f, 0.0f);
        }
        
        // STEP 5: Write all 3 vertices
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float3 p = pTri[j];

            // Use per-vertex color (preserves smooth color transitions)
            grid.d_vertices[write + j] = make_float4(p.x, p.y, p.z, 1.0f);
            if (grid.d_vertexColors) {
                grid.d_vertexColors[write + j] = make_float4(color_center.x, color_center.y, color_center.z, 1.0f);
            }
            
            // COMPUTE AND WRITE NORMALS (per-vertex for smooth shading)
            if (grid.d_normals) {
                float3 normal = computeNormal(p, grid, time);
                grid.d_normals[write + j] = make_float4(normal.x, normal.y, normal.z, 0.0f);
            }
            
            // WRITE PRIMITIVE ID - ALL vertices use DOMINANT primitive
            if (grid.d_primitiveIDs) {
                // Store the primitive index (not the texture layer).
                // The renderer can derive texture layer from the primitive UBO (see packPrimitives()).
                *((float*)&grid.d_primitiveIDs[write + j]) = (float)dominantPrimID;
            }
            
            // WRITE UVS - Use pre-computed seam-aware triangle UVs
            if (grid.d_uvCoords) {
                grid.d_uvCoords[write + j] = triangleUVs[j];
            }
        }

        if (grid.d_indices) {
            grid.d_indices[write] = write;
            grid.d_indices[write + 1] = write + 1;
            grid.d_indices[write + 2] = write + 2;
        }

        write += 3;
    }
}

// ==========================================================================
// Dual Contouring (Option A: triangle soup)
// - per-quad dominant primitive (sampled at quad center)
// - normals via computeNormal() (SDF gradient, consistent w/ displacements)
// - requires block->activeBlockId map to stitch across blocks
// ==========================================================================

__device__ __forceinline__ int linearBlockIndex(const int3& b, const int3& blocksDim) {
    return b.x + b.y * blocksDim.x + b.z * blocksDim.x * blocksDim.y;
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__device__ __forceinline__ float3 clamp3(float3 p, float3 lo, float3 hi) {
    return make_float3(
        clamp(p.x, lo.x, hi.x),
        clamp(p.y, lo.y, hi.y),
        clamp(p.z, lo.z, hi.z)
    );
}

__device__ __forceinline__ float3 decodeCellVertexWorld(
    const ushort4 q,
    const float3 cellMin,
    const float cellSize
) {
    const float inv = 1.0f / 65535.0f;
    return make_float3(
        cellMin.x + (float)q.x * inv * cellSize,
        cellMin.y + (float)q.y * inv * cellSize,
        cellMin.z + (float)q.z * inv * cellSize
    );
}

__device__ __forceinline__ float3 decodeCellNormal(const short4 q) {
    const float inv = 1.0f / 32767.0f;
    float3 n = make_float3((float)q.x * inv, (float)q.y * inv, (float)q.z * inv);
    const float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (len > 1e-6f) n = make_float3(n.x/len, n.y/len, n.z/len);
    else n = make_float3(0.0f, 1.0f, 0.0f);
    return n;
}

__device__ __forceinline__ short4 encodeCellNormal(const float3 nIn) {
    float3 n = nIn;
    const float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (len > 1e-6f) n = make_float3(n.x/len, n.y/len, n.z/len);
    else n = make_float3(0.0f, 1.0f, 0.0f);

    short4 out;
    out.x = (short)clampi((int)lrintf(clamp(n.x, -1.0f, 1.0f) * 32767.0f), -32767, 32767);
    out.y = (short)clampi((int)lrintf(clamp(n.y, -1.0f, 1.0f) * 32767.0f), -32767, 32767);
    out.z = (short)clampi((int)lrintf(clamp(n.z, -1.0f, 1.0f) * 32767.0f), -32767, 32767);
    out.w = 0;
    return out;
}

__device__ __forceinline__ ushort4 encodeCellVertexLocal(const float3 pWorld, const float3 cellMin, float cellSize) {
    float3 t = make_float3(
        (pWorld.x - cellMin.x) / cellSize,
        (pWorld.y - cellMin.y) / cellSize,
        (pWorld.z - cellMin.z) / cellSize
    );
    t = clamp3(t, make_float3(0, 0, 0), make_float3(1, 1, 1));
    ushort4 out;
    out.x = (unsigned short)clampi((int)lrintf(t.x * 65535.0f), 0, 65535);
    out.y = (unsigned short)clampi((int)lrintf(t.y * 65535.0f), 0, 65535);
    out.z = (unsigned short)clampi((int)lrintf(t.z * 65535.0f), 0, 65535);
    out.w = 0;
    return out;
}

// Solve 3x3 linear system A x = b using Cramer's rule. Returns false if singular.
__device__ __forceinline__ bool solve3x3(const float A[9], const float b[3], float x[3]) {
    const float a00 = A[0], a01 = A[1], a02 = A[2];
    const float a10 = A[3], a11 = A[4], a12 = A[5];
    const float a20 = A[6], a21 = A[7], a22 = A[8];

    const float det =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);

    if (fabsf(det) < 1e-12f) return false;
    const float invDet = 1.0f / det;

    const float bx = b[0], by = b[1], bz = b[2];
    const float detX =
        bx * (a11 * a22 - a12 * a21) -
        a01 * (by * a22 - a12 * bz) +
        a02 * (by * a21 - a11 * bz);

    const float detY =
        a00 * (by * a22 - a12 * bz) -
        bx * (a10 * a22 - a12 * a20) +
        a02 * (a10 * bz - by * a20);

    const float detZ =
        a00 * (a11 * bz - by * a21) -
        a01 * (a10 * bz - by * a20) +
        bx * (a10 * a21 - a11 * a20);

    x[0] = detX * invDet;
    x[1] = detY * invDet;
    x[2] = detZ * invDet;
    return true;
}

// Minimize sum_i (n_i · (x - p_i))^2  =>  (Σ n n^T) x = Σ n (n·p)
__device__ __forceinline__ bool solveQEF(const float3* pts, const float3* ns, int m, float3& outX) {
    if (m <= 0) return false;

    float M[9] = {0};
    float rhs[3] = {0, 0, 0};

    for (int i = 0; i < m; ++i) {
        float3 n = ns[i];
        const float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len < 1e-6f) continue;
        n = make_float3(n.x / len, n.y / len, n.z / len);
        const float d = n.x * pts[i].x + n.y * pts[i].y + n.z * pts[i].z;

        // M += n n^T
        M[0] += n.x * n.x; M[1] += n.x * n.y; M[2] += n.x * n.z;
        M[3] += n.y * n.x; M[4] += n.y * n.y; M[5] += n.y * n.z;
        M[6] += n.z * n.x; M[7] += n.z * n.y; M[8] += n.z * n.z;

        // rhs += n * d
        rhs[0] += n.x * d;
        rhs[1] += n.y * d;
        rhs[2] += n.z * d;
    }

    // Regularize to stabilize corners/edges and blended SDF regions
    const float lambda = 1e-5f;
    M[0] += lambda; M[4] += lambda; M[8] += lambda;

    float x[3];
    if (!solve3x3(M, rhs, x)) return false;
    outX = make_float3(x[0], x[1], x[2]);
    return true;
}

// Regularized QEF with position bias: minimize Σ(n·(x-p))^2 + α||x-c||^2
// => (M + αI) x = rhs + α c
__device__ __forceinline__ bool solveQEFRegularized(
    const float3* pts,
    const float3* ns,
    int m,
    float alpha,
    float3 c,
    float3& outX
) {
    if (m <= 0) return false;

    float M[9] = {0};
    float rhs[3] = {0, 0, 0};

    for (int i = 0; i < m; ++i) {
        float3 n = ns[i];
        const float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len < 1e-6f) continue;
        n = make_float3(n.x / len, n.y / len, n.z / len);
        const float d = n.x * pts[i].x + n.y * pts[i].y + n.z * pts[i].z;

        M[0] += n.x * n.x; M[1] += n.x * n.y; M[2] += n.x * n.z;
        M[3] += n.y * n.x; M[4] += n.y * n.y; M[5] += n.y * n.z;
        M[6] += n.z * n.x; M[7] += n.z * n.y; M[8] += n.z * n.z;

        rhs[0] += n.x * d;
        rhs[1] += n.y * d;
        rhs[2] += n.z * d;
    }

    // Bias toward centroid / cell center to stabilize near-planar regions
    const float a = fmaxf(alpha, 0.0f);
    M[0] += a; M[4] += a; M[8] += a;
    rhs[0] += a * c.x;
    rhs[1] += a * c.y;
    rhs[2] += a * c.z;

    // Mild extra regularization (keeps Cramer's stable)
    const float lambda = 1e-6f;
    M[0] += lambda; M[4] += lambda; M[8] += lambda;

    float x[3];
    if (!solve3x3(M, rhs, x)) return false;
    outX = make_float3(x[0], x[1], x[2]);
    return true;
}

__global__ void buildBlockToActiveMap(SDFGrid grid) {
    const int activeBlockId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (activeBlockId >= *grid.d_activeBlockCount) return;
    const int blockIndex = grid.d_activeBlocks[activeBlockId];
    if (blockIndex >= 0 && blockIndex < grid.maxBlocks) {
        grid.d_blockToActiveId[blockIndex] = activeBlockId;
    }
}

__global__ void dcMarkCells(SDFGrid grid, float time) {
    const int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;

    const int blockIndex = grid.d_activeBlocks[activeBlockId];
    const int tid = (int)threadIdx.x;
    const int packetIdx = activeBlockId * SDF_BLOCK_SIZE_CUBED + tid;

    const int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    if (outOfBounds(grid, xyz)) {
        grid.d_packetVertexCounts[packetIdx] = 0;
        grid.d_dcCornerMasks[packetIdx] = 0;
        return;
    }

    unsigned char mask = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int dx = marchingCubeCorners[i][0];
        const int dy = marchingCubeCorners[i][1];
        const int dz = marchingCubeCorners[i][2];
        const float3 p = getLocation(grid, make_int4(xyz.x + dx, xyz.y + dy, xyz.z + dz, 0));
        float d; float3 c; int primID; float3 localPos;
        map(p, grid, time, d, c, primID, localPos);
        if (d > isoThreshold) mask |= (1u << i);
    }

    grid.d_dcCornerMasks[packetIdx] = mask;

    // Uniform mask => no surface in this cell
    const bool allIn = (mask == 0xFF);
    const bool allOut = (mask == 0x00);
    grid.d_packetVertexCounts[packetIdx] = (allIn || allOut) ? 0u : 1u;
}

__global__ void dcSolveCellVertices(SDFGrid grid, float time, unsigned int maxCellVertices, float qefBlend) {
    const int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;

    const int blockIndex = grid.d_activeBlocks[activeBlockId];
    const int tid = (int)threadIdx.x;
    const int packetIdx = activeBlockId * SDF_BLOCK_SIZE_CUBED + tid;

    if (grid.d_packetVertexCounts[packetIdx] == 0) return; // no vertex

    const int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    if (outOfBounds(grid, xyz)) return;

    const unsigned int vIdx = grid.d_dcCellVertexOffsets[packetIdx];
    if (vIdx >= maxCellVertices) return;

    // Evaluate corners (values) again (keeps dcMarkCells simple; can optimize later)
    float cubeVal[8];
    unsigned char mask = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int dx = marchingCubeCorners[i][0];
        const int dy = marchingCubeCorners[i][1];
        const int dz = marchingCubeCorners[i][2];
        const float3 p = getLocation(grid, make_int4(xyz.x + dx, xyz.y + dy, xyz.z + dz, 0));
        float d; float3 c; int primID; float3 localPos;
        map(p, grid, time, d, c, primID, localPos);
        cubeVal[i] = d;
        if (d > isoThreshold) mask |= (1u << i);
    }

    float3 pts[12];
    float3 ns[12];
    int m = 0;

    // Hermite samples from sign-changing edges
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        const int c1 = edgeConnections[e][0];
        const int c2 = edgeConnections[e][1];
        const bool s1 = ((mask >> c1) & 1) != 0;
        const bool s2 = ((mask >> c2) & 1) != 0;
        if (s1 == s2) continue;

        const float t = safeEdgeT(cubeVal[c1], cubeVal[c2]);

        const int4 pos1I = make_int4(
            xyz.x + marchingCubeCorners[c1][0],
            xyz.y + marchingCubeCorners[c1][1],
            xyz.z + marchingCubeCorners[c1][2], 0);
        const int4 pos2I = make_int4(
            xyz.x + marchingCubeCorners[c2][0],
            xyz.y + marchingCubeCorners[c2][1],
            xyz.z + marchingCubeCorners[c2][2], 0);

        const float3 p1 = getLocation(grid, pos1I);
        const float3 p2 = getLocation(grid, pos2I);
        const float3 p = lerp3(p1, p2, t);
        const float3 n = computeNormal(p, grid, time);

        pts[m] = p;
        ns[m] = n;
        m++;
    }

    // Fallback: if something went wrong, place at cell center
    const float3 cellMin = getLocation(grid, xyz);
    const float3 cellMax = make_float3(cellMin.x + grid.cellSize, cellMin.y + grid.cellSize, cellMin.z + grid.cellSize);
    float3 x = make_float3(cellMin.x + 0.5f * grid.cellSize, cellMin.y + 0.5f * grid.cellSize, cellMin.z + 0.5f * grid.cellSize);
    float3 pCentroid = x;
    if (m > 0) {
        float3 sum = make_float3(0, 0, 0);
        for (int i = 0; i < m; ++i) sum = sum + pts[i];
        const float invM = 1.0f / (float)m;
        pCentroid = sum * invM;
    }

    // Feature-preserving heuristic: cluster normals into up to 3 dominant directions and try plane intersections
    float3 clusterN[3] = {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)};
    float3 clusterP[3] = {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)};
    int clusterC[3] = {0,0,0};
    int k = 0;
    const float cosThresh = 0.92f; // ~23 degrees

    for (int i = 0; i < m; ++i) {
        float3 n = ns[i];
        const float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
        if (len < 1e-6f) continue;
        n = make_float3(n.x/len, n.y/len, n.z/len);

        int best = -1;
        float bestDot = -1.0f;
        for (int ci = 0; ci < k; ++ci) {
            float3 cn = clusterN[ci];
            const float clen = sqrtf(cn.x*cn.x + cn.y*cn.y + cn.z*cn.z);
            if (clen > 1e-6f) cn = make_float3(cn.x/clen, cn.y/clen, cn.z/clen);
            const float d = fabsf(cn.x*n.x + cn.y*n.y + cn.z*n.z);
            if (d > bestDot) { bestDot = d; best = ci; }
        }

        if (best >= 0 && bestDot > cosThresh) {
            clusterN[best] = clusterN[best] + n;
            clusterP[best] = clusterP[best] + pts[i];
            clusterC[best]++;
        } else if (k < 3) {
            clusterN[k] = n;
            clusterP[k] = pts[i];
            clusterC[k] = 1;
            k++;
        } else {
            // Assign to closest anyway
            clusterN[best] = clusterN[best] + n;
            clusterP[best] = clusterP[best] + pts[i];
            clusterC[best]++;
        }
    }

    bool solved = false;
    // Only apply "feature preserving" plane intersections when clusters are clearly separated.
    // Otherwise (smooth surfaces) fall back to standard QEF to avoid noisy/spiky silhouettes.
    if (k >= 2) {
        // Normalize cluster mean normals
        float3 n1 = clusterN[0];
        float3 n2 = clusterN[1];
        float3 n3 = clusterN[2];
        const float l1 = sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z);
        const float l2 = sqrtf(n2.x*n2.x + n2.y*n2.y + n2.z*n2.z);
        const float l3 = sqrtf(n3.x*n3.x + n3.y*n3.y + n3.z*n3.z);
        if (l1 > 1e-6f && l2 > 1e-6f) {
            n1 = make_float3(n1.x/l1, n1.y/l1, n1.z/l1);
            n2 = make_float3(n2.x/l2, n2.y/l2, n2.z/l2);
            if (l3 > 1e-6f) n3 = make_float3(n3.x/l3, n3.y/l3, n3.z/l3);

            const float d12 = fabsf(dot(n1, n2));
            const float d13 = (k >= 3 && l3 > 1e-6f) ? fabsf(dot(n1, n3)) : 1.0f;
            const float d23 = (k >= 3 && l3 > 1e-6f) ? fabsf(dot(n2, n3)) : 1.0f;

            // Require enough support in each cluster before treating it as a distinct feature plane.
            const bool c01 = (clusterC[0] >= 2) && (clusterC[1] >= 2);
            const bool c012 = c01 && (k >= 3) && (clusterC[2] >= 2) && (l3 > 1e-6f);

            // Edge/corner thresholds (tuned to be conservative).
            // Smaller dot => more orthogonal => sharper feature.
            const float edgeDotThresh = 0.75f;   // ~41 degrees
            const float cornerDotThresh = 0.80f; // ~37 degrees

            bool doEdge = (k == 2) && c01 && (d12 < edgeDotThresh);
            bool doCorner = (k >= 3) && c012 && (d12 < cornerDotThresh) && (d13 < cornerDotThresh) && (d23 < cornerDotThresh);

            if (doEdge || doCorner) {
                const float invC0 = 1.0f / (float)((clusterC[0] > 0) ? clusterC[0] : 1);
                const float invC1 = 1.0f / (float)((clusterC[1] > 0) ? clusterC[1] : 1);
                float3 p1 = clusterP[0] * invC0;
                float3 p2 = clusterP[1] * invC1;

                float A[9];
                float b[3];

                if (doEdge) {
                    float3 nEdge = cross3(n1, n2);
                    const float le = sqrtf(nEdge.x*nEdge.x + nEdge.y*nEdge.y + nEdge.z*nEdge.z);
                    if (le > 1e-6f) {
                        nEdge = make_float3(nEdge.x/le, nEdge.y/le, nEdge.z/le);
                        // Anchor the 3rd plane with average point to select a stable point on the edge line.
                        float3 pAvg = make_float3(0,0,0);
                        for (int i = 0; i < m; ++i) pAvg = pAvg + pts[i];
                        const float invM = 1.0f / (float)((m > 0) ? m : 1);
                        pAvg = pAvg * invM;

                        A[0]=n1.x;   A[1]=n1.y;   A[2]=n1.z;
                        A[3]=n2.x;   A[4]=n2.y;   A[5]=n2.z;
                        A[6]=nEdge.x;A[7]=nEdge.y;A[8]=nEdge.z;
                        b[0]=dot(n1, p1);
                        b[1]=dot(n2, p2);
                        b[2]=dot(nEdge, pAvg);

                        float sol[3];
                        if (solve3x3(A, b, sol)) {
                            x = make_float3(sol[0], sol[1], sol[2]);
                            solved = true;
                        }
                    }
                } else if (doCorner) {
                    const float invC2 = 1.0f / (float)((clusterC[2] > 0) ? clusterC[2] : 1);
                    float3 p3 = clusterP[2] * invC2;

                    A[0]=n1.x; A[1]=n1.y; A[2]=n1.z;
                    A[3]=n2.x; A[4]=n2.y; A[5]=n2.z;
                    A[6]=n3.x; A[7]=n3.y; A[8]=n3.z;
                    b[0]=dot(n1, p1);
                    b[1]=dot(n2, p2);
                    b[2]=dot(n3, p3);

                    float sol[3];
                    if (solve3x3(A, b, sol)) {
                        x = make_float3(sol[0], sol[1], sol[2]);
                        solved = true;
                    }
                }
            }
        }
    }

    if (!solved) {
        // Regularized QEF (stable on smooth surfaces)
        // alpha tuned conservatively: strong enough to prevent wild solutions,
        // weak enough to preserve detail where constraints are well-conditioned.
        const float alpha = 0.05f;
        if (!solveQEFRegularized(pts, ns, m, alpha, pCentroid, x)) {
            (void)solveQEF(pts, ns, m, x);
        }
    }

    // Soft clamp: allow a tiny margin to reduce discontinuities from hard clipping,
    // then clamp to an expanded box and finally to the true cell bounds.
    const float margin = 0.10f * grid.cellSize;
    x = clamp3(x,
        make_float3(cellMin.x - margin, cellMin.y - margin, cellMin.z - margin),
        make_float3(cellMax.x + margin, cellMax.y + margin, cellMax.z + margin)
    );
    x = clamp3(x, cellMin, cellMax);
    
    // QEF blend: 0 = blocky (cell center), 1 = full QEF solve
    // cellCenter is the center of the voxel cell
    const float3 cellCenter = make_float3(
        cellMin.x + 0.5f * grid.cellSize,
        cellMin.y + 0.5f * grid.cellSize,
        cellMin.z + 0.5f * grid.cellSize
    );
    x = lerp3(cellCenter, x, qefBlend);
    
    grid.d_dcCellVertices[vIdx] = encodeCellVertexLocal(x, cellMin, grid.cellSize);

    // Store a smooth per-cell-vertex normal:
    // Prefer averaged Hermite normals (stable across quads), fallback to SDF gradient at x.
    if (grid.d_dcCellNormals) {
        float3 nSum = make_float3(0, 0, 0);
        for (int i = 0; i < m; ++i) nSum = nSum + ns[i];
        const float len = sqrtf(nSum.x*nSum.x + nSum.y*nSum.y + nSum.z*nSum.z);
        float3 nOut = (len > 1e-6f) ? make_float3(nSum.x/len, nSum.y/len, nSum.z/len) : computeNormal(x, grid, time);
        // Ensure consistent orientation (outward) by matching the SDF gradient at the solved point.
        {
            const float3 nRef = computeNormal(x, grid, time);
            if (dot(nOut, nRef) < 0.0f) {
                nOut = make_float3(-nOut.x, -nOut.y, -nOut.z);
            }
        }
        grid.d_dcCellNormals[vIdx] = encodeCellNormal(nOut);
    }
}

// Helper: lookup packed-cell index for a global cell coordinate. Returns false if block not active.
__device__ __forceinline__ bool lookupPackedCell(
    const SDFGrid& grid,
    int cx, int cy, int cz,
    int& outPacketIdx
) {
    if (cx < 0 || cy < 0 || cz < 0) return false;
    if (cx >= (int)grid.width - 1 || cy >= (int)grid.height - 1 || cz >= (int)grid.depth - 1) return false;

    const int bx = cx / SDF_BLOCK_SIZE;
    const int by = cy / SDF_BLOCK_SIZE;
    const int bz = cz / SDF_BLOCK_SIZE;
    if (bx < 0 || by < 0 || bz < 0 || bx >= grid.blocksDim.x || by >= grid.blocksDim.y || bz >= grid.blocksDim.z) return false;

    const int blockIndex = bx + by * grid.blocksDim.x + bz * grid.blocksDim.x * grid.blocksDim.y;
    const int activeBlockId = grid.d_blockToActiveId[blockIndex];
    if (activeBlockId < 0) return false;

    const int lx = cx - bx * SDF_BLOCK_SIZE;
    const int ly = cy - by * SDF_BLOCK_SIZE;
    const int lz = cz - bz * SDF_BLOCK_SIZE;
    const int tid = lx + ly * SDF_BLOCK_SIZE + lz * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);

    outPacketIdx = activeBlockId * SDF_BLOCK_SIZE_CUBED + tid;
    return true;
}

__global__ void dcSmoothCellNormals(SDFGrid grid, float cosAngleThreshold) {
    const int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;

    const int blockIndex = grid.d_activeBlocks[activeBlockId];
    const int tid = (int)threadIdx.x;
    const int packetIdx = activeBlockId * SDF_BLOCK_SIZE_CUBED + tid;

    if (!grid.d_dcCellNormals || !grid.d_dcCellNormalsTmp) return;

    const int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    if (outOfBounds(grid, xyz)) return;

    const unsigned char mask = grid.d_dcCornerMasks[packetIdx];
    if (mask == 0x00 || mask == 0xFF) return; // not a surface cell

    const unsigned int vIdx = grid.d_dcCellVertexOffsets[packetIdx];
    if (vIdx >= grid.maxVertices) return;

    const float3 nSelf = decodeCellNormal(grid.d_dcCellNormals[vIdx]);
    float3 nSum = nSelf;
    int count = 1;

    // 6-neighborhood smoothing (face-adjacent only), adaptive by angle threshold.
    // This avoids smearing across diagonals which can bridge unrelated surface sheets near CSG seams.
    const int3 offs[6] = {
        make_int3( 1, 0, 0), make_int3(-1, 0, 0),
        make_int3( 0, 1, 0), make_int3( 0,-1, 0),
        make_int3( 0, 0, 1), make_int3( 0, 0,-1)
    };

    for (int i = 0; i < 6; ++i) {
        int pN;
        if (!lookupPackedCell(grid, xyz.x + offs[i].x, xyz.y + offs[i].y, xyz.z + offs[i].z, pN)) continue;
        const unsigned char mN = grid.d_dcCornerMasks[pN];
        if (mN == 0x00 || mN == 0xFF) continue;

        const unsigned int vN = grid.d_dcCellVertexOffsets[pN];
        if (vN >= grid.maxVertices) continue;
        const float3 nNbr = decodeCellNormal(grid.d_dcCellNormals[vN]);

        const float d = dot(nNbr, nSelf);
        if (d >= cosAngleThreshold) {
            // Weight neighbors by alignment so "almost threshold" has small influence.
            const float w = (d - cosAngleThreshold) / fmaxf(1e-6f, (1.0f - cosAngleThreshold));
            nSum = nSum + (w * nNbr);
            count++;
        }
    }

    const float len = sqrtf(nSum.x*nSum.x + nSum.y*nSum.y + nSum.z*nSum.z);
    float3 nOut = (len > 1e-6f) ? make_float3(nSum.x/len, nSum.y/len, nSum.z/len) : nSelf;
    grid.d_dcCellNormalsTmp[vIdx] = encodeCellNormal(nOut);
}

__global__ void dcCountQuads(SDFGrid grid) {
    const int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;

    const int blockIndex = grid.d_activeBlocks[activeBlockId];
    const int tid = (int)threadIdx.x;
    const int packetIdx = activeBlockId * SDF_BLOCK_SIZE_CUBED + tid;

    const int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    if (outOfBounds(grid, xyz)) {
        grid.d_packetVertexCounts[packetIdx] = 0;
        return;
    }

    const unsigned char mask = grid.d_dcCornerMasks[packetIdx];
    unsigned int count = 0;

    // Only generate quads for edges emanating from the cell's min corner (corner 0):
    // X edge: corners 0-1, Y edge: 0-3, Z edge: 0-4. This avoids duplicate edge ownership.
    const bool s0 = (mask & 1) != 0;

    // Helper: surface cell if mask is neither all-in nor all-out
    auto isSurfaceCell = [&](int packet) {
        const unsigned char m = grid.d_dcCornerMasks[packet];
        return (m != 0x00) && (m != 0xFF);
    };

    // X edge (requires neighbor cells in -Y and -Z)
    if (((mask >> 1) & 1) != (unsigned char)s0) {
        if (xyz.y > 0 && xyz.z > 0) {
            int pA, pB, pC, pD;
            if (lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z,     pA) &&
                lookupPackedCell(grid, xyz.x,     xyz.y - 1, xyz.z,     pB) &&
                lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z - 1, pC) &&
                lookupPackedCell(grid, xyz.x,     xyz.y - 1, xyz.z - 1, pD) &&
                isSurfaceCell(pA) && isSurfaceCell(pB) && isSurfaceCell(pC) && isSurfaceCell(pD)) {
                count += 6;
            }
        }
    }

    // Y edge (requires neighbor cells in -X and -Z)
    if (((mask >> 3) & 1) != (unsigned char)s0) {
        if (xyz.x > 0 && xyz.z > 0) {
            int pA, pB, pC, pD;
            if (lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z,     pA) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y,     xyz.z,     pB) &&
                lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z - 1, pC) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y,     xyz.z - 1, pD) &&
                isSurfaceCell(pA) && isSurfaceCell(pB) && isSurfaceCell(pC) && isSurfaceCell(pD)) {
                count += 6;
            }
        }
    }

    // Z edge (requires neighbor cells in -X and -Y)
    if (((mask >> 4) & 1) != (unsigned char)s0) {
        if (xyz.x > 0 && xyz.y > 0) {
            int pA, pB, pC, pD;
            if (lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z,     pA) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y,     xyz.z,     pB) &&
                lookupPackedCell(grid, xyz.x,     xyz.y - 1, xyz.z,     pC) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y - 1, xyz.z,     pD) &&
                isSurfaceCell(pA) && isSurfaceCell(pB) && isSurfaceCell(pC) && isSurfaceCell(pD)) {
                count += 6;
            }
        }
    }

    grid.d_packetVertexCounts[packetIdx] = count;
}

// Seam-aware UV for a quad: compute as two triangles (0,1,2) and (0,2,3) but keep consistent UVs.
__device__ void computeQuadUVs(
    const float3 localPos[4],
    const SDFPrimitive& prim,
    float time,
    float2 outUVs[4]
) {
    float3 tri0[3] = { localPos[0], localPos[1], localPos[2] };
    float3 tri1[3] = { localPos[0], localPos[2], localPos[3] };
    float2 uv0[3], uv1[3];
    computeTriangleUVs(tri0, prim, time, uv0);
    computeTriangleUVs(tri1, prim, time, uv1);

    // Use tri0 for verts 0,1,2; use tri1 for vert3; average shared verts for coherence.
    outUVs[0] = (uv0[0] + uv1[0]) * 0.5f;
    outUVs[1] = uv0[1];
    outUVs[2] = (uv0[2] + uv1[1]) * 0.5f;
    outUVs[3] = uv1[2];
}

__global__ void dcGenerateQuads(SDFGrid grid, float time) {
    const int activeBlockId = blockIdx.x;
    if (activeBlockId >= *grid.d_activeBlockCount) return;

    const int blockIndex = grid.d_activeBlocks[activeBlockId];
    const int tid = (int)threadIdx.x;
    const int packetIdx = activeBlockId * SDF_BLOCK_SIZE_CUBED + tid;

    const unsigned int first = grid.d_packetVertexOffsets[packetIdx];
    const unsigned int emitCount = grid.d_packetVertexCounts[packetIdx];
    if (emitCount == 0) return;

    const int4 xyz = getGlobalCoordsFromBlock(blockIndex, tid, grid);
    if (outOfBounds(grid, xyz)) return;

    const unsigned char mask = grid.d_dcCornerMasks[packetIdx];
    const bool s0 = (mask & 1) != 0;

    unsigned int write = first;

    // NOTE: For now, we will compute the four cell mins directly from the global coords, since those are cheap.
    // Each packetIdx corresponds to a specific global cell (xyz). For neighbor packet indices we reconstruct their global coords
    // by computing the cell coords from the packet's tid and active block. To avoid this overhead, a future optimization would store
    // global cell coords per packet or decode via precomputed block coords.

    auto packetToXYZ = [&](int packet, int4& outXYZ) {
        const int ab = packet / SDF_BLOCK_SIZE_CUBED;
        const int t = packet - ab * SDF_BLOCK_SIZE_CUBED;
        const int bIndex = grid.d_activeBlocks[ab];
        outXYZ = getGlobalCoordsFromBlock(bIndex, t, grid);
    };

    auto emitQuadFromPackets = [&](int p0, int p1, int p2, int p3) {
        // Must be surface cells (otherwise offsets/vertices are undefined for those packets)
        auto isSurface = [&](int packet) {
            const unsigned char m = grid.d_dcCornerMasks[packet];
            return (m != 0x00) && (m != 0xFF);
        };
        if (!isSurface(p0) || !isSurface(p1) || !isSurface(p2) || !isSurface(p3)) return;

        int4 aXYZ, bXYZ, cXYZ, dXYZ;
        packetToXYZ(p0, aXYZ);
        packetToXYZ(p1, bXYZ);
        packetToXYZ(p2, cXYZ);
        packetToXYZ(p3, dXYZ);

        const unsigned int i0 = grid.d_dcCellVertexOffsets[p0];
        const unsigned int i1 = grid.d_dcCellVertexOffsets[p1];
        const unsigned int i2 = grid.d_dcCellVertexOffsets[p2];
        const unsigned int i3 = grid.d_dcCellVertexOffsets[p3];
        if (i0 >= grid.maxVertices || i1 >= grid.maxVertices || i2 >= grid.maxVertices || i3 >= grid.maxVertices) return;

        const float3 pMin0 = getLocation(grid, aXYZ);
        const float3 pMin1 = getLocation(grid, bXYZ);
        const float3 pMin2 = getLocation(grid, cXYZ);
        const float3 pMin3 = getLocation(grid, dXYZ);

        float3 v0 = decodeCellVertexWorld(grid.d_dcCellVertices[i0], pMin0, grid.cellSize);
        float3 v1 = decodeCellVertexWorld(grid.d_dcCellVertices[i1], pMin1, grid.cellSize);
        float3 v2 = decodeCellVertexWorld(grid.d_dcCellVertices[i2], pMin2, grid.cellSize);
        float3 v3 = decodeCellVertexWorld(grid.d_dcCellVertices[i3], pMin3, grid.cellSize);

        // Choose dominant primitive per-quad at quad center (world)
        const float3 quadCenter = make_float3(
            0.25f * (v0.x + v1.x + v2.x + v3.x),
            0.25f * (v0.y + v1.y + v2.y + v3.y),
            0.25f * (v0.z + v1.z + v2.z + v3.z)
        );

        // Guard against bad decode (shouldn't happen, but prevents catastrophic stretched tris)
        if (!isFinite3(v0)) v0 = quadCenter;
        if (!isFinite3(v1)) v1 = quadCenter;
        if (!isFinite3(v2)) v2 = quadCenter;
        if (!isFinite3(v3)) v3 = quadCenter;

        // Enforce consistent winding against SDF gradient to avoid "spike" triangles from flipped quads.
        // If the quad is oriented opposite the local surface normal, flip (swap v1 <-> v3).
        bool didFlip = false;
        {
            const float3 nRef = computeNormal(quadCenter, grid, time);
            const float3 nTri = cross3(v1 - v0, v2 - v0);
            if (dot(nTri, nRef) < 0.0f) {
                const float3 tmp = v1; v1 = v3; v3 = tmp;
                didFlip = true;
            }
        }

        float dist_center;
        float3 color_center;
        int dominantPrimID;
        float3 localPos_center;
        map(quadCenter, grid, time, dist_center, color_center, dominantPrimID, localPos_center);

        // Transform quad vertices to dominant primitive local space for UVs
        float2 quadUVs[4] = {make_float2(0,0), make_float2(0,0), make_float2(0,0), make_float2(0,0)};
        if (grid.d_uvCoords && dominantPrimID >= 0 && dominantPrimID < (int)grid.numPrimitives) {
            const SDFPrimitive prim = grid.d_primitives[dominantPrimID];
            float3 localPos[4];
            const float3 vw[4] = {v0, v1, v2, v3};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float3 p_local = vw[i] - prim.position;
                p_local = invRotateVector(p_local, prim.rotation);
                p_local = make_float3(p_local.x / prim.scale.x, p_local.y / prim.scale.y, p_local.z / prim.scale.z);
                localPos[i] = p_local;
            }
            computeQuadUVs(localPos, prim, time, quadUVs);
        }

        // Fetch smooth per-cell-vertex normals (one per DC cell vertex), fallback to quadCenter gradient if not available.
        float3 n0 = make_float3(0.0f, 1.0f, 0.0f);
        float3 n1 = n0, n2 = n0, n3 = n0;
        if (grid.d_dcCellNormals) {
            // If we flipped winding by swapping v1/v3, we must also swap which cell-normal is paired with those vertices.
            const unsigned int ni1 = didFlip ? i3 : i1;
            const unsigned int ni3 = didFlip ? i1 : i3;
            n0 = decodeCellNormal(grid.d_dcCellNormals[i0]);
            n1 = decodeCellNormal(grid.d_dcCellNormals[ni1]);
            n2 = decodeCellNormal(grid.d_dcCellNormals[i2]);
            n3 = decodeCellNormal(grid.d_dcCellNormals[ni3]);
        } else if (grid.d_normals) {
            const float3 nRef = computeNormal(quadCenter, grid, time);
            n0 = n1 = n2 = n3 = nRef;
        }

        // Emit two triangles: (0,1,2) and (0,2,3)
        const float3 tri[6] = {v0, v1, v2, v0, v2, v3};
        const float2 triUV[6] = {quadUVs[0], quadUVs[1], quadUVs[2], quadUVs[0], quadUVs[2], quadUVs[3]};
        const float3 triN[6] = {n0, n1, n2, n0, n2, n3};

        if (write + 5 >= grid.maxVertices) return;

        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            const float3 p = tri[i];
            grid.d_vertices[write + i] = make_float4(p.x, p.y, p.z, 1.0f);

            if (grid.d_vertexColors) {
                grid.d_vertexColors[write + i] = make_float4(color_center.x, color_center.y, color_center.z, 1.0f);
            }

            if (grid.d_normals) {
                const float3 n = triN[i];
                grid.d_normals[write + i] = make_float4(n.x, n.y, n.z, 0.0f);
            }

            if (grid.d_primitiveIDs) {
                *((float*)&grid.d_primitiveIDs[write + i]) = (float)dominantPrimID;
            }

            if (grid.d_uvCoords) {
                grid.d_uvCoords[write + i] = triUV[i];
            }

            if (grid.d_indices) {
                grid.d_indices[write + i] = write + i;
            }
        }

        write += 6;
    };

    // X edge: corners 0-1, quad uses cells (x,y,z), (x,y-1,z), (x,y,z-1), (x,y-1,z-1)
    if (((mask >> 1) & 1) != (unsigned char)s0) {
        if (xyz.y > 0 && xyz.z > 0) {
            int pA, pB, pC, pD;
            if (lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z,     pA) &&
                lookupPackedCell(grid, xyz.x,     xyz.y - 1, xyz.z,     pB) &&
                lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z - 1, pC) &&
                lookupPackedCell(grid, xyz.x,     xyz.y - 1, xyz.z - 1, pD)) {
                emitQuadFromPackets(pA, pB, pD, pC); // ordering chosen for consistent winding (approx)
            }
        }
    }

    // Y edge: corners 0-3, quad uses cells (x,y,z), (x-1,y,z), (x,y,z-1), (x-1,y,z-1)
    if (((mask >> 3) & 1) != (unsigned char)s0) {
        if (xyz.x > 0 && xyz.z > 0) {
            int pA, pB, pC, pD;
            if (lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z,     pA) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y,     xyz.z,     pB) &&
                lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z - 1, pC) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y,     xyz.z - 1, pD)) {
                emitQuadFromPackets(pA, pC, pD, pB);
            }
        }
    }

    // Z edge: corners 0-4, quad uses cells (x,y,z), (x-1,y,z), (x,y-1,z), (x-1,y-1,z)
    if (((mask >> 4) & 1) != (unsigned char)s0) {
        if (xyz.x > 0 && xyz.y > 0) {
            int pA, pB, pC, pD;
            if (lookupPackedCell(grid, xyz.x,     xyz.y,     xyz.z,     pA) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y,     xyz.z,     pB) &&
                lookupPackedCell(grid, xyz.x,     xyz.y - 1, xyz.z,     pC) &&
                lookupPackedCell(grid, xyz.x - 1, xyz.y - 1, xyz.z,     pD)) {
                emitQuadFromPackets(pA, pB, pD, pC);
            }
        }
    }
}
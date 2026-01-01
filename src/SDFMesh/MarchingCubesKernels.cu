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

        // STEP 1: Evaluate all 3 vertices to get their individual data
        float dist_v[3];
        float3 color_v[3];
        int primitiveID_v[3];
        float3 localPos_v[3];
        
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            map(pTri[j], grid, time, dist_v[j], color_v[j], primitiveID_v[j], localPos_v[j]);
        }
        
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
                grid.d_vertexColors[write + j] = make_float4(color_v[j].x, color_v[j].y, color_v[j].z, 1.0f);
            }
            
            // COMPUTE AND WRITE NORMALS (per-vertex for smooth shading)
            if (grid.d_normals) {
                float3 normal = computeNormal(p, grid, time);
                grid.d_normals[write + j] = make_float4(normal.x, normal.y, normal.z, 0.0f);
            }
            
            // WRITE PRIMITIVE ID - ALL vertices use DOMINANT primitive
            if (grid.d_primitiveIDs) {
                int texID = (dominantPrimID >= 0 && dominantPrimID < grid.numPrimitives) 
                    ? grid.d_primitives[dominantPrimID].textureID 
                    : 0;
                *((float*)&grid.d_primitiveIDs[write + j]) = (float)texID;
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

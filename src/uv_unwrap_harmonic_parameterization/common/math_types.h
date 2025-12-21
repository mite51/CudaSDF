#pragma once
#include <cstdint>
#include <cmath>

namespace uv {

struct vec2 { float x, y; };
struct vec3 { float x, y, z; };
struct vec4 { float x, y, z, w; };

struct uvec3 { uint32_t x, y, z; };
struct ivec2  { int x, y; };
struct ivec3  { int x, y, z; };

inline vec3 operator+(const vec3& a, const vec3& b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline vec3 operator-(const vec3& a, const vec3& b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline vec3 operator*(const vec3& a, float s){ return {a.x*s,a.y*s,a.z*s}; }

inline float dot(const vec3& a, const vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline vec3 cross(const vec3& a, const vec3& b){
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float norm(const vec3& a){ return std::sqrt(dot(a,a)); }
inline vec3 normalize(const vec3& a){
  float n = norm(a); return (n>0) ? a*(1.0f/n) : vec3{0,0,0};
}

inline vec2 operator+(const vec2& a, const vec2& b){ return {a.x+b.x,a.y+b.y}; }
inline vec2 operator-(const vec2& a, const vec2& b){ return {a.x-b.x,a.y-b.y}; }
inline vec2 operator*(const vec2& a, float s){ return {a.x*s,a.y*s}; }

} // namespace uv

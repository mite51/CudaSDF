#pragma once
#include <vector>
#include "math_types.h"

namespace uv {

struct Mesh {
  std::vector<vec3> V;   // positions
  std::vector<uvec3> F;   // triangles (indices into V)
  std::vector<vec3> N;   // optional vertex normals (can be empty)
};

struct MeshDerived {
  std::vector<vec3> faceNormal; // per triangle
  std::vector<float>  faceArea;   // per triangle
};

MeshDerived ComputeMeshDerived(const Mesh& mesh);

} // namespace uv

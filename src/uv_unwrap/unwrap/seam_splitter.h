#pragma once
#include <vector>
#include "../common/mesh.h"
#include "../common/math_types.h"
#include "unwrap_result.h"

namespace uv {

// Re-creates the mesh topology such that vertices on chart boundaries are duplicated.
// This allows for hard UV seams.
// Returns a new Mesh (with more vertices) and a remapped UnwrapResult.
void SplitVerticesByChart(const Mesh& inputMesh,
                          const UnwrapResult& inputResult,
                          Mesh& outMesh,
                          UnwrapResult& outResult);

} // namespace uv


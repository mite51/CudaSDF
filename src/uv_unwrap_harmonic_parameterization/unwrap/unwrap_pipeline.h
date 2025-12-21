#pragma once
#include "../common/mesh.h"
#include "unwrap_config.h"
#include "unwrap_result.h"

namespace uv {

class UnwrapPipeline {
public:
  UnwrapResult Run(const Mesh& mesh, const UnwrapConfig& cfg);
};

} // namespace uv

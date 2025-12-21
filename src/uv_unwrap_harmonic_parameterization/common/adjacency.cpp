#include "adjacency.h"
#include "hash_utils.h"
#include <unordered_map>

namespace uv {

TriAdjacency BuildTriangleAdjacency(const Mesh& mesh) {
    TriAdjacency adj;
    adj.triNbr.resize(mesh.F.size(), {-1, -1, -1});

    // Map: EdgeKey -> {triIndex, localEdgeIndex}
    std::unordered_map<uint64_t, std::pair<int, int>> edgeMap;
    edgeMap.reserve(mesh.F.size() * 3); 

    for (int t = 0; t < (int)mesh.F.size(); ++t) {
        uint32_t idx[3] = { mesh.F[t].x, mesh.F[t].y, mesh.F[t].z };

        for (int k = 0; k < 3; ++k) {
            uint32_t v0 = idx[k];
            uint32_t v1 = idx[(k + 1) % 3];
            
            uint64_t key = PackEdgeKey(v0, v1);
            auto it = edgeMap.find(key);

            if (it != edgeMap.end()) {
                int otherT = it->second.first;
                int otherK = it->second.second;

                if (k == 0) adj.triNbr[t].x = otherT;
                else if (k == 1) adj.triNbr[t].y = otherT;
                else adj.triNbr[t].z = otherT;

                if (otherK == 0) adj.triNbr[otherT].x = t;
                else if (otherK == 1) adj.triNbr[otherT].y = t;
                else adj.triNbr[otherT].z = t;
                
                edgeMap.erase(it); 
            } else {
                edgeMap[key] = {t, k};
            }
        }
    }

    return adj;
}

} // namespace uv

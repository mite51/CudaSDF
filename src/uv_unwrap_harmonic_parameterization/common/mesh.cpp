#include "mesh.h"

namespace uv {

MeshDerived ComputeMeshDerived(const Mesh& mesh) {
    MeshDerived d;
    size_t numTris = mesh.F.size();
    d.faceNormal.resize(numTris);
    d.faceArea.resize(numTris);

    for(size_t i=0; i<numTris; ++i) {
        uint32_t i0 = mesh.F[i].x;
        uint32_t i1 = mesh.F[i].y;
        uint32_t i2 = mesh.F[i].z;

        vec3 v0 = mesh.V[i0];
        vec3 v1 = mesh.V[i1];
        vec3 v2 = mesh.V[i2];

        vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
        vec3 c = cross(e1, e2);
        
        float len = norm(c);
        d.faceArea[i] = 0.5f * len;
        
        if (len > 1e-8f) {
            d.faceNormal[i] = c * (1.0f / len);
        } else {
            d.faceNormal[i] = {0,0,0};
        }
    }
    return d;
}

} // namespace uv

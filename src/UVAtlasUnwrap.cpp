#include "UVAtlasUnwrap.h"

#include <unordered_map>

#include <DirectXMath.h>
#include <dxgiformat.h>

#include "UVAtlas.h"

#include "uv_unwrap_harmonic_parameterization/common/adjacency.h"

namespace uvatlas {
namespace {

static std::vector<uint32_t> BuildUVAtlasAdjacency(const uv::Mesh& mesh)
{
    // UVAtlas expects an array of nFaces*3 uint32_t where each entry is the neighboring
    // face index across that edge or uint32_t(-1) for boundary edges.
    uv::TriAdjacency triAdj = uv::BuildTriangleAdjacency(mesh);

    const size_t nFaces = mesh.F.size();
    std::vector<uint32_t> adjacency(nFaces * 3, uint32_t(-1));

    for (size_t f = 0; f < nFaces; ++f)
    {
        const auto& nbr = triAdj.triNbr[f];

        adjacency[f * 3 + 0] = (nbr.x >= 0) ? uint32_t(nbr.x) : uint32_t(-1);
        adjacency[f * 3 + 1] = (nbr.y >= 0) ? uint32_t(nbr.y) : uint32_t(-1);
        adjacency[f * 3 + 2] = (nbr.z >= 0) ? uint32_t(nbr.z) : uint32_t(-1);
    }

    return adjacency;
}

} // namespace

bool CreateAtlas(
    const uv::Mesh& mesh,
    const Config& cfg,
    std::vector<float4>& outPositions,
    std::vector<float2>& outUVs,
    std::vector<uint32_t>& outIndices,
    std::string* outError)
{
    if (mesh.V.empty() || mesh.F.empty())
    {
        if (outError) *outError = "Empty mesh";
        return false;
    }

    // Convert inputs
    std::vector<DirectX::XMFLOAT3> positions(mesh.V.size());
    for (size_t i = 0; i < mesh.V.size(); ++i)
    {
        positions[i] = DirectX::XMFLOAT3(mesh.V[i].x, mesh.V[i].y, mesh.V[i].z);
    }

    std::vector<uint32_t> indices(mesh.F.size() * 3);
    for (size_t f = 0; f < mesh.F.size(); ++f)
    {
        indices[f * 3 + 0] = mesh.F[f].x;
        indices[f * 3 + 1] = mesh.F[f].y;
        indices[f * 3 + 2] = mesh.F[f].z;
    }

    std::vector<uint32_t> adjacency = BuildUVAtlasAdjacency(mesh);

    // Run UVAtlasCreate (partition + pack)
    std::vector<DirectX::UVAtlasVertex> outVerts;
    std::vector<uint8_t> outIndexBytes;

    float maxStretchOut = 0.f;
    size_t numChartsOut = 0;

    HRESULT hr = DirectX::UVAtlasCreate(
        positions.data(),
        positions.size(),
        indices.data(),
        DXGI_FORMAT_R32_UINT,
        mesh.F.size(),
        cfg.maxCharts,
        cfg.maxStretch,
        cfg.width,
        cfg.height,
        cfg.gutterPx,
        adjacency.data(),
        nullptr, // falseEdgeAdjacency
        nullptr, // pIMTArray
        [](float) -> HRESULT { return S_OK; }, // status callback
        DirectX::UVATLAS_DEFAULT_CALLBACK_FREQUENCY,
        DirectX::UVATLAS_DEFAULT,
        outVerts,
        outIndexBytes,
        nullptr,
        nullptr,
        &maxStretchOut,
        &numChartsOut);

    if (FAILED(hr))
    {
        if (outError)
        {
            *outError = "UVAtlasCreate failed, HRESULT=0x" + std::to_string(uint32_t(hr));
        }
        return false;
    }

    // Decode indices
    if (outIndexBytes.size() != mesh.F.size() * 3 * sizeof(uint32_t))
    {
        if (outError) *outError = "UVAtlas output index buffer size mismatch";
        return false;
    }

    outIndices.resize(mesh.F.size() * 3);
    memcpy(outIndices.data(), outIndexBytes.data(), outIndexBytes.size());

    // Convert outputs
    outPositions.resize(outVerts.size());
    outUVs.resize(outVerts.size());
    for (size_t i = 0; i < outVerts.size(); ++i)
    {
        outPositions[i] = make_float4(outVerts[i].pos.x, outVerts[i].pos.y, outVerts[i].pos.z, 1.0f);
        outUVs[i] = make_float2(outVerts[i].uv.x, outVerts[i].uv.y);
    }

    return true;
}

} // namespace uvatlas



# CUDA UV Atlas Unwrap + Multi-View Projection Bake

---

## Project Summary — Automatic UV Atlas + Bake Pipeline (CUDA / C++)

### Context / Goal

We are building an automatic, robust UV unwrapping pipeline for meshes generated via marching cubes from SDF primitives, targeting single 0–1 UV atlas generation suitable for baking multi-view image projections (e.g., from LLM-generated images).

### Key Constraints

- **Mesh size**: low–mid triangle counts (arbitrary but not huge)
- **Seams acceptable** if not excessive (padding + dilation allowed)
- **One-time preprocess** per mesh is fine (but runs async in a job system)
- **C++ with CUDA 11.0**
- Texture projection / depth / segmentation will be a separate module later

---

## High-Level Pipeline

### Inputs

- **Triangle mesh** (V, F) from marching cubes


### Outputs

- Per-vertex atlas UVs in `[0,1]`
- Chart ID per triangle
- Atlas dimensions
- (Later) atlas textures from baking

---

## Unwrap Design (Main Focus)

### 1) Chart Generation (CPU)

**Goal**: few seams, robust charts.

- Compute per-triangle normals
- Cluster normals into K fixed directions (K = 12 or 24)
- Connected components per cluster → initial charts
- Merge tiny charts into neighbors
- Split oversized charts (by tri count / geodesic size)

**Result**: modest chart count, seams mostly where normals change.

### 2) Boundary Loop Extraction

- Identify chart boundary edges (neighbor tri in different chart or none)
- Stitch directed edges into loops
- Choose largest loop as outer boundary
- **v1 simplification**: if a chart has holes, split into multiple charts
  - This avoids constrained solves and improves robustness.

### 3) Per-Chart Flattening (CPU, Eigen)

**Goal**: fast, flip-free initial UVs.

- Map boundary loop to a circle (by cumulative edge length)
- Solve harmonic parameterization for interior vertices
  - Laplacian with cotangent weights (or uniform fallback)
  - Dirichlet boundary conditions
- Normalize chart UVs to chart-local `[0,1]`

**Notes**:
- Eigen is used initially for simplicity and stability
- SLIM is not required for v1
- AMGX integration point is reserved here if GPU solves are needed later

### 4) Atlas Packing (CPU)

**Goal**: single UV atlas, no overlaps.

- Compute chart UV AABBs
- Add padding (gutter) in pixels
- Pack rectangles with a skyline packer
- Compute per-chart affine transform:
  ```
  uv_atlas = offset + uv_chart01 * scale
  ```
- Assign final per-vertex atlas UVs

**CPU packing is chosen because**:
- chart counts are modest
- cost is negligible compared to GPU baking
- simpler + more deterministic

---

## Implementation Details

**Target**: existing C++ / CUDA 11.0 project  
**Goal**: automatic, robust unwrap to a single 0–1 UV atlas + bake multiple view images into the atlas (async job-friendly).  
**Assumptions**: marching cubes iso-surface triangles, low-to-mid triangle counts, seams acceptable but not excessive, padding + dilation allowed.

### 0) Big Picture Pipeline

#### Inputs

**Triangle mesh**:
- `vertices V[i] : float3`
- `triangles F[t] : uint3` (indices into V)
- optional vertex normals; otherwise compute

#### Outputs

- `uv[i] : float2` (atlas UV in `[0,1]`)
- `triChart[t] : int` (chart id per tri)
- atlas texture RGBA + debug buffers (optional)

#### Stages

1. **Adjacency**
2. **Charting** (normal clustering + connected components + merge/split)
3. **Boundary loops** (outer loop; split holes for v1)
4. **Per-chart flatten** (boundary circle + harmonic interior solve; optional SLIM later)
5. **Atlas packing** (CPU skyline pack)
6. **Bake** (CUDA UV raster + multi-view project/blend + dilation)
7. **Optional**: Depth prepass per view for occlusion

---

## 1) File Structure

```
src/
  common/
    math_types.h
    mesh.h
    adjacency.h
    adjacency.cpp
    timer.h
    logger.h
    hash_utils.h
    image_view.h

  unwrap/
    unwrap_config.h
    unwrap_result.h
    unwrap_pipeline.h
    unwrap_pipeline.cpp

    chart_builder.h
    chart_builder.cpp

    boundary_loops.h
    boundary_loops.cpp

    flatten_harmonic.h
    flatten_harmonic.cpp

    atlas_packer.h
    atlas_packer.cpp

    uv_postprocess.h
    uv_postprocess.cpp

  bake/
    bake_config.h
    bake_types.h

    cuda_texture.h
    cuda_texture.cu

    atlas_buffers.h
    atlas_buffers.cu

    tile_binner.cuh
    tile_binner.cu

    atlas_raster.cuh
    atlas_raster.cu

    project_blend.cuh
    project_blend.cu

    depth_prepass.cuh
    depth_prepass.cu

    dilation.cuh
    dilation.cu

    bake_pipeline.h
    bake_pipeline.cu

  amgx/
    amgx_context.h
    amgx_context.cpp
    amgx_solver.h
    amgx_solver.cpp

tests/
  CMakeLists.txt
  test_main.cpp
  test_adjacency.cpp
  test_charting.cpp
  test_boundary_loops.cpp
  test_flatten_harmonic.cpp
  test_atlas_packer.cpp
  test_bake_projection_smoke.cpp
```

---

## 2) Common Types (Headers)

### `src/common/math_types.h`

```cpp
#pragma once
#include <cstdint>
#include <cmath>

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct uint3 { uint32_t x, y, z; };
struct int2  { int x, y; };
struct int3  { int x, y, z; };

inline float3 operator+(const float3& a, const float3& b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline float3 operator-(const float3& a, const float3& b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline float3 operator*(const float3& a, float s){ return {a.x*s,a.y*s,a.z*s}; }

inline float dot(const float3& a, const float3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float3 cross(const float3& a, const float3& b){
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float norm(const float3& a){ return std::sqrt(dot(a,a)); }
inline float3 normalize(const float3& a){
  float n = norm(a); return (n>0) ? a*(1.0f/n) : float3{0,0,0};
}

// Column-major 4x4 like OpenGL
struct float4x4 {
  float m[16];
};

inline float4 mul(const float4x4& M, const float4& v){
  const float* a = M.m;
  return {
    a[0]*v.x + a[4]*v.y + a[8]*v.z  + a[12]*v.w,
    a[1]*v.x + a[5]*v.y + a[9]*v.z  + a[13]*v.w,
    a[2]*v.x + a[6]*v.y + a[10]*v.z + a[14]*v.w,
    a[3]*v.x + a[7]*v.y + a[11]*v.z + a[15]*v.w
  };
}
```

### `src/common/mesh.h`

```cpp
#pragma once
#include <vector>
#include "math_types.h"

struct Mesh {
  std::vector<float3> V;   // positions
  std::vector<uint3>  F;   // triangles (indices into V)
  std::vector<float3> N;   // optional vertex normals (can be empty)
};

struct MeshDerived {
  std::vector<float3> faceNormal; // per triangle
  std::vector<float>  faceArea;   // per triangle
};

MeshDerived ComputeMeshDerived(const Mesh& mesh);
// TODO: implement (compute face normals, areas, optionally vertex normals).
```

### `src/common/adjacency.h`

```cpp
#pragma once
#include <vector>
#include "math_types.h"
#include "mesh.h"

struct TriAdjacency {
  // For each tri t, neighbor across each local edge e in {0,1,2}, or -1
  std::vector<int3> triNbr;
};

TriAdjacency BuildTriangleAdjacency(const Mesh& mesh);
// TODO: edge hash map: undirected edge (min,max) -> (tri, localEdge).
```

### `src/common/hash_utils.h`

```cpp
#pragma once
#include <cstdint>
#include <utility>

// Hash for 64-bit packed edge key
inline uint64_t PackEdgeKey(uint32_t a, uint32_t b){
  uint32_t lo = (a < b) ? a : b;
  uint32_t hi = (a < b) ? b : a;
  return (uint64_t(hi) << 32) | uint64_t(lo);
}
```

### `src/common/image_view.h`

```cpp
#pragma once
#include <cstdint>

struct ImageViewRGBA8 {
  int width = 0, height = 0;
  const uint8_t* data = nullptr; // RGBA interleaved
  int strideBytes = 0;           // bytes per row
};
```

> **Note**: You likely already have timer/logger; keep those minimal.

---

## 3) Unwrap Stage (Headers + Responsibilities)

### `src/unwrap/unwrap_config.h`

```cpp
#pragma once

struct UnwrapConfig {
  int normalClusterK = 12;     // 12 or 24
  int minChartTris   = 64;     // merge smaller
  int maxChartTris   = 5000;   // split bigger
  float maxChartGeodesic = 0.0f; // 0 = disabled; else split by approx diameter threshold
  int atlasMaxSize   = 4096;   // max atlas dimension
  int paddingPx      = 8;      // gutter padding in pixels
  bool useCotanWeights = true; // harmonic Laplacian weighting
};
```

### `src/unwrap/unwrap_result.h`

```cpp
#pragma once
#include <vector>
#include "common/math_types.h"

struct UnwrapResult {
  std::vector<float2> uvAtlas;    // per vertex UV in [0,1]
  std::vector<int>    triChart;   // per triangle chart ID
  int chartCount = 0;

  int atlasW = 0, atlasH = 0;

  // Optional: chart-local UV (debug / packing)
  std::vector<float2> uvChart01; // per vertex, in chart-local [0,1], meaningful if stored per-vertex-per-chart (v2)
};
```

### `src/unwrap/unwrap_pipeline.h`

```cpp
#pragma once
#include "common/mesh.h"
#include "unwrap_config.h"
#include "unwrap_result.h"

class UnwrapPipeline {
public:
  UnwrapResult Run(const Mesh& mesh, const UnwrapConfig& cfg);

  // TODO: add debug outputs / intermediate dumps if needed.
};
```

### `src/unwrap/chart_builder.h`

```cpp
#pragma once
#include <vector>
#include "common/mesh.h"
#include "common/adjacency.h"
#include "unwrap_config.h"

struct Charts {
  std::vector<int> triChart;   // size = T
  int chartCount = 0;

  // Optional convenience:
  std::vector<std::vector<int>> chartTris;
};

class ChartBuilder {
public:
  Charts Build(const Mesh& mesh,
               const MeshDerived& derived,
               const TriAdjacency& adj,
               const UnwrapConfig& cfg);

private:
  std::vector<float3> BuildClusterDirections(int K);
  std::vector<int> AssignClusters(const MeshDerived& derived,
                                  const std::vector<float3>& dirs);

  Charts ConnectedComponentsByCluster(const TriAdjacency& adj,
                                      const std::vector<int>& triCluster);

  void MergeSmallCharts(const Mesh& mesh,
                        const TriAdjacency& adj,
                        const MeshDerived& derived,
                        const UnwrapConfig& cfg,
                        Charts& charts);

  void SplitLargeCharts(const Mesh& mesh,
                        const TriAdjacency& adj,
                        const UnwrapConfig& cfg,
                        Charts& charts);

  // TODO: helpers to build chart adjacency, compute chart area/normal, etc.
};
```

### `src/unwrap/boundary_loops.h`

```cpp
#pragma once
#include <vector>
#include <cstdint>
#include "common/mesh.h"
#include "common/adjacency.h"
#include "unwrap_result.h"

struct BoundaryLoop {
  std::vector<uint32_t> verts; // ordered loop vertex ids (global)
};

struct ChartBoundary {
  int chartId = -1;
  BoundaryLoop outer;
  std::vector<BoundaryLoop> holes;
};

class BoundaryLoops {
public:
  // Extract boundary for each chart.
  // v1 strategy: if multiple loops, keep largest as outer and treat others as holes
  // (optionally split charts to remove holes).
  std::vector<ChartBoundary> Extract(const Mesh& mesh,
                                     const TriAdjacency& adj,
                                     const std::vector<int>& triChart,
                                     int chartCount);

  // Utility: split charts with holes into separate charts (v1 robust path)
  // TODO: implement if you want to avoid constrained solves.
  // void SplitChartsOnHoles(...);

private:
  // TODO: boundary edge extraction (directed edges), loop stitching.
};
```

### `src/unwrap/flatten_harmonic.h`

```cpp
#pragma once
#include <vector>
#include <unordered_map>
#include "common/mesh.h"
#include "common/adjacency.h"
#include "common/math_types.h"
#include "boundary_loops.h"

struct ChartUV {
  int chartId = -1;
  std::vector<uint32_t> chartVerts;  // global vertex ids in chart
  std::vector<float2> uv;            // chart-local UV per chartVerts (not atlas packed)
  float2 minUV{0,0}, maxUV{1,1};      // AABB in chart UV space
};

class HarmonicFlattener {
public:
  // Produces chart-local UVs for each chart boundary.
  std::vector<ChartUV> Flatten(const Mesh& mesh,
                               const std::vector<int>& triChart,
                               int chartCount,
                               const std::vector<ChartBoundary>& boundaries,
                               bool useCotanWeights);

private:
  // Build per-chart local indexing (global vertex id -> local idx)
  void BuildLocalIndex(const Mesh& mesh,
                       const std::vector<int>& triChart,
                       int chartId,
                       std::vector<uint32_t>& outChartVerts,
                       std::unordered_map<uint32_t,int>& outLocalIndex);

  // Boundary circle map (edge length parameterization)
  void MapBoundaryToCircle(const Mesh& mesh,
                           const BoundaryLoop& loop,
                           const std::unordered_map<uint32_t,int>& localIndex,
                           std::vector<float2>& uvFixed,
                           std::vector<uint8_t>& isBoundary);

  // Harmonic solve for interior with Dirichlet boundary constraints
  // NOTE: Eigen recommended for v1 implementation.
  void SolveHarmonic(const Mesh& mesh,
                     int chartId,
                     const std::vector<uint32_t>& chartVerts,
                     const std::unordered_map<uint32_t,int>& localIndex,
                     const std::vector<int>& triChart,
                     const std::vector<uint8_t>& isBoundary,
                     const std::vector<float2>& uvBoundary,
                     bool useCotanWeights,
                     std::vector<float2>& outUV);

  // TODO: cotan weight compute, Laplacian assembly, constraint elimination.
};
```

### `src/unwrap/atlas_packer.h`

```cpp
#pragma once
#include <vector>
#include "common/math_types.h"
#include "unwrap_config.h"
#include "flatten_harmonic.h"
#include "unwrap_result.h"

struct PackedChart {
  int chartId = -1;
  int x=0, y=0, w=0, h=0;    // in pixels in atlas
  float2 offset01{0,0};      // normalized atlas offset
  float2 scale01{1,1};       // normalized atlas scale
};

class AtlasPacker {
public:
  // Packs chart rectangles into atlas, returns packed chart transforms and atlas size.
  // Also writes final per-vertex atlas UVs into result (via chart mapping).
  void PackAndAssignUVs(const std::vector<ChartUV>& chartUVs,
                        const UnwrapConfig& cfg,
                        const Mesh& mesh,
                        const std::vector<int>& triChart,
                        UnwrapResult& out);

  const std::vector<PackedChart>& GetPackedCharts() const { return packed_; }

private:
  std::vector<PackedChart> packed_;

  // Skyline packer
  void SkylinePack(const std::vector<int2>& rectWH,
                   int maxSize,
                   int paddingPx,
                   int& outW,
                   int& outH,
                   std::vector<int2>& outXY);

  // TODO: rect sizing policy (by chart area, visibility weights later).
};
```

### `src/unwrap/uv_postprocess.h`

```cpp
#pragma once
#include <vector>
#include "common/math_types.h"

// Optional helpers: normalize charts, compute UV AABB, etc.
namespace uv_post {
  void ComputeAABB(const std::vector<float2>& uv, float2& outMin, float2& outMax);
  void NormalizeTo01(std::vector<float2>& uv, float2& inOutMin, float2& inOutMax);
}
```

### `src/unwrap/unwrap_pipeline.cpp` Flow (Conceptual)

```cpp
derived = ComputeMeshDerived(mesh)

adj = BuildTriangleAdjacency(mesh)

charts = ChartBuilder.Build(...)

boundaries = BoundaryLoops.Extract(...)

chartUVs = HarmonicFlattener.Flatten(...)

AtlasPacker.PackAndAssignUVs(...) → UnwrapResult
```

---

## 4) Bake Stage (Headers + CUDA Kernels)

### `src/bake/bake_config.h`

```cpp
#pragma once

struct BakeConfig {
  int atlasW = 2048;         // set from unwrap
  int atlasH = 2048;

  int tileSize = 16;

  // Multi-view blend settings
  float anglePower = 4.0f;   // weight = max(0,dot(N,viewDir))^p
  bool useDepthTest = false;
  float depthEps = 1e-3f;

  // Padding fill
  int dilationIters = 16;
};
```

### `src/bake/bake_types.h`

```cpp
#pragma once
#include <vector>
#include "common/math_types.h"
#include <cuda_runtime.h>

struct BakeCamera {
  float4x4 view;
  float4x4 proj;
  float4x4 viewProj;     // proj*view
  int width = 0, height = 0;
  float3 positionWS{0,0,0};
};

// Wrap a CUDA texture object (RGBA8 or float4 etc.)
struct CudaTex2D {
  cudaTextureObject_t tex = 0;
  int width = 0, height = 0;
};

struct BakeView {
  BakeCamera cam;
  CudaTex2D  colorTex;

  // Optional depth buffer (device pointer or CUDA array/texture)
  float* d_depth = nullptr;  // size width*height
};
```

### `src/bake/cuda_texture.h`

```cpp
#pragma once
#include "common/image_view.h"
#include "bake_types.h"

class CudaTextureUploader {
public:
  // Upload RGBA8 image to CUDA array + create cudaTextureObject_t
  static CudaTex2D UploadRGBA8(const ImageViewRGBA8& img);

  // Free texture object + underlying cudaArray
  static void Destroy(CudaTex2D& t);

  // TODO: store cudaArray pointer inside CudaTex2D if you want ownership tracking.
};
```

### `src/bake/atlas_buffers.h`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "common/math_types.h"

struct AtlasGPU {
  int W=0, H=0;
  uchar4* d_rgba = nullptr;   // atlas pixels
  uint8_t* d_mask = nullptr;  // 1 if filled else 0

  // Optional debug:
  // float4* d_accum; // if you want float accum before conversion
};

class AtlasBuffers {
public:
  static AtlasGPU Allocate(int W, int H);
  static void Free(AtlasGPU& a);
  static void Clear(AtlasGPU& a); // set rgba=0, mask=0
};
```

### `src/bake/tile_binner.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "common/math_types.h"

// Device-side binning structures
struct TileBins {
  int tileW=0, tileH=0;
  int tileSize=16;

  int* d_counts = nullptr;   // tileW*tileH
  int* d_offsets = nullptr;  // tileW*tileH
  int* d_triIds = nullptr;   // total refs
};

struct DeviceMeshUV {
  const float3* d_pos = nullptr;
  const float3* d_nrm = nullptr; // can be null if not used
  const float2* d_uv  = nullptr;
  const uint3*  d_tri = nullptr;
  int vertCount=0;
  int triCount=0;
};

namespace tile_binner {
  // Two-pass:
  // 1) count refs per tile
  // 2) prefix sum offsets
  // 3) fill tri ids
  void BuildBins(const DeviceMeshUV& mesh, int atlasW, int atlasH, int tileSize, TileBins& outBins);

  void FreeBins(TileBins& bins);

  // TODO: prefix sum implementation (thrust or custom scan). Thrust is easiest in CUDA 11.
}
```

### `src/bake/atlas_raster.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "tile_binner.cuh"
#include "atlas_buffers.h"
#include "bake_types.h"

namespace atlas_raster {

  // Rasterize in UV space, generate per-pixel P/N and call project_blend
  // v1: direct write to atlas inside this kernel (simplest)
  void RasterAndBake(const DeviceMeshUV& mesh,
                     const TileBins& bins,
                     const BakeView* d_views,
                     int viewCount,
                     AtlasGPU atlas,
                     const BakeConfig& cfg);

  // TODO: define device struct array for BakeView (copy host->device).
}
```

### `src/bake/project_blend.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include "bake_types.h"
#include "common/math_types.h"

__device__ inline bool ProjectPoint(const BakeCamera& cam, const float3& P,
                                    float& outPx, float& outPy, float& outDepthNdc)
{
  float4 clip = mul(cam.viewProj, float4{P.x,P.y,P.z,1.0f});
  if (clip.w == 0.0f) return false;
  float invW = 1.0f / clip.w;
  float ndcX = clip.x * invW;
  float ndcY = clip.y * invW;
  float ndcZ = clip.z * invW;

  // OpenGL NDC x,y in [-1,1], z in [-1,1] typically
  outPx = (ndcX * 0.5f + 0.5f) * cam.width;
  outPy = (ndcY * 0.5f + 0.5f) * cam.height;
  outDepthNdc = ndcZ;
  return true;
}

__device__ inline float FacingWeight(const float3& N, const float3& viewDir, float power)
{
  float d = dot(N, viewDir);
  if (d <= 0.0f) return 0.0f;
  // powf is fine; can approximate later if needed
  return powf(d, power);
}

// Sample from CudaTex2D with normalized coords
__device__ inline float4 SampleRGBA(const CudaTex2D& t, float u, float v)
{
  // TODO: use tex2D<float4>(t.tex, u, v) with normalized coords or unnormalized based on setup.
  // Keep as placeholder; actual implementation depends on cudaTextureDesc normalizedCoords.
  return {0,0,0,0};
}
```

### `src/bake/depth_prepass.cuh`

```cpp
#pragma once
#include "tile_binner.cuh"
#include "bake_types.h"

namespace depth_prepass {
  // Build per-view depth buffer by screen-space rasterization
  // TODO: implement UV-independent binner for screen space or reuse a generic binner with projected tri AABBs.
  void RasterDepth(const DeviceMeshUV& mesh,
                   const BakeCamera& cam,
                   float* d_depth,
                   int W, int H);
}
```

### `src/bake/dilation.cuh`

```cpp
#pragma once
#include "atlas_buffers.h"

namespace dilation {
  // Fill empty pixels by expanding nearby filled pixels N iterations
  void Dilate(AtlasGPU atlas, int iters);
}
```

### `src/bake/bake_pipeline.h`

```cpp
#pragma once
#include <vector>
#include "common/mesh.h"
#include "unwrap/unwrap_result.h"
#include "bake_config.h"
#include "bake_types.h"
#include "atlas_buffers.h"

struct MeshGPU {
  float3* d_pos = nullptr;
  float3* d_nrm = nullptr;
  float2* d_uv  = nullptr;
  uint3*  d_tri = nullptr;
  int vertCount=0, triCount=0;

  // TODO: allocate/free helpers
};

class BakePipeline {
public:
  // Upload mesh + UVs and bake atlas from multiple views
  AtlasGPU Bake(const Mesh& mesh,
                const UnwrapResult& unwrap,
                const std::vector<BakeView>& views,
                const BakeConfig& cfg);

private:
  // TODO: upload mesh to GPU, compute normals if needed, copy views to device, etc.
};
```

---

## 5) AMGX Integration Points (Recommended)

You don't need AMGX to ship v1. You do want to wire it in cleanly, so later you can:

- run harmonic solves on GPU (optional)
- add SLIM per chart (likely where AMGX really shines)

### Where AMGX Plugs In

#### A) Harmonic Flatten (Stage 2)

In `HarmonicFlattener::SolveHarmonic(...)`:

- **current v1**: build sparse `L_ii` on CPU and solve with Eigen
- **later GPU option**:
  - assemble CSR (pattern fixed per chart)
  - call AMGX solve twice (for u and v)

This is a good integration point because:
- matrix is SPD
- per chart sizes are manageable
- you can reuse configuration

#### B) SLIM Per Chart (Optional Stage Later)

If you add `SlimRefiner`:

- local step on CPU/GPU (per triangle)
- global step is sparse solve (SPD-ish), perfect for AMGX
- you can reuse AMG hierarchy across a few SLIM iterations per chart ("frozen preconditioner")

### Suggested AMGX Wrapper Files

#### `src/amgx/amgx_context.h`

```cpp
#pragma once

// Forward declare AMGX types without pulling headers everywhere
struct AMGXResources;
struct AMGXConfig;

class AmgxContext {
public:
  AmgxContext();
  ~AmgxContext();

  bool Initialize(const char* configString); // or config file text
  void Shutdown();

  // Accessors for solver wrappers
  void* Resources() const; // AMGX_resources_handle
  void* Config() const;    // AMGX_config_handle

private:
  void* resources_ = nullptr;
  void* config_ = nullptr;
  bool initialized_ = false;

  // TODO: store AMGX init state; call AMGX_initialize / finalize.
};
```

#### `src/amgx/amgx_solver.h`

```cpp
#pragma once
#include <cstdint>

struct CsrDeviceMatrix {
  int n = 0;
  int nnz = 0;
  int* d_rowPtr = nullptr;
  int* d_colInd = nullptr;
  float* d_vals = nullptr;
};

class AmgxSolver {
public:
  AmgxSolver() = default;
  ~AmgxSolver();

  bool Create(AmgxContext& ctx, int n);
  void Destroy();

  // Provide / update matrix values (pattern assumed stable after first set)
  bool SetMatrix(const CsrDeviceMatrix& A);

  // Solve Ax=b into x
  bool Solve(const float* d_b, float* d_x);

  // TODO: allow updating values without rebuilding hierarchy (depending on AMGX config).
private:
  void* solver_ = nullptr; // AMGX_solver_handle
  void* A_ = nullptr;      // AMGX_matrix_handle
  void* x_ = nullptr;      // AMGX_vector_handle
  void* b_ = nullptr;      // AMGX_vector_handle
  int n_ = 0;
};
```

### AMGX Configs to Consider (Later)

- **PCG + AMG preconditioner** for SPD systems (harmonic / SLIM global)
- Use a **config string in code** to avoid config files initially
- Consider **"setup once, solve many"** and rebuild hierarchy only occasionally for SLIM

---

## 6) Third-Party Recommendations

### For v1 (Fastest to Implement + Stable)

**Eigen** for harmonic solve per chart:
- easiest, robust, small integration cost
- you can use `Eigen::SimplicialLLT` for smaller charts or `ConjugateGradient` for larger

### For Later GPU Solver Acceleration / SLIM

**AMGX integration**:
- keep it behind `amgx/` wrappers
- use it first for optional GPU harmonic solve or SLIM global step

### For Scans/Prefix Sums in Binning

**Thrust** is fine in CUDA 11.0 for prefix sums (lowest friction).
- If you prefer zero dependencies, later implement your own scan.

### For Unit Testing

**Catch2** (single header) or **GoogleTest**.
- Catch2 is simplest drop-in for CMake.
- GoogleTest is more "standard" but heavier.

---

## 7) Unit Tests Task (Generate & Run)

### Testing Framework Choice

**Catch2** recommended for minimal friction.

### `tests/CMakeLists.txt` (Sketch)

- builds a tests executable
- links against your library targets (unwrap + common)
- runs with ctest

### Test Cases (What to Implement)

#### Adjacency Correctness

- Build a simple known mesh (two triangles making a quad)
- Verify neighbors match expected, boundary edges have -1

#### Charting Sanity

- Generate a sphere-ish mesh (or cube) with known normals
- Ensure:
  - `chartCount > 0`
  - all triangles assigned chart `>=0`
  - charts merged/split within thresholds

#### Boundary Loops

- Simple planar mesh chart should have exactly one outer loop
- Loop is closed, no repeats except closure

#### Harmonic Flatten (Smoke + Invariants)

For a simple disk-like chart:
- boundary vertices should lie near circle
- UVs finite (not NaN/inf)
- triangle UV area not all zero

#### Atlas Packing

- No chart rectangles overlap
- All packed within atlas bounds
- UVs within `[0,1]` (allow tiny epsilon)

#### Bake Projection Smoke Test

- Tiny mesh + one or two views with synthetic images
- Bake to small atlas (e.g., 128×128)
- Assert atlas has nonzero filled pixels (mask count > 0)

### "Generate Test Meshes"

Provide helper functions in tests:
- `MakeQuadMesh()`
- `MakeIcoSphereLikeMesh(subdiv)` or a quick lat-long sphere generator
- Or reuse your marching cubes SDF extractor to produce a mesh for tests (if deterministic and fast)

---

## 8) Notes on Async Job System

- CPU unwrap and packing can run in a background job thread.
- GPU bake can run on a dedicated CUDA stream (per job).
- Keep intermediate allocations pooled if you'll regenerate frequently (tile bins, atlas buffers, depth buffers).

---

## 9) Suggested Implementation Order

(Lowest conflict / easiest integration)

1. **common/\*** types + adjacency
2. **unwrap/chart_builder** (no Eigen yet)
3. **unwrap/boundary_loops**
4. **unwrap/flatten_harmonic** using Eigen
5. **unwrap/atlas_packer**
6. **unwrap/unwrap_pipeline** (end-to-end UVs)
7. **Bake v1**: texture upload + tile binner + raster/project/blend + dilation
8. **Tests**
9. **AMGX wrappers** can be added early as stubs, but only "activated" after v1 works.

---

**End of Document**

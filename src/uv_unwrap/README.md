# UV Atlas Unwrapping Pipeline

## Overview

This module implements a fully automatic UV atlas unwrapping system for triangle meshes. It takes a 3D mesh (typically from Marching Cubes SDF extraction) and generates UV coordinates suitable for texture mapping.

**Key Features:**
- Automatic mesh segmentation into charts
- Harmonic parameterization for minimal distortion
- Efficient texture atlas packing
- Hard UV seam support via vertex splitting
- Configurable atlas resolution and padding

**Typical Performance:** ~700ms for 380K triangles on CPU (189K unique vertices)

---

## Algorithm

The unwrapping process consists of 6 stages:

### 1. **Mesh Welding**
Converts triangle soup (raw Marching Cubes output) into an indexed mesh with shared vertices.
- **Input:** `N` float4 vertices (triangle soup)
- **Output:** `V` unique vertices + `F` triangle indices
- **Method:** Spatial hashing with quantization (10000.0 factor)

### 2. **Chart Generation**
Segments the mesh into topologically simple patches ("charts") for flattening.
- **Input:** Indexed mesh
- **Output:** `~500` charts (per-triangle chart ID)
- **Method:** Region growing based on surface normal similarity
  - Seed selection: Unmarked triangle with largest area
  - Growth criteria: `dot(normal_i, normal_seed) > threshold` (typically 0.6)
  - Termination: All triangles assigned

**Why charts?** Flattening an entire complex 3D mesh to 2D would create severe distortion. Charts break the mesh into near-planar regions.

### 3. **Boundary Extraction**
Finds the outer loops of each chart for parameterization constraints.
- **Input:** Chart mesh + adjacency
- **Output:** Per-chart boundary loops (ordered vertex sequences)
- **Method:** 
  - Identify boundary edges (edges with only one adjacent triangle in chart)
  - Link edges into closed loops
  - Most charts have 1 loop; complex topology may have multiple

### 4. **Harmonic Flattening**
Parameterizes each chart to 2D UV space using Laplacian smoothing.
- **Input:** Chart geometry + boundary loop
- **Output:** Per-chart UV coordinates (0-1 range)
- **Method:** Harmonic map with fixed boundary
  - Pin boundary vertices to convex polygon (unit circle)
  - Solve: `Δu = 0` and `Δv = 0` (Laplace equation)
  - Interior vertices relax to minimize distortion
  - **Solver:** Eigen `SimplicialLDLT` (sparse Cholesky)

**Why harmonic?** It's a good balance between angle preservation (conformal) and area preservation, with fast convergence.

### 5. **Atlas Packing**
Arranges all charts into a single square texture atlas.
- **Input:** Chart UVs (local 0-1 space)
- **Output:** Global atlas UVs, atlas dimensions
- **Method:** Shelf packing
  - Sort charts by height (descending)
  - Pack left-to-right on current shelf
  - Start new shelf when width exceeded
  - Scale charts based on 3D surface area (larger charts get more pixels)

**Atlas Size:** 8192×8192 by default, 2px padding between charts

### 6. **Vertex Splitting**
Duplicates vertices at chart boundaries to support hard UV seams.
- **Input:** Welded mesh + atlas UVs
- **Output:** Split mesh (more vertices, each with unique UV)
- **Why?** 
  - A vertex on the edge of 2 charts needs 2 different UVs
  - Standard mesh format: 1 UV per vertex
  - Solution: Duplicate the vertex (one copy per chart)

---

## Implementation Details

### Directory Structure

```
src/uv_unwrap/
├── common/               # Data structures and utilities
│   ├── math_types.h      # vec2, vec3, vec4, uvec3, etc. (namespace uv::)
│   ├── mesh.h/.cpp       # Mesh and MeshDerived structures
│   ├── adjacency.h/.cpp  # Triangle adjacency graph
│   └── hash_utils.h      # Edge hashing utilities
│
└── unwrap/               # Core unwrapping algorithms
    ├── unwrap_config.h   # Configuration (atlas size, padding, thresholds)
    ├── unwrap_result.h   # Output structure (uvAtlas, wedgeUVs, etc.)
    ├── unwrap_pipeline.h/.cpp  # Main orchestration
    ├── chart_builder.h/.cpp    # Chart segmentation
    ├── boundary_loops.h/.cpp   # Boundary extraction
    ├── flatten_harmonic.h/.cpp # Harmonic parameterization
    ├── atlas_packer.h/.cpp     # Texture atlas packing
    └── seam_splitter.h/.cpp    # Vertex splitting for seams
```

### Key Data Structures

**`uv::Mesh`**
```cpp
struct Mesh {
    std::vector<vec3> V;      // Vertex positions
    std::vector<uvec3> F;     // Triangle indices (3 per face)
};
```

**`uv::UnwrapResult`**
```cpp
struct UnwrapResult {
    std::vector<vec2> uvAtlas;      // Per-vertex UVs (legacy)
    std::vector<vec2> wedgeUVs;     // Per-corner UVs (3 * numTriangles)
    std::vector<int> triChart;      // Per-triangle chart ID
    int chartCount;
    int atlasW, atlasH;             // Atlas dimensions
};
```

**`uv::UnwrapConfig`**
```cpp
struct UnwrapConfig {
    int atlasMaxSize = 8192;        // Max atlas dimension
    int paddingPx = 2;              // Padding between charts
    float chartNormalThreshold = 0.6f;  // Dot product threshold
    float chartMaxArea = 1e6f;      // Max triangles per chart
};
```

### Dependencies

- **Eigen 3.4.0**: For sparse linear algebra (harmonic solver)
  - Required: `SimplicialLDLT`, `SparseMatrix`, `Triplet`
  - CMake: `FetchContent` from GitLab

- **Standard Library**: `<vector>`, `<map>`, `<unordered_map>`, `<cmath>`, `<algorithm>`

### Usage Example

```cpp
#include "uv_unwrap/unwrap/unwrap_pipeline.h"
#include "uv_unwrap/unwrap/seam_splitter.h"

// 1. Convert raw vertices to indexed mesh
uv::Mesh mesh = WeldMesh(rawVertices);

// 2. Configure unwrap
uv::UnwrapConfig cfg;
cfg.atlasMaxSize = 8192;
cfg.paddingPx = 2;

// 3. Run pipeline
uv::UnwrapPipeline pipeline;
uv::UnwrapResult result = pipeline.Run(mesh, cfg);

// 4. Split vertices for rendering
uv::Mesh splitMesh;
uv::UnwrapResult splitResult;
uv::SplitVerticesByChart(mesh, result, splitMesh, splitResult);

// 5. Use splitMesh.V, splitMesh.F, splitResult.uvAtlas for rendering
```

### Performance Characteristics

| Stage | Time (380K tri) | Complexity | Notes |
|-------|-----------------|------------|-------|
| Welding | ~50ms | O(n log n) | `std::map` lookup |
| Derived Data | ~10ms | O(n) | Normal computation |
| Adjacency | ~20ms | O(n) | Hash map construction |
| Charting | ~100ms | O(n × k) | k = avg chart size |
| Boundaries | ~10ms | O(n) | Edge traversal |
| Harmonic Solve | ~500ms | O(m³) | m = avg chart verts (~1000) |
| Atlas Packing | ~5ms | O(c log c) | c = num charts (~500) |
| Vertex Split | ~10ms | O(n) | Linear scan |
| **Total** | **~700ms** | - | - |

**Bottleneck:** Harmonic solver (70% of total time)

### Known Limitations

1. **Small Charts**: If atlas fills up, later charts are skipped (warning printed)
   - Solution: Increase `atlasMaxSize` or reduce `paddingPx`

2. **Degenerate Geometry**: Zero-area triangles or non-manifold edges may cause issues
   - Charts with <3 vertices skip harmonic solve (UVs set to 0)

3. **Non-Manifold Meshes**: Algorithm assumes manifold input
   - Welding helps but doesn't guarantee manifold topology

4. **Memory**: Large meshes (>2M triangles) may need 64-bit builds
   - Sparse matrices can consume significant RAM

### Configuration Tips

**For High Quality:**
```cpp
cfg.atlasMaxSize = 8192;
cfg.paddingPx = 4;  // More padding = less bleeding
cfg.chartNormalThreshold = 0.7f;  // Stricter = more charts = less distortion
```

**For Speed:**
```cpp
cfg.atlasMaxSize = 4096;
cfg.paddingPx = 1;
cfg.chartNormalThreshold = 0.5f;  // Fewer, larger charts
```

**For Large Meshes:**
```cpp
cfg.chartMaxArea = 5000.0f;  // Limit chart size to improve solver performance
```

---

## Future Improvements (See `CUDA_Parallelization_Plan.md`)

- CUDA welding (10-50x speedup)
- CUDA adjacency building (5-10x speedup)
- CUDA vertex splitting (10-20x speedup)
- Parallel boundary extraction (3-5x speedup)
- Optional: GPU sparse solver for large charts (2-5x speedup)

**Estimated Total CUDA Speedup:** 5-10x (700ms → 70-150ms)


# Dual Contouring (CUDA) — Implementation Notes

This document describes the Dual Contouring (DC) implementation added alongside the existing Marching Cubes (MC) pipeline.

The goal of this DC path is to preserve the **same “feature contract”** as MC:
- Sparse **active-block scouting**
- Triangle-soup output (no indexing required)
- Optional per-vertex outputs: **UVs**, **primitive IDs**, **normals**, **colors**
- Compatibility with existing renderer / buffer bindings

It also adds:
- A runtime technique toggle (MC vs DC)
- A DC-specific, configurable **adaptive normal smoothing** pass

---

## Where the code lives

- **Kernels**
  - `src/SDFMesh/MarchingCubesKernels.cu`
    - Existing MC kernels remain intact
    - DC kernels are appended at the bottom of the file
- **Kernel launch wrappers**
  - `src/SDFMesh/Wrappers.cu`
- **Host orchestration**
  - `src/SDFMesh/CudaSDFMesh.cpp`
  - `src/SDFMesh/CudaSDFMesh.h`
- **GPU-side shared structs / buffers**
  - `src/SDFMesh/Commons.cuh`
- **Runtime toggles (MC/DC + smoothing knob)**
  - `src/main.cpp`

---

## Runtime usage / toggles

In `src/main.cpp`:
- Press **`M`** to select **Marching Cubes**
- Press **`D`** to select **Dual Contouring**
- Press **`[`** / **`]`** to decrease/increase **DC normal smoothing angle** (degrees)
  - Default is **30°**

The technique choice is passed into:
- `CudaSDFMesh::Update(..., MeshExtractionTechnique technique, float dcNormalSmoothAngleDeg)`

---

## Data model / buffers

### Existing packed MC buffers (reused)

In `SDFGrid` (`src/SDFMesh/Commons.cuh`):
- `d_activeBlocks`, `d_activeBlockCount`
- `d_packetVertexCounts`
- `d_packetVertexOffsets`

These are reused by DC for:
- Temporary *hasVertex* counting (0/1 per packed cell)
- Per-cell emitted soup-vertex counting (0/6/12… per packed cell)

### DC-specific buffers (lazy-allocated)

Added to `SDFGrid`:
- `int* d_blockToActiveId`
  - Maps global linear `blockIndex -> activeBlockId` (or -1)
  - Required so a cell can find neighbor cells across block boundaries
- `unsigned char* d_dcCornerMasks`
  - Per packed-cell 8-bit sign mask of corners (same corner ordering as MC tables)
  - Also acts as a stable “surface cell?” flag: surface iff mask != `0x00` and != `0xFF`
- `unsigned int* d_dcCellVertexOffsets`
  - Scan output for compact **DC cell-vertex index** (per packed-cell)
- `ushort4* d_dcCellVertices`
  - Compact cell-vertex storage (quantized local-in-cell position):
  - Decode: `p = cellMin + (ushort.xyz / 65535) * cellSize`
- `short4* d_dcCellNormals`
  - Compact per-cell-vertex normal storage, snorm int16 in [-32767, 32767]
- `short4* d_dcCellNormalsTmp`
  - Temporary buffer for the smoothing pass (ping-pong)

Allocation strategy:
- DC buffers are allocated only if `technique == DualContouring`
- Allocations are sized with the same safety envelope as existing code (`maxVertices`, packed size derived from estimated active blocks)

---

## DC pipeline overview (GPU passes)

DC follows the same outer “sparse packed” structure as MC, but needs an additional “cell-vertex solve” step.

### Pass 0 — Scout active blocks (shared with MC)

Kernel:
- `scoutActiveBlocks(SDFGrid grid, float time)`

Behavior:
- Sample SDF at block center, mark active if `abs(d) <= radius`

### Pass 1 — Mark surface cells + store corner sign masks

Kernel:
- `dcMarkCells(SDFGrid grid, float time)`

Per packed-cell:
- Evaluate SDF at the 8 cell corners
- Store an 8-bit sign mask into `d_dcCornerMasks`
- Write `d_packetVertexCounts[cell] = 1` if surface cell else 0

### Scan 1 — Compact cell-vertex indices

Operation:
- CUB exclusive scan:
  - `d_packetVertexCounts -> d_dcCellVertexOffsets`

Result:
- Each surface cell gets a compact index `cellVertexIdx = d_dcCellVertexOffsets[cell]`

### Pass 2 — Solve DC cell vertices (+ store per-cell normals)

Kernel:
- `dcSolveCellVertices(SDFGrid grid, float time, unsigned int maxCellVertices)`

Per surface cell:
- Build Hermite constraints from sign-changing edges (12 edges)
  - Intersection point via the same robust interpolation style as MC (`safeEdgeT`)
  - Constraint normal from `computeNormal(intersectionPoint, ...)`
- Solve for cell vertex:
  - Conservative “feature preserve” gating:
    - Only use plane-intersection shortcuts when normal clusters are clearly separated
    - Otherwise fall back to a regularized QEF with centroid bias
  - Clamp to cell bounds
- Store:
  - `d_dcCellVertices[cellVertexIdx]` (quantized position)
  - `d_dcCellNormals[cellVertexIdx]`
    - Primary: normalized sum of Hermite normals
    - Orientation fix: aligned to SDF gradient at solved point

### Pass 2.5 — Adaptive normal smoothing (optional)

Kernel:
- `dcSmoothCellNormals(SDFGrid grid, float cosAngleThreshold)`

When enabled:
- Runs only if output normals are requested (`grid.d_normals != nullptr`)
- Uses **6-neighborhood (face-adjacent)** averaging of neighboring surface-cell normals
- Includes a neighbor only if `dot(nNbr, nSelf) >= cosAngleThreshold`
- Weighted contribution by alignment so near-threshold neighbors contribute less
- Writes to `d_dcCellNormalsTmp`, then the host swaps buffers

Runtime control:
- `dcNormalSmoothAngleDeg` converted to `cosAngleThreshold` on host
- Default: **30°**

### Pass 3 — Count emitted triangles (quads → 2 tris → 6 verts)

Kernel:
- `dcCountQuads(SDFGrid grid)`

Per surface cell:
- Consider the 3 “emanating” edges from corner 0 to avoid duplicates:
  - X edge (0–1), Y edge (0–3), Z edge (0–4)
- If the edge sign flips, attempt to assemble the quad’s 4 incident cells:
  - Uses `lookupPackedCell(...)` + `d_blockToActiveId` for cross-block neighbors
- If all 4 cells are surface cells, count +6 vertices
- Write `d_packetVertexCounts[cell] = numSoupVertices`

### Scan 2 — Output offsets for triangle soup

Operation:
- CUB exclusive scan:
  - `d_packetVertexCounts -> d_packetVertexOffsets`

### Pass 4 — Generate quads as triangle soup

Kernel:
- `dcGenerateQuads(SDFGrid grid, float time)`

For each emitted quad:
- Decode the 4 DC vertex positions (each vertex uses its own cell’s `cellMin`)
- Compute `quadCenter`
- Winding fix:
  - Compare triangle normal against `computeNormal(quadCenter, ...)`
  - If flipped, swap `v1 <-> v3`
  - Also swap the associated normal indices so vertex normals stay attached to the correct vertex
- Per-quad dominant primitive (for UV/ID/color):
  - `map(quadCenter, ...) -> dominantPrimID + color`
- UVs:
  - Convert quad vertices to dominant primitive local space
  - Compute seam-aware UVs via `computeQuadUVs()` (built from two `computeTriangleUVs()` calls)
- Normals:
  - Use per-vertex normals from `d_dcCellNormals` (smoothed or raw)
- Emit 2 triangles:
  - `(v0, v1, v2)` and `(v0, v2, v3)`
  - Write vertex soup + optional `d_indices` as identity

---

## Notes on quality + known limitations

### Why subtract boundaries still look “edgy”

The CSG **subtract** boundary is typically **non-differentiable** in the combined SDF field.
Even with DC, any SDF-gradient-based normal will jump across that seam.

Additionally, the current “dominant primitive” rule for UV/ID/color is sampled at the quad center. At subtract seams, “dominant primitive for shading/mapping” may not match the user’s desired artistic rule (e.g., cutter interior mapping).

This is intentionally deferred to a follow-up design decision.

### Why increasing grid resolution helps

Both DC geometry and gradient/constraint normals are derived from:
- corner sampling
- edge intersection interpolation
- finite-difference gradients (`computeNormal`)

All improve with smaller `cellSize` / higher grid resolution.

---

## Summary of key kernels (DC)

- `buildBlockToActiveMap` — build neighbor lookup map
- `dcMarkCells` — corner mask + surface-cell marking
- `dcSolveCellVertices` — Hermite sampling + feature-gated solve + store vertex + store normal
- `dcSmoothCellNormals` — adaptive smoothing (6-neighbor, angle gated)
- `dcCountQuads` — count soup output vertices
- `dcGenerateQuads` — emit triangles + UV/ID/color/normals



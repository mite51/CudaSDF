## CUDA UV Atlas Packer (Aligned-X, Quantized Grid) — Implementation Notes

### Purpose
This document describes the **current implementation** of the CUDA UV atlas packer integrated into this project. It repacks existing per-primitive UV “charts” (generated during marching cubes) into a **single UV atlas** suitable for projection baking.

### User-facing behavior
- **`V`**: Toggle UV generation (must be ON to pack)
- **`A`**: Run the CUDA atlas packer and **remap UVs in-place** into \([0,1]\) atlas space
- **`P`**: Projection bake (works after `U` unwrap *or* after `A` atlas pack)

Packing runs on the current CUDA/GL interop buffers and freezes animation after success (so UVs remain stable for projection tests).

---

## Implementation overview

### High-level pipeline
1. **Generate per-vertex primitive UVs** during marching cubes (GPU).
2. **Extract UV charts** by grouping triangles by **primitive index**.
3. **Quantize each chart** onto a global grid (cells per UV unit), producing a **bitmask footprint**.
4. Run many **parallel packing attempts** on GPU.
5. Select the **best attempt** on CPU (most islands placed, then best fitness).
6. **Remap UVs** in-place using the best placement, producing atlas UVs in \([0,1]\).

---

## Key code locations

### Entry point (keypress integration)
- `src/main.cpp`
  - `GLFW_KEY_A`: triggers packing
  - Uses CUDA/GL interop pointers for `vbo`, `uvbo`, and the “primitive id” attribute buffer
  - Key config knobs are set near the `A` handler (see “Tuning knobs” below)

### UV generation + primitive ownership
- `src/SDFMesh/MarchingCubesKernels.cu`
  - Writes `grid.d_uvCoords` per vertex (primitive UV mapping)
  - Writes “primitive id” (actually **primitive index**) into the `primidbo` attribute buffer (stored as float bits)

### Shader mapping (direct vs atlas)
- `src/SDFMesh/CudaSDFUtil.h`
  - Direct texture-array mode uses **primitive.textureID** as the array layer
  - Atlas/single-texture mode uses a regular `sampler2D` and the packed UVs

### Packer implementation
- `src/SDFMesh/GridUVPacker.cuh`
- `src/SDFMesh/GridUVPacker.cu`

### Build integration
- `CMakeLists.txt`
  - Includes `src/SDFMesh/GridUVPacker.cu` in build sources.

---

## Data model

### Charts
Charts are extracted by grouping triangle soup triangles by the per-vertex primitive index:
- A “chart” == “all triangles whose vertices came from primitive N”.

Implementation:
- `GridUVPacker::ExtractCharts(...)` downloads `uvCoords` and primitive IDs (as float bits) and groups triangles by primitive.

### Quantized island masks
The packer uses **one global quantization rate**:
- `gridResolution` = cells per UV unit (e.g. 64)

For each chart:
- `minCell = floor(minUV * gridResolution) - gutter`
- `maxCell = ceil(maxUV * gridResolution) + gutter`
- `islandWidth/Height = maxCell - minCell`

Then triangles are rasterized into a **bit-packed grid**:
- One row is `stride = ceil(width/32)` `uint32_t` words.

Implementation:
- `GridUVPacker::CreateIslands(...)`
- CUDA rasterization kernel: `rasterizeIslandKernel(...)`

---

## GPU packing algorithm (v1)

### Execution model
- **One CUDA block = one packing attempt** (`packSolutionKernel`)
- Many attempts execute in parallel; each attempt has:
  - Its own occupancy bitmask (`Solution::occupancyMask`)
  - RNG seed / state
  - Per-island placement results

### Placement search (per island)
Each attempt:
- Optionally **shuffles island order per attempt** (`shuffleOrderPerAttempt`)
- For each island:
  - Try rotations (v1 uses **0° and 90°**)
  - Generate multiple **aligned-X candidate columns**
  - For each candidate:
    - Find a collision-free start Y near current top
    - “Drop” downward to lowest valid Y (capped by `maxDropSteps` if non-zero)
  - Pick the best candidate by a simple score based primarily on bounding area.

### Attempt selection
On CPU:
- Choose the attempt that places the most islands.
- Tie-break by attempt “fitness”.

---

## Correctness & robustness measures

### Conservative rasterization
Triangle-to-cell overlap uses:
- vertex-in-cell
- corner-in-triangle
- **edge-edge intersection** (added to prevent thin-slit misses)
- centroid-in-cell (cheap fallback)

This reduces the chance that the island mask under-represents triangle coverage, which can cause visible overlaps even if bitmasks “don’t collide”.

### Stride-correct bit addressing
Bit addressing uses `row * stride + wordIndex`, not `row * width`, to avoid corruption for widths not divisible by 32.

---

## Performance optimizations (current)

### Fast collision checks (bitwise)
Packing collision checks are performed via **word-wise bitmask AND** against the occupancy grid:
- Uses fast 32-bit ops instead of per-cell probing.

### Precomputed 90° masks
Rotation checks use precomputed:
- `collisionMask` (rot0)
- `collisionMask90` (rot90)

So rotation does not require per-bit coordinate transforms during packing.

### Per-row active word bounds
For each collision mask row, we precompute:
- first non-zero word
- last non-zero word

Collision/write loops only scan the active word range for that row.

---

## Tuning knobs
Configured at the `A` key handler in `src/main.cpp`:

- **`gridSize`**: board size in cells (e.g. 2048)
- **`gridResolution`**: quantization (cells per UV unit; e.g. 64)
- **`marginCells`**: gutter around islands (cells)
- **`xAlignment`**: aligned-X constraint (cells; e.g. 8). Larger = faster but less flexible.
- **`maxIterations`**: number of attempts (capped internally)
- **`maxCandidatesPerIsland`**: number of candidate X positions per island per attempt
- **`maxDropSteps`**: downward search cap (0 = no cap)
- **`shuffleOrderPerAttempt`**: whether attempts vary island order

Practical guidance:
- If you see overlaps: increase `gridResolution` modestly or keep rasterization conservative.
- If packing is slow: lower `maxIterations` and/or `maxCandidatesPerIsland`, increase `xAlignment`, or cap `maxDropSteps`.

---

## Known limitations / v1 scope
- Charts are **per primitive**, not general mesh UV islands (no seam detection/connected components).
- Rotation is limited to **0° / 90°**.
- Current scoring is simple (primarily bounding-box area); no explicit perimeter/contact heuristic beyond what drop-search implies.



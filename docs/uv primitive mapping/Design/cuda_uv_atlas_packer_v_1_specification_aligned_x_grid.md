# CUDA UV Atlas Packer – v1 Specification (Aligned-X Grid)

## 0. Purpose

Design a **CUDA-based UV atlas packer** that packs pre-existing UV islands into a single atlas while:

- Preserving **relative texel density** (no per-island scaling)
- Allowing **uniform global scaling only** at the end
- Supporting **0° / 90° rotation** per island
- Using a **grid-quantized, bitmask-based representation** suitable for fast GPU overlap tests
- Employing **multiple stochastic packing attempts** evaluated in parallel

This v1 spec intentionally constrains placement to **aligned X positions** (multiples of 4 or 8 grid cells) to greatly simplify bitmask shifting and reduce kernel complexity.

---

## 1. Key Design Decisions (v1)

- **Input islands already exist** (no connected-component detection)
- **No per-island UV normalization** — relative UV scale encodes texel density
- **Quantization-driven sizing**: island size determined by grid resolution
- **Uniform global scale** applied after packing
- **Aligned X placement**: `posX % X_ALIGNMENT == 0`
- **Fixed maximum board dimensions** (guardrails for memory)
- **0° / 90° rotation only**
- **Fragmentation discouraged via perimeter/contact heuristics** (not full hole detection)

---

## 2. Configuration

```cpp
struct AtlasPackerConfig {
  int gridResolution;          // cells per UV unit (e.g. 1024–4096)

  int maxBoardW;               // max board width in cells
  int maxBoardH;               // max board height in cells

  int gutterCells;             // padding around islands (cells)

  bool allowRotate90;          // true

  int numAttempts;             // e.g. 256–4096
  uint32_t rngSeed;

  int maxCandidatesPerIsland;  // e.g. 32–64
  int maxSearchDrop;           // vertical scan cap per candidate

  int xAlignment;              // 4 or 8 (cells)

  // Scoring weights
  float wArea;                 // typically 1.0
  float wWaste;                // e.g. 0.1–0.3
  float wPerimeter;            // e.g. 0.05
  float wContact;              // e.g. -0.05 (reward contact)
};
```

---

## 3. Data Model

### 3.1 Input Geometry (device)

```cpp
struct UVTri {
  float2 uv0, uv1, uv2;
};

struct IslandRange {
  int triStart;
  int triCount;
};
```

- Triangles are grouped by island via `IslandRange`
- UVs may be outside [0,1]

---

## 4. Island Grid Representation

### 4.1 Grid Metadata

```cpp
struct IslandGridMeta {
  int width;        // in cells (with gutter)
  int height;       // in cells (with gutter)
  int originX;      // grid-space origin (cell coords)
  int originY;
  int areaCells;    // filled after rasterization
  int perimeter;    // precomputed (approx)
  int bitRowsOffset;
};
```

### 4.2 Bitmask Storage

- Each row stored as `uint32_t` words
- `wordsPerRow = ceil(width / 32)`

```cpp
uint32_t* d_islandBits;  // packed row bitmasks
```

### 4.3 Rotations

```cpp
struct IslandFootprint {
  IslandGridMeta rot0;
  IslandGridMeta rot90;
};
```

- Both rotations are precomputed for simplicity

---

## 5. GPU Preprocessing Pipeline

### 5.1 Compute Island UV Bounds

- One CUDA block per island
- Reduce min/max UV over island triangles

Output:
```cpp
struct IslandBounds {
  float2 minUV;
  float2 maxUV;
};
```

---

### 5.2 Compute Grid Extents

For each island:

- `minCell = floor(minUV * R) - gutter`
- `maxCell = ceil(maxUV * R) + gutter`

```cpp
width  = maxCell.x - minCell.x;
height = maxCell.y - minCell.y;
```

Compute:
- `wordsNeeded = height * wordsPerRow`

Use GPU prefix-sum (CUB) to allocate packed bit storage.

---

### 5.3 Rasterize Triangles to Grid

- One block per island
- Threads rasterize triangles conservatively into grid cells
- Cell center point-in-triangle test
- Bits set via `atomicOr`

Result:
- Island footprint bitmask
- `areaCells` via popcount reduction

---

### 5.4 Perimeter Estimation (Precompute)

Approximate island perimeter:

```
perimeter ≈ 4 * areaCells - 2 * internalAdjacency
```

Internal adjacency is computed once per island using row bit adjacency checks.

---

### 5.5 90° Rotation Kernel

- Swap width/height
- Map `(x,y) → (y, W-1-x)`
- Recompute area/perimeter (or reuse area)

---

## 6. Packing Stage (GPU)

### 6.1 Execution Model

- **One CUDA block = one packing attempt**
- All attempts run in parallel

Each block maintains:
- Its own board bitmask
- RNG state
- Placement results

---

### 6.2 Board Representation

```cpp
uint32_t* d_attemptBoards;
// Layout: [attempt][row][word]
```

- Board width fixed to `maxBoardW`
- Height capped at `maxBoardH`
- X placement restricted to `xAlignment` multiples

---

### 6.3 Island Order

- Islands sorted once (CPU or GPU):
  - descending `areaCells`
  - tie-break by `max(width,height)`

```cpp
int* d_islandOrder;
```

---

### 6.4 Candidate X Selection (Aligned)

Candidate X positions:

- `0`
- `alignDown(usedW - islandW)`
- transitions from early board rows (edge detection)
- random aligned columns

All X satisfy:
```
(x % xAlignment) == 0
```

---

### 6.5 Y Placement (Drop + Cave Fill)

For each candidate X:

1. Start near current top (`usedH`)
2. Drop downward until no overlap
3. Continue pushing downward while overlap-free (cave filling)
4. Stop at first collision or bottom

Y scan capped by `maxSearchDrop`.

---

### 6.6 Overlap Test (Aligned-X)

Because X is aligned:
- No cross-word bit shifts
- Only whole-word shifts needed

Overlap test:

```cpp
for each row r:
  if (islandRow[word + shift] & boardRow[word]) != 0 → overlap
```

---

### 6.7 Fragmentation / Compactness Heuristic

For candidate placement compute:

- `contactEdges`: island edges touching existing board
- Approx effective perimeter:

```
effectivePerimeter = islandPerimeter - 2 * contactEdges
```

---

### 6.8 Scoring Function

```cpp
score =
  wArea  * (newW * newH)
+ wWaste * ((newW * newH) - (sumArea + islandArea))
+ wPerimeter * effectivePerimeter
+ wContact   * contactEdges;
```

Lowest score wins.

---

### 6.9 Commit Placement

Once best candidate chosen:

- OR island bitmask into board
- Update `usedW`, `usedH`
- Record per-island placement

```cpp
struct IslandPlacement {
  int posXCells;
  int posYCells;
  uint8_t rot90;
};
```

---

## 7. Attempt Reduction

- Reduce across attempts to find minimum score
- Retrieve best attempt index

---

## 8. Final UV Reconstruction (CPU or GPU)

### 8.1 Global Scale

```cpp
S = 1.0f / max(usedW, usedH);
```

### 8.2 Per-Vertex UV Transform

For each vertex UV in island:

1. Convert to grid space:
   ```cpp
   cell = uv * gridResolution - island.origin
   ```
2. Apply rotation (if rot90)
3. Add atlas offset `(posXCells, posYCells)`
4. Scale by `S`

Resulting UVs lie in `[0,1]`.

---

## 9. Outputs

- Per-island placement `(x, y, rotation)`
- Global atlas scale `S`
- Final packed UVs (applied downstream)

---

## 10. Explicit Non-Goals (v1)

- Arbitrary-angle rotation
- Dynamic board resizing beyond limits
- Perfect hole detection / region connectivity
- Sub-cell placement

---

## 11. Future Extensions (v2+)

- Adaptive X alignment
- Sparse / tiled board representation
- True hole detection via flood-fill
- Non-square atlas output
- Multi-atlas output

---

## 12. Summary

This design provides a **GPU-first, robust, debuggable UV atlas packer** that:

- Avoids fragile CPU algorithms
- Scales with attempts, not complexity
- Preserves texel density
- Produces high-quality atlases suitable for projection mapping

The aligned-X constraint dramatically simplifies v1 while preserving most packing quality and performance.


This is the final piece of the puzzle. You are absolutely right—Stage 3 is a diagnostic stopgap, while **Stage 4** is the actual production delivery mechanism.

I have updated the documentation package below. It now explicitly frames Stage 3 as "Intermediate Validation" and adds a new **Stage 4: GPU-Based UV Bin Packing** using a robust "Sort & Shelf" algorithm that fits the parallel architecture perfectly.

---

# Product Requirements & Technical Design: CUDA Wavefront UV Unwrap (V1)

## Part 1: Product Requirements Document (PRD)

### 1. Introduction

**Product:** CUDA-based Wavefront UV Unwrapping Library
**Purpose:** To provide a robust, high-performance GPU-native solution for generating non-overlapping UV coordinates for arbitrary triangle meshes (specifically targeting Marching Cubes "soup" output).
**Core Philosophy:** "Correctness first, Performance second." The pipeline prioritizes strict validity (no overlaps, flush edges) using deterministic data structures, avoiding "probabilistic" infinite loops or heavy host-synchronization.

### 2. Goals & Objectives

* **Robustness:** 100% valid UVs for standard closed meshes. Zero overlaps within islands.
* **Performance:** Fully GPU-resident pipeline. Minimized Host  Device transfers.
* **Scalability:** Efficiently handle ~1M+ triangles using bounded memory structures (Paged Grids).
* **Validation:** Intermediate "Rough Packing" allows visual verification before final packing.

### 3. Functional Requirements

#### 3.1 Data Model & Inputs

* **Input:** Triangle Soup (`float3* pos`, `int count`). No shared vertex indices assumed.
* **Coordinate Space:**
* **3D World:** Used for welding and edge lengths.
* **2D UV:** Calculated in **Local Island Space** (re-centered relative to seed) to maintain floating-point precision during growth.



#### 3.2 Stage 0: Vertex Welding & Remap (Static Broadphase)

* **Goal:** Convert "Triangle Soup" into a connected graph.
* **Algorithm:** Radix Sort & Scan (CUB).
* Quantize 3D positions to integer grid keys.
* Sort keys to group co-located vertices.
* Scan groups to assign a unique `WeldedVertexID`.


* **Output:** `int* cornerToVertID` (Map: Corner Index  Welded Vertex ID).

#### 3.3 Stage 1: Adjacency Build

* **Goal:** Identify neighbors without  comparisons.
* **Algorithm:** Edge Sort.
* Emit 3 undirected edge keys per triangle: `(min(vA, vB), max(vA, vB))`.
* Radix Sort edges.
* Run-length encode to find pairs.


* **Constraints:**
* 2 edges = Neighbor.
* 1 edge = Boundary.
*  edges = Non-Manifold (Mark as Boundary/Invalid for growth).



#### 3.4 Stage 2: Island Growth (The Core)

* **Execution Model:** Device-driven. Persistent kernel logic using Cooperative Groups or Global Barriers to minimize CPU sync.
* **Logic:** Breadth-First Search (BFS) propagation from Seed Triangles.

**3.4.1 Neighbor Placement**

* **Edge Alignment:** MUST detect edge direction.
* Compare `(vA, vB)` of parent vs. neighbor.
* If `Reversed` (typical): Swap UV endpoints before calculating the 3rd vertex.
* **Result:** Shared edges must be geometrically "flush" (identical coordinates).



**3.4.2 Dynamic Broadphase: The "Paged Chunk Grid"**

* **Scope:** Per-Island (reset after each island completes).
* **Structure:**
1. **Grid Head:** 2D Array covering the island's bounding box.
2. **Chunk Heap:** Large linear buffer of `Nodes` (Linked Lists).
3. **Allocator:** Global atomic counter for `next_free_chunk_index`.


* **Insertion Logic:**
* Compute Cell ID for triangle.
* Check `GridHead` for a chunk with space.
* If none or full: Atomically allocate new Chunk ID, add to `GridHead` list.
* **Constraint:** If `GridHead` list is full (max chunks per cell reached), trigger "Cell Overflow" and defer/reject.



**3.4.3 Narrowphase & Validity**

* **Query:** Iterate through all Chunks listed in the target Cell(s).
* **Overlap Check:**
* **Strict Interior:** Point-in-triangle must be strictly internal ().
* **Segment Intersection:** Must strictly cross edges (exclude endpoints).
* **Goal:** Touching edges/vertices is VALID. Crossing is INVALID.


* **Retry Policy:** **Per-Connection (3-Bit Mask).**
* If connection via Edge A fails (overlap), mark "Edge A Blocked" on the candidate.
* Allow candidate to be re-discovered via Edge B or C later.



#### 3.5 Stage 3: Rough Packing (Intermediate Validation)

* **Goal:** Temporary visualization to verify Stage 2 unfolding quality.
* **Algorithm:** Simple non-overlapping arrangement (e.g., arrange islands in a grid by ID).
* **Lifecycle:** Results are **discarded** (or overwritten) after verification passes.

#### 3.6 Stage 4: UV Bin Packing (Production Output)

* **Goal:** Efficiently pack islands into a final  UV Atlas.
* **Algorithm:** **GPU Sort & Shelf Packing**.
1. Compute AABB for all islands.
2. Sort Islands by Height (descending).
3. Pack into "Shelves" (Rows) using prefix sums.


* **Constraint:** Must minimize wasted texture space while maintaining a configurable padding/margin.
* **Output:** Final global UV buffer normalized to .

### 4. Non-Functional Requirements

1. **Memory Safety:** No dynamic `malloc` inside kernels. All pools (Chunk Heap, Adjacency) are pre-allocated at startup based on mesh heuristics (e.g., 120% of triangle count).
2. **Tunability:** The following must be exposed as constants:
* `GRID_CELL_SIZE` (Local UV units).
* `CHUNK_CAPACITY` (Triangles per chunk).
* `MAX_CHUNKS_PER_CELL` (Depth of the Grid Head list).
* `EPSILON` (For geometric robust predicates).


3. **Observability:** Pipeline must emit a generic `Stats` struct back to Host:
* `TotalIslands`, `TotalRejections`, `CellOverflows`, `HeapUsage`.



---

## Part 2: Technical Implementation Design (IDD)

### 1. Core Data Structures (C++ / CUDA)

#### 1.1 The Paged Chunk Grid

Replaces hash maps with a **Fixed-Head, Linked-Chunk** system for  insertion.

```cpp
// Configuration Constants
constexpr int GRID_DIM = 1024;        // 1024x1024 cells 
constexpr int CHUNK_CAPACITY = 8;     // Triangles per chunk node
constexpr int MAX_CHUNKS_PER_CELL = 4;// Limit to prevent worst-case scan

// 1. The Chunk Node (Stored in a linear heap)
struct GridChunk {
    int triangleIndices[CHUNK_CAPACITY]; // Payload
    int nextChunkIndex;                  // Pointer to next chunk (-1 if end)
    int count;                           // Number of items currently in this chunk
};

// 2. The Grid Header (The 2D Lookup Table)
struct GridCell {
    int headChunkIndex; // Pointer to the first chunk (-1 if empty)
};

// 3. The Allocator
struct GridAllocator {
    GridChunk* chunkHeap;      // Pre-allocated array
    int* nextFreeIndex;        // Global atomic counter
    GridCell* gridCells;       // The 2D grid array
};

```

#### 1.2 Triangle Adjacency & State

```cpp
struct TriangleState {
    int   islandId;      // -1 = unassigned
    uint8 connectionMask;// Bit 0=Edge0, Bit 1=Edge1, Bit 2=Edge2.
                         // 0 = Open, 1 = Blocked/Visited.
};

struct DerivedAdjacency {
    int3 neighborTriIds;     // Neighbors for edges 0, 1, 2
    int3 neighborEdgeIndices;// Which edge on the neighbor connects back to us
};

struct IslandInfo {
    int   seedTriIdx;
    float4 aabb;         // minX, minY, maxX, maxY
    float2 finalOffset;  // Calculated in Stage 4
};

```

### 2. Algorithms

#### 2.1 Grid Insertion Logic (Device)

**Strategy:** "Append-Only Linked List"

1. Compute `CellID`.
2. Access `GridCell[CellID]`.
3. **Atomic Loop:**
* Read `headChunkIndex`.
* If `-1`: Allocate new chunk, `atomicCAS` to replace head.
* If `>= 0`: Check `chunkHeap[head].count`.
* If `< Capacity`: `atomicAdd` count to reserve slot. Write.
* If `== Capacity`: Allocate new chunk. Set `newChunk.next = head`. `atomicCAS` to replace head with new chunk.





#### 2.2 Unfolding Logic (3D to 2D)

**Strategy:** "Circle-Circle Intersection with Basis Correction"

1. **Gather:** Parent UVs , 3D edge lengths .
2. **Reversal Check:**
* If Neighbor's edge vertices  match Parent's   **Reversed** (Standard).
* If match   **Aligned**.
* **Action:** If Reversed, swap inputs: Pivot = , Other = .


3. **Projection:**
* Calculate local coordinate  based on lengths.
* Construct basis vectors from Pivot  Other.
* Project  into UV space.



#### 2.3 Persistent Growth Kernel

**Strategy:** "Producer-Consumer with Warp Aggregation"

```cpp
__global__ void UnwrapPersistentKernel(...) {
    while(true) {
        // 1. Process Current Queue
        int idx = GetWorkItem(); // Warp-aggregated fetch
        if (idx < currentQueueCount) {
             int parent = currentQueue[idx];
             // ... Propagate neighbor, check overlap ...
             if (success) {
                 // 2. Enqueue Next (Warp-Aggregated)
                 unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
                 int leader = __ffs(mask) - 1;
                 int count = __popc(mask);
                 if (laneId == leader) base = atomicAdd(nextQueueCount, count);
                 nextQueue[base + laneOffset] = neighbor;
             }
        }
        
        // 3. Global Sync & Queue Swap
        grid.sync(); // Cooperative Groups recommended
        if (threadIdx.x==0 && blockIdx.x==0) {
            SwapQueues();
            if (*nextQueueCount == 0) *done = true;
        }
        grid.sync();
        if (*done) break;
    }
}

```

#### 2.4 Stage 4: GPU Shelf Packing

**Strategy:** "Sort-Based Shelf Binning"

1. **Compute AABBs:** Kernel iterates all islands, atomicMin/Max to find `IslandInfo[i].aabb`.
2. **Sort:** Radix sort islands by Height (descending).
3. **Shelf Logic (Device Kernel):**
* Maintain global `currentY` and `rowMaxH`.
* Iterate sorted islands.
* If `currentX + islandW > ATLAS_WIDTH`:
* New Row: `currentY += rowMaxH`, `currentX = 0`, `rowMaxH = 0`.


* Place Island: `finalOffset = {currentX, currentY}`.
* `currentX += islandW`.
* `rowMaxH = max(rowMaxH, islandH)`.


4. **Apply:** Final Kernel transforms all vertex UVs by `IslandInfo[islandID].finalOffset` and normalizes to .

### 3. Implementation Specifics & Edge Cases

1. **Coordinate Space & Bounds:**
* **Bounds:** `[-100.0, +100.0]` Local UV units.
* **Action:** If UV falls outside, **reject** (do not wrap/hash). Record `boundaryFailures`.


2. **Degenerate Geometry:**
* **Check:** Before processing edge, if `length(edge) < EPSILON`:
* **Action:** Mark edge `BLOCKED` in `connectionMask`. Do not traverse.


3. **Heap Overflow Safety:**
* **Check:** If `newChunkIndex >= HEAP_CAPACITY`:
* **Action:** `atomicExch(&errorFlag, 1)`. **Return immediately**. Host must detect and abort.


4. **Seed Selection:**
* **Heuristic:** `First Found`.
* **Optimization:** Maintain global `lastScannedIndex` to prevent re-scanning visited triangles.


5. **Host Orchestration:**
* **Requirement:** None during growth loop. Host only launches the Persistent Kernel and waits for completion.
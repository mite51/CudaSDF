# CUDA UV Unwrap v1 — Design Doc & Implementation Plan (Updated with Implementation Friction)

This document is the *revised* design, incorporating issues discovered during initial implementation attempts on large marching-cubes triangle soup meshes (~400K triangles).

**Inputs reality check:** In your current pipeline the mesh is a **triangle soup** (no shared vertex indices). That changes two “must-haves”:
1) you must build a robust **vertex welding / remap** step (or an equivalent edge-matching strategy), and  
2) edge-to-edge adjacency must handle **edge direction reversal** (A→B vs B→A), otherwise unfolding will fold back on itself. fileciteturn0file2L9-L21

---

## 0. Goals and non-goals

### Goals
- Fully GPU (CUDA) unwrap + pack.
- Island growth across adjacency.
- **Strict no-overlap within an island** during growth (reject → defer).
- Broadphase is **Option A**: sparse hashed grid → candidate triangles → **exact** overlap tests.
- Works reliably for marching cubes soup meshes at ~400K triangles.

### Non-goals (v1)
- ABF/LSCM global solves, seam optimization.
- Perfect packing efficiency.
- Perfect support for severely non-manifold geometry; we detect and treat carefully.

---

## 1. High-level pipeline

1. **Vertex remap (triangle soup → welded vertex IDs)** *(new, required)*
2. **Adjacency build (radix/sort-based; no O(N²))**
3. **Seed selection**
4. **Island growth**
   - shared-edge propagation with **edge order correction**
   - overlap checks with **boundary-safe exact tests**
5. **Packing**

---

## 2. Key friction points (what went wrong and how we prevent it)

### 2.1 Adjacency can’t be brute-forced; must be accelerated and stable
A naïve O(N²) edge match is both too slow and numerically unstable at 1.16M edges. It produced wildly varying “manifold” rates and blocked growth. fileciteturn0file0L107-L152  
A spatial-grid / hashed approach dropped adjacency time to milliseconds and achieved ~99.9% connectivity. fileciteturn0file1L22-L33

**Design update:** adjacency is **radix/sort-based** on stable keys, with a spatial grid only as a helper where needed (e.g., triangle-soup vertex remap).

---

### 2.2 Triangle soup needs edge direction handling (forward vs reversed)
Shared edges in soup are duplicated (A,B) and (A',B') with unknown ordering. If you always assume the same order, unfolded neighbors fold back and overlap. Fix was to detect forward vs reversed and swap UV endpoints accordingly. fileciteturn0file2L9-L27

**Design update:** in the growth step, every “expand across edge” operation must:
- determine whether the neighbor’s shared edge matches **forward** or **reversed**, and
- reorder (`uA,uB`) used for placement so the shared edge is *flush* in UV.

---

### 2.3 Overlap checks falsely rejected valid growth due to boundary-inclusive tests
Exact overlap routines that treat boundary contact as overlap will reject triangles that share an edge (the most common “overlap” in a valid unwrap). This was a major blocker until tests were changed to **strict interior** (exclude boundary/endpoints). fileciteturn0file2L45-L76

**Design update:** exact overlap tests must:
- use **strict interior** containment (`u > eps`, `v > eps`, `u+v < 1-eps`) and
- use strict segment intersection (exclude endpoint touches) with a consistent eps.

Additionally:
- Skip testing candidate vs its parent triangle (and optionally vs triangles that share the same edge).

---

### 2.4 CUDA memory initialization and init order are non-negotiable
`cudaMalloc` does not zero memory; uninitialized flags/IDs caused most triangles to be marked invalid or pre-assigned. fileciteturn0file0L68-L92  
Also, init order mattered (state must be initialized before adjacency processing). fileciteturn0file0L84-L92

**Design update:** explicitly define an initialization phase:
- `cudaMemset(d_triFlags, 0, ...)`
- `cudaMemset(d_triIslandId, 0xFF, ...)` to set `-1`
- run `initTriangleState()` before any kernel that expects initialized state.

---

### 2.5 Work queue race conditions (GPU queue semantics)
A lock-free queue without careful snapshotting allowed head to overrun tail. The fix was “snapshot batch processing”: launch exactly `tail-head` work items each iteration. fileciteturn0file0L54-L67

**Design update:** for v1, keep island growth sequential and queue processing snapshot-based.

---

### 2.6 “Manifold data” is not needed for v1
Early passes tried to build/track manifold-ness beyond what is required, which increased complexity. (You only need adjacency + the ability to treat non-manifold edges as boundary/invalid in growth.)

**Design update:** adjacency output is:
- neighbor triangle id per edge: `>=0`, `-1 boundary`, `-2 nonmanifold/invalid`
- optional per-triangle flags for debug stats only

No extra manifold structures unless needed for later quality improvements.

---

## 3. Data model (GPU)

### Inputs
- Triangle soup positions: `float3* d_posSoup` length `numTris*3`
  - triangle t corners at `3*t + {0,1,2}`

### Vertex remap (new)
- `int* d_cornerToVert` length `numTris*3` (welded vertex id per corner)
- `float3* d_verts` length `numVertsWelded` (optional; or reuse representative positions)
- *You can keep everything corner-based; vertex ids are mainly for adjacency.*

### Adjacency
- `int3 d_triNbrTri` (per tri edge neighbor)
- `uchar3 d_triNbrEdge` (which edge on neighbor corresponds)

### State
- `int* d_triIslandId`  (-1 unassigned, >=0 island, -2 deferred, -3 invalid)
- `uint8* d_triFlags`

### UVs
- `float2* d_cornerUV` length `numTris*3`

---

## 4. Stage 0 (NEW): Vertex remap for triangle soup

### Why
To get stable adjacency keys, you need stable vertex identifiers. For marching cubes soup, “same vertex” appears multiple times with tiny floating differences.

### Approach (CUDA-friendly)
1. Quantize positions to a 3D grid:
   - `q = floor((p - origin)/cellSize)`
   - key = morton(q) or packed 64-bit
2. Sort corners by key (radix sort).
3. Within each key-group, perform epsilon clustering:
   - compare positions to assign a welded vertex id
   - output `cornerToVert[corner] = weldedId`

**Notes**
- This step replaces the earlier “hash collisions” failure mode and makes adjacency deterministic. fileciteturn0file0L42-L53
- cellSize should be tied to marching cubes resolution (or estimated from median edge length in 3D).

---

## 5. Stage 1: Adjacency build (radix/sort)

Now that every corner has a welded vertex id:

1. For each triangle, emit 3 directed edges:
   - edge e0: (v0,v1), e1: (v1,v2), e2: (v2,v0)
2. Canonical undirected edge key:
   - `lo=min(va,vb), hi=max(va,vb)`
   - `key=(uint64(lo)<<32)|hi`
3. Sort edges by `key` (CUB).
4. Scan groups:
   - size 1 → boundary
   - size 2 → neighbors
   - size >2 → non-manifold; mark neighbor=-2 (treat as boundary/invalid)

**Important:** in triangle soup, even with welded vertex ids, directed order may differ. That’s fine for adjacency; direction handling is done at growth-time.

---

## 6. Island growth (strict no-overlap) — updated details

### 6.1 Seed placement
Seed triangle UV initialization:
- Use a local 2D basis from its 3D triangle plane
- Assign consistent scale (e.g., make one edge length in UV equal to its 3D length, or normalized to 1.0)

### 6.2 Shared-edge propagation must be “flush”
When processing a work item: (parent tri `P`, edge `eP`)
1. Find neighbor tri `N` from adjacency.
2. Identify shared edge endpoints on P:
   - welded vertex ids `(vA, vB)` for P’s edge
   - corresponding corner UVs `(uA, uB)` in P
3. Identify which two corners of N correspond to `(vA, vB)`
4. Determine ordering:
   - if N’s edge corners match (vA,vB) → forward
   - if match (vB,vA) → reversed
5. For placement, ensure the UV pair passed into placement is ordered to match N’s corner order:
   - if reversed, swap `uA,uB` before computing `uC`

This is the exact issue that blocked growth earlier. fileciteturn0file2L9-L27

### 6.3 Overlap check must ignore boundary contact
Exact tests must **exclude boundary** to avoid false positives from shared edges. fileciteturn0file2L45-L76

Policy:
- Skip testing candidate vs its parent triangle
- Use strict interior containment and strict segment intersection
- Use one eps definition consistently (recommend start at `1e-6` to `1e-5` depending on UV scale; tune)

### 6.4 Broadphase Option A (sparse hash buckets)
Keep broadphase as in the original plan, but add these safeguards:

- **cellSize selection**: base on *measured UV edge length* from early accepted triangles (this was necessary; hardcoded 0.1 was far too coarse). fileciteturn0file1L60-L69
- On bucket overflow during query: treat as “unsafe” and **defer** (don’t accept without confidence).
- Expect collisions in hash buckets; correctness is preserved because you do exact tests afterward.

---

## 7. Validation gates (make problems obvious early)

The debugging handoff emphasizes validating each stage before moving on. fileciteturn0file0L154-L214

### Gate A: After vertex remap
- % corners mapped (should be 100%)
- distribution of welded vertex reuse (sanity)
- spot check that the same spatial vertex gets same id across triangles

### Gate B: After adjacency
- sample triangles: neighbors not all -1/-2
- connectivity ratio should be high for closed MC surfaces (~99%+) fileciteturn0file1L22-L33

### Gate C: After first island growth
- island grows beyond 2–3 triangles; if not, print rejection reasons
- ensure shared-edge reversal logic is exercised

### Gate D: Overlap false positives
- explicitly test two triangles sharing an edge should **not** be rejected as overlap (regression test from the boundary-inclusive bug) fileciteturn0file2L45-L76

---

## 8. Implementation plan updates (only the changed chunks)

### New Chunk 0 — Vertex remap (triangle soup → welded IDs)
**Cursor prompt**
> Implement a GPU vertex remap for triangle soup. Quantize corner positions into a 3D grid (cellSize configurable). Create records (key, cornerId, position). Radix sort by key. Within each key group, cluster by epsilon to assign welded vertex IDs. Output `cornerToVert[cornerId]`. Add tests: a mesh with duplicated vertices with tiny jitter should map to the same welded id; distinct vertices should not merge.

### Updated Chunk 2 — Adjacency uses welded vertex ids + radix sort
**Cursor prompt**
> Build adjacency using welded vertex IDs: emit 3 edges per tri, key undirected (min,max), radix sort by key, link groups of size 2, mark boundaries, flag non-manifold (>2). Do not do O(N²) comparisons.

### Updated Chunk 6 — Growth must implement shared-edge direction fix
**Cursor prompt**
> In growth, when expanding across a shared edge, determine whether neighbor edge is forward or reversed by comparing welded vertex IDs of the shared edge corners. If reversed, swap the UV endpoints before placing the third vertex so the shared edge is flush in UV. Add a unit test on a two-triangle quad verifying the new triangle’s UV is placed correctly.

### Updated overlap tests — boundary-exclusive
**Cursor prompt**
> Update overlap detection so shared-edge contact is not treated as overlap: use strict interior point-in-triangle and strict segment intersection (exclude endpoints) with consistent eps. Add a regression test: two adjacent triangles sharing an edge should not be rejected.

---

## 9. Tuning notes (based on the sessions)

- If growth stops at 2–3 triangles, first suspect:
  1) shared edge forward/reversed mismatch fileciteturn0file2L9-L27
  2) boundary-inclusive overlap tests fileciteturn0file2L45-L76
- Keep `maxIterationsPerIsland` sane (e.g., 1k–10k); extremely high defaults can mask stalls and tank performance. fileciteturn0file2L88-L101
- Prefer snapshot queue processing to avoid head/tail races. fileciteturn0file0L54-L67

---

## 10. Summary of the “avoid repeat pain” changes

1) **Add Vertex Remap stage** for triangle soup (weld IDs)  
2) **Adjacency is radix/sort** on welded edge keys (no brute force)  
3) **Growth handles edge reversal** by swapping UV endpoints as needed  
4) **Overlap tests are boundary-exclusive** (strict interior)  
5) **Explicit CUDA initialization and validation gates** after each stage  

These directly address the friction points that blocked progress or created confusing false failures. fileciteturn0file0L154-L214 fileciteturn0file1L68-L83 fileciteturn0file2L45-L76

# Particle Simulation Optimization

## TODO
- [ ] **Profiling & Measurement**
    - [x] Run with realistic particle count.
    - [ ] Measure time spent on:
        - [ ] Map rebuilding
        - [ ] Neighbor iteration (`rep_force` calls)
        - [ ] Force evaluation (kernel)
    - [ ] Use tools: `cProfile`, `pyinstrument`, `line_profiler`.
    - [ ] Record metrics: seconds/update, particles/sec, avg neighbors/particle.
    - In progress using cprofile/pstats (`Profiling.py`)
    - Currently justs measures total time, cumulative time, 
    and time per call for each function

- [ ] **Data Layout & Python Overhead**
    - [x] Maintain Structure of Arrays (SoA) with NumPy.
        - Most of the data is process through numpy 
    - [ ] Avoid Python loops in hot inner loops.
        - Made maps preprocess which maps are neighbouring and turned store
        them in a dict
    - [ ] Consider Numba JIT or C/C++/Cython for speed.
        - Long term goal - to rewrite in c/c++
    - [ ] Minimize Python object access in loops.
        - in progress

- [ ] **Grid Construction & Updates**
    - [ ] Consider incremental `pos_map` updates per particle.
    - [ ] Tune `map_update_freq` based on displacement and cell size.
    - [ ] Use flat arrays and counts for neighbor indices.
    - [ ] Preallocate per-cell buffers if possible.
    - [ ] Prefer integer cell indices and NumPy-based buckets.

- [ ] **Reduce Redundant Pair Evaluations**
    - [ ] Avoid double-counting pairs (use `j > i` ordering).
    - [ ] Implement per-cell iteration with index ordering.

- [ ] **Cell Size & Mapping Choices**
    - [ ] Set cell size ≈ kernel radius (`force_radius`).
    - [ ] Consider adaptive grid/quadtree for non-uniform density.

- [ ] **Neighbor Lists & Verlet Lists**
    - [ ] Build Verlet lists if particles move slowly.
    - [ ] Update lists infrequently (every K frames or on large moves).

- [ ] **Minimize Python Function Call Overhead**
    - [ ] Inline `rep_force` logic in compiled function (Numba/Cython/C++).
    - [ ] Localize variables and minimize attribute lookups.

- [ ] **Flat-Indexing & Precomputed Neighbor Offsets**
    - [ ] Store neighbor cell offsets as small arrays.
    - [ ] Use integer arithmetic and branchless clamping for boundaries.

- [ ] **Concurrency & Parallelism**
    - [ ] Use Numba `prange` or C/C++ with OpenMP for multi-threading.
    - [ ] Consider GPU kernels (CUDA/OpenCL) for large N.

- [ ] **Memory & Numeric Choices**
    - [ ] Use `float32` for arrays if precision allows.
    - [ ] Reuse temporary arrays; avoid per-frame allocations.

- [ ] **Edge Cases & Correctness Checks**
    - [ ] Clamp cell indices at boundaries.
    - [ ] Ensure force symmetry and consistency after updates.

- [ ] **Step-by-Step Plan**
    - [ ] Profile to find hotspots.
    - [ ] Optimize map rebuilds and neighbor loops.
    - [ ] Eliminate double-counting.
    - [ ] Replace inner loops with compiled kernels.
    - [ ] Optimize memory allocations.
    - [ ] Consider GPU acceleration if needed.

## Known problems

## Dev ideas
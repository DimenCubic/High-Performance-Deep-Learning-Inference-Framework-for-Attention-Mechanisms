# CUDA GEMM Performance Analysis

## 1. Scope

This report analyzes the current CUDA GEMM implementations:

- **CUDA Naive GEMM**
- **CUDA Reordered / Shared-Scalar GEMM**
- **CUDA Unrolled GEMM**

The CUDA benchmarks were measured on the current NVIDIA GPU server using CUDA events, with:

- 10 warm-up runs
- 100 benchmark runs
- kernel execution time only
- FP32 GEMM
- matrix sizes from `128 × 128` to `4096 × 4096`

The CPU comparison uses the existing Apple M2 GEMM benchmark results from the previous performance log. For each matching CPU implementation and matrix size, the **best recorded CPU runtime across the available compiler optimization levels** is used. This gives the CPU side a reasonably strong baseline rather than comparing CUDA only against `-O0`.

> Note: CPU and GPU timings are not an architecture-neutral comparison. They were measured on different hardware, compilers, execution models, and memory systems. The comparison is therefore useful for showing the magnitude and scaling behavior of GPU acceleration, but it should not be interpreted as an isolated measurement of "CUDA language speedup."

---

# 2. CUDA Benchmark Results

## 2.1 Runtime

| Matrix Size | CUDA Naive (ms) | CUDA Reordered / Shared-Scalar (ms) | CUDA Unrolled (ms) |
|---:|---:|---:|---:|
| 128 | 0.008223 | 0.022118 | 0.008888 |
| 256 | 0.028324 | 0.043582 | 0.028631 |
| 512 | 0.169134 | 0.268236 | 0.171366 |
| 1024 | 1.223130 | 1.931070 | 1.234260 |
| 2048 | 8.680390 | 17.843900 | 8.484080 |
| 4096 | 71.006600 | 247.184000 | 67.604200 |

## 2.2 Performance

| Matrix Size | CUDA Naive (GFLOPS) | CUDA Reordered / Shared-Scalar (GFLOPS) | CUDA Unrolled (GFLOPS) |
|---:|---:|---:|---:|
| 128 | 508.095 | 188.889 | 470.046 |
| 256 | 1182.370 | 768.415 | 1169.670 |
| 512 | 1585.570 | 999.764 | 1564.910 |
| 1024 | 1754.880 | 1111.530 | 1739.050 |
| 2048 | 1978.680 | 962.554 | 2024.460 |
| 4096 | 1935.340 | 555.952 | 2032.750 |

---

# 3. CUDA Horizontal Comparison

## 3.1 Relative Runtime

The table below uses CUDA Naive as the `1.00×` baseline.

A value greater than `1.00×` means the implementation takes **more time** than Naive.  
A value below `1.00×` means it is faster than Naive.

| Matrix Size | Naive | Reordered / Shared-Scalar | Unrolled |
|---:|---:|---:|---:|
| 128 | 1.00× | 2.69× | 1.08× |
| 256 | 1.00× | 1.54× | 1.01× |
| 512 | 1.00× | 1.59× | 1.01× |
| 1024 | 1.00× | 1.58× | 1.01× |
| 2048 | 1.00× | 2.06× | 0.98× |
| 4096 | 1.00× | 3.48× | 0.95× |

## 3.2 Relative GFLOPS

| Matrix Size | Reordered vs Naive | Unrolled vs Naive |
|---:|---:|---:|
| 128 | 37.18% of Naive | 92.51% of Naive |
| 256 | 64.99% of Naive | 98.93% of Naive |
| 512 | 63.05% of Naive | 98.70% of Naive |
| 1024 | 63.34% of Naive | 99.10% of Naive |
| 2048 | 48.65% of Naive | 102.31% of Naive |
| 4096 | 28.73% of Naive | 105.03% of Naive |

---

# 4. CUDA Scaling Analysis

## 4.1 Naive GEMM

### Observation

CUDA Naive performance increases strongly as the matrix gets larger:

- `128`: 508.095 GFLOPS
- `256`: 1182.370 GFLOPS
- `512`: 1585.570 GFLOPS
- `1024`: 1754.880 GFLOPS
- `2048`: 1978.680 GFLOPS
- `4096`: 1935.340 GFLOPS

The throughput grows rapidly at first and then approaches a plateau around `~2 TFLOPS`.

### Interpretation

For small matrices, there is not enough work to fully occupy the GPU for long enough. Fixed costs such as:

- kernel launch overhead,
- block scheduling,
- pipeline warm-up,
- limited number of active warps,

represent a larger fraction of the total runtime.

As `N` increases, the GPU has far more independent threads and arithmetic work available. This allows the GPU to use more of its SMs and hide memory latency more effectively.

The transition:

```text
508 GFLOPS
→ 1182 GFLOPS
→ 1586 GFLOPS
→ 1755 GFLOPS
→ 1979 GFLOPS
```

shows the GPU moving from an under-utilized regime toward a sustained-throughput regime.

The slight decrease from `1978.68 GFLOPS` at `N = 2048` to `1935.34 GFLOPS` at `N = 4096` suggests that the naive kernel has reached approximately its practical throughput ceiling under the current access pattern.

---

# 5. Why the Current "Reordered / Block-Tile" Version Becomes Slower

## 5.1 Important Terminology Correction

The current CUDA reordered implementation should **not** be described as a conventional 2D tiled GEMM.

Its structure is closer to:

```text
one block
→ one matrix row + a range of columns

thread 0
→ loads one A[row][k] scalar into shared memory

all threads
→ reuse the shared scalar

all threads
→ read different B[k][col] values
```

It uses shared memory, but it does **not** load an `A tile` and a `B tile` such as:

```text
A_tile[16][16]
B_tile[16][16]
```

Therefore, it is better described as:

> **Reordered / shared-scalar CUDA GEMM**

rather than a true shared-memory tiled GEMM.

A true tiled GEMM will be a separate optimization later.

---

## 5.2 The Core Problem: Two Block-Wide Synchronizations for Every `k`

The current reordered kernel conceptually performs:

```cpp
for (int k = 0; k < N; k++) {

    if (threadIdx.x == 0)
        shared_a = A[row * N + k];

    __syncthreads();

    sum += shared_a * B[k * N + col];

    __syncthreads();
}
```

For every single `k`, the entire block performs:

```text
load one scalar
↓
block-wide synchronization
↓
one multiply-add per thread
↓
block-wide synchronization
```

For `N = 1024`, this means approximately:

```text
2 × 1024 = 2048
```

block-wide synchronization points per block.

For `N = 4096`:

```text
2 × 4096 = 8192
```

synchronization points per block.

This overhead grows directly with `N`.

That explains why the relative penalty becomes especially severe at large matrix sizes.

---

## 5.3 Scaling Evidence

Compared with Naive:

| Matrix Size | Reordered Runtime / Naive Runtime |
|---:|---:|
| 128 | 2.69× |
| 256 | 1.54× |
| 512 | 1.59× |
| 1024 | 1.58× |
| 2048 | 2.06× |
| 4096 | 3.48× |

The important trend is at the large end:

```text
1024: 1.58× slower
2048: 2.06× slower
4096: 3.48× slower
```

The synchronization cost is becoming increasingly dominant.

At `N = 4096`:

- Naive: `71.01 ms`
- Reordered: `247.18 ms`

The reordered kernel therefore takes almost three and a half times as long.

---

## 5.4 Only One Thread Performs the Shared-Memory Load

Another problem is:

```cpp
if (threadIdx.x == 0)
{
    shared_a = A[row * N + k];
}
```

Only one thread in the block performs useful loading work for `A`.

The remaining threads wait.

Conceptually:

```text
Thread 0    → load A
Thread 1    → wait
Thread 2    → wait
...
Thread 255  → wait
```

Then all threads hit:

```cpp
__syncthreads();
```

This is very different from a true tiled GEMM, where all threads cooperate to load a whole tile:

```text
256 threads
↓
256 elements loaded cooperatively
↓
shared-memory tile
↓
many arithmetic operations reuse the tile
```

The current reordered version performs a block-wide synchronization after loading only **one float**.

The amount of reusable data obtained per synchronization is therefore extremely small.

---

# 6. Why Naive CUDA Is Already Better Than the CPU Naive Access Pattern Suggests

This is one of the most important CPU-vs-GPU differences in the experiment.

The CPU naive loop accesses:

```cpp
B[k * N + j]
```

while `k` changes and `j` stays fixed.

For a single CPU thread, this is a strided access through matrix `B`.

That is bad for spatial locality.

On the GPU, however, the execution mapping is different.

Adjacent CUDA threads have adjacent `col` values:

```text
Thread 0 → col
Thread 1 → col + 1
Thread 2 → col + 2
...
```

At a fixed `k`, a warp therefore accesses:

```text
B[k][col]
B[k][col+1]
B[k][col+2]
B[k][col+3]
...
```

These are contiguous addresses.

Therefore, even though each individual thread walks down `B` with a stride as `k` changes, the **warp-level memory access at each iteration is already coalesced**.

This means the GPU naive implementation does not suffer from exactly the same memory-layout problem as the CPU naive implementation.

That is why translating the CPU loop-reordering idea directly to CUDA does not automatically help.

---

# 7. Why the Reordered Version Does Not Gain Enough from Reusing `A`

The reordered implementation attempts to reduce repeated reads of:

```text
A[row][k]
```

by loading it once into shared memory.

This does reduce explicit global loads of `A` at source-code level.

However, the GPU hardware already has:

- L1 cache,
- L2 cache,
- memory broadcast behavior,
- warp scheduling,

and many threads request the same `A[row][k]` value.

The benefit of manually putting one scalar into shared memory is therefore limited.

Meanwhile, the implementation pays for:

```text
two __syncthreads()
for every k
```

The trade-off becomes:

```text
small reduction in A traffic
vs
very large synchronization overhead
```

The measured results show that the synchronization cost dominates.

---

# 8. Why This Is Not Yet True Shared-Memory Tiling

A conventional tiled GEMM works differently.

For example:

```text
16 × 16 A tile
+
16 × 16 B tile
```

are cooperatively loaded into shared memory.

One loading phase brings:

```text
256 A elements
+
256 B elements
```

into shared memory.

Those values are then reused for many multiply-add operations before the next synchronization.

Conceptually:

```text
Global Memory
↓
load large A/B tiles cooperatively
↓
__syncthreads()
↓
many FMAs using shared memory
↓
__syncthreads()
↓
next tile
```

If `TILE_SIZE = 16`, synchronization happens approximately:

```text
N / 16
```

times per phase rather than once per individual `k`.

For `N = 4096`:

Current shared-scalar design:

```text
4096 k iterations
× 2 barriers
= 8192 barriers
```

A 16-wide tiled implementation would require approximately:

```text
4096 / 16 = 256 tile iterations
× 2 barriers
= 512 barriers
```

That is roughly **16× fewer synchronization phases**, while each tile load provides far more reusable data.

This is why true tiling has a much stronger theoretical justification than the current shared-scalar implementation.

---

# 9. Unrolled CUDA vs Naive CUDA

## 9.1 Raw Comparison

| Matrix Size | Naive Runtime (ms) | Unrolled Runtime (ms) | Runtime Change | Naive GFLOPS | Unrolled GFLOPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.008223 | 0.008888 | 8.09% slower | 508.095 | 470.046 |
| 256 | 0.028324 | 0.028631 | 1.08% slower | 1182.370 | 1169.670 |
| 512 | 0.169134 | 0.171366 | 1.32% slower | 1585.570 | 1564.910 |
| 1024 | 1.223130 | 1.234260 | 0.91% slower | 1754.880 | 1739.050 |
| 2048 | 8.680390 | 8.484080 | 2.26% faster | 1978.680 | 2024.460 |
| 4096 | 71.006600 | 67.604200 | 4.79% faster | 1935.340 | 2032.750 |

---

## 9.2 Small Matrices

For `128–1024`, unrolling is slightly slower.

The difference is small:

```text
~1% around 256–1024
```

except for `N = 128`, where very small runtimes make fixed overhead and measurement noise more significant.

Possible reasons include:

- the CUDA compiler may already optimize the simple loop effectively,
- loop-control overhead is a very small fraction of total GEMM cost,
- the unrolled body contains more instructions,
- extra instructions can increase register or instruction pressure,
- the kernel is still dominated by memory access and the dependency chain in `sum`.

The unrolled version still uses one accumulator:

```cpp
sum += ...
sum += ...
sum += ...
sum += ...
```

These operations are dependent on the previous value of `sum`.

Therefore, manual unrolling does not create four completely independent FMA chains.

---

## 9.3 Large Matrices

At larger sizes, manual unrolling starts to provide a measurable benefit:

```text
N = 2048:
2024.46 GFLOPS
vs
1978.68 GFLOPS
≈ 2.31% gain

N = 4096:
2032.75 GFLOPS
vs
1935.34 GFLOPS
≈ 5.03% gain
```

The `4096 × 4096` result is especially useful because the workload is long enough that the small per-iteration reduction in loop-control overhead accumulates into a measurable difference.

The result suggests that unrolling has a real but modest benefit for this kernel.

It does not fundamentally change the memory-access behavior, so it should not be expected to produce the type of large speedup that a successful memory-reuse optimization could potentially provide.

---

# 10. CPU Baseline Used for Comparison

The CPU measurements were collected on:

```text
Apple M2
ARM64
macOS
Apple Clang
C++17
```

For each matching implementation below, the best recorded runtime among `O0`, `O1`, `O2`, `O3`, and `Ofast` is used.

## 10.1 CPU Best Recorded Runtime

| Matrix Size | CPU Naive Best (ms) | CPU Reordered Best (ms) | CPU Unrolled Best (ms) |
|---:|---:|---:|---:|
| 128 | 2.10771 | 0.149017 | 0.162133 |
| 256 | 10.32290 | 1.249200 | 1.236940 |
| 512 | 86.96560 | 11.672500 | 9.820590 |
| 1024 | 866.86200 | 78.991600 | 76.512200 |

## 10.2 CPU Best Recorded GFLOPS

| Matrix Size | CPU Naive Best | CPU Reordered Best | CPU Unrolled Best |
|---:|---:|---:|---:|
| 128 | 1.98221 | 28.03660 | 25.76840 |
| 256 | 3.24415 | 26.80830 | 27.07390 |
| 512 | 3.08367 | 22.97480 | 27.30720 |
| 1024 | 2.47610 | 27.17290 | 28.05350 |

---

# 11. CPU vs CUDA Comparison

## 11.1 Naive GEMM

| Matrix Size | CPU Runtime (ms) | CUDA Runtime (ms) | CUDA Speedup | CPU GFLOPS | CUDA GFLOPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.10771 | 0.008223 | 256.33× | 1.982 | 508.095 |
| 256 | 10.32290 | 0.028324 | 364.46× | 3.244 | 1182.370 |
| 512 | 86.96560 | 0.169134 | 514.18× | 3.084 | 1585.570 |
| 1024 | 866.86200 | 1.223130 | 708.72× | 2.476 | 1754.880 |

### Analysis

The speedup grows with matrix size:

```text
256×
→ 364×
→ 514×
→ 709×
```

This reflects the GPU's throughput-oriented architecture.

For small matrices, GPU resources are under-utilized. As the problem size increases, more SMs and warps remain active and the GPU can hide latency more effectively.

The CPU naive version also suffers strongly from the `B[k * N + j]` strided access pattern, while the CUDA thread mapping naturally produces coalesced accesses to adjacent `B` columns at warp level.

Therefore, CUDA benefits from both:

- massive thread-level parallelism,
- a much more favorable effective memory-access pattern across a warp.

---

## 11.2 Reordered CPU vs Current CUDA Reordered / Shared-Scalar

| Matrix Size | CPU Runtime (ms) | CUDA Runtime (ms) | CUDA Speedup | CPU GFLOPS | CUDA GFLOPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.149017 | 0.022118 | 6.74× | 28.037 | 188.889 |
| 256 | 1.249200 | 0.043582 | 28.66× | 26.808 | 768.415 |
| 512 | 11.672500 | 0.268236 | 43.52× | 22.975 | 999.764 |
| 1024 | 78.991600 | 1.931070 | 40.91× | 27.173 | 1111.530 |

### Analysis

CUDA is still substantially faster than the CPU reordered implementation because of GPU parallelism.

However, the GPU speedup is dramatically smaller than in the naive comparison.

This does **not** mean loop reordering is more suitable for the CPU in every sense. It means that the current CUDA translation introduces expensive synchronization that the CPU version does not have.

On the CPU, reordering improves spatial locality with relatively little structural overhead.

On the GPU, the current design adds:

```text
shared-memory scalar load
+
two block-wide barriers per k
```

The optimization mechanism therefore changes substantially between the two architectures.

---

## 11.3 Unrolled GEMM

| Matrix Size | CPU Runtime (ms) | CUDA Runtime (ms) | CUDA Speedup | CPU GFLOPS | CUDA GFLOPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.162133 | 0.008888 | 18.24× | 25.768 | 470.046 |
| 256 | 1.236940 | 0.028631 | 43.20× | 27.074 | 1169.670 |
| 512 | 9.820590 | 0.171366 | 57.31× | 27.307 | 1564.910 |
| 1024 | 76.512200 | 1.234260 | 61.99× | 28.054 | 1739.050 |

### Analysis

The GPU speedup again becomes larger as matrix size increases.

Unlike the current reordered CUDA kernel, the unrolled kernel preserves the same basic thread mapping as the naive implementation and does not add repeated block-wide synchronization.

Its performance therefore remains very close to the CUDA naive baseline.

At large sizes, it begins to slightly outperform naive due to reduced loop-control overhead.

---

# 12. CPU vs GPU Optimization Behavior

The experiments reveal an important architectural difference.

## CPU

The CPU results show that transformations such as loop reordering and unrolling can produce large improvements because they directly affect:

- cache locality,
- instruction-level parallelism,
- loop overhead,
- compiler vectorization opportunities.

For example, the previous CPU results showed that loop reordering dramatically improved cache behavior, and unrolling provided further gains.

## GPU

The CUDA results show that the same source-level idea cannot always be transferred directly.

GPU performance depends heavily on:

- thread mapping,
- warp-level memory coalescing,
- synchronization,
- occupancy,
- shared-memory reuse,
- register pressure,
- number of active warps.

Therefore:

```text
CPU optimization idea
≠
direct CUDA source-code translation
```

The optimization must be reformulated around the GPU execution model.

---

# 13. Overall CUDA Ranking

## Small Matrices (`128–1024`)

The ranking is:

```text
Naive
≈
Unrolled
>>
Reordered / Shared-Scalar
```

Naive is slightly faster than Unrolled for these sizes.

The difference between Naive and Unrolled is generally only around 1%.

The reordered version is consistently much slower.

---

## Large Matrices (`2048–4096`)

The ranking becomes:

```text
Unrolled
>
Naive
>>
Reordered / Shared-Scalar
```

At `N = 4096`:

| Implementation | Runtime | GFLOPS |
|---|---:|---:|
| Unrolled | 67.604 ms | **2032.75** |
| Naive | 71.007 ms | 1935.34 |
| Reordered / Shared-Scalar | 247.184 ms | 555.95 |

Unrolled achieves the best current result:

```text
2032.75 GFLOPS
≈ 2.03 TFLOPS
```

and is about:

```text
5.03%
```

higher throughput than Naive at `4096 × 4096`.

---

# 14. Key Conclusions

1. **CUDA Naive is already surprisingly strong.**

   Its warp-level memory accesses to matrix `B` are coalesced even though each individual thread walks through `k`.

2. **The CPU reordered optimization does not translate directly to GPU.**

   The current CUDA reordered version introduces expensive synchronization.

3. **The current reordered version is not true 2D tiled GEMM.**

   It shares only one `A[row][k]` scalar per block at a time.

4. **Two `__syncthreads()` calls per `k` are extremely expensive.**

   At `N = 4096`, each block executes roughly `8192` barriers.

5. **The synchronization penalty gets worse as matrix size increases.**

   Reordered goes from `1.58×` slower than Naive at `N = 1024` to `3.48×` slower at `N = 4096`.

6. **Manual unrolling provides only modest gains.**

   It is slightly worse for small matrices but becomes beneficial for large matrices.

7. **The best current CUDA result is Unrolled at `N = 4096`.**

   ```text
   67.6042 ms
   2032.75 GFLOPS
   ```

8. **The next meaningful GEMM optimization should be true shared-memory tiling.**

   A true tiled kernel should cooperatively load `A` and `B` tiles and perform many FMAs per synchronization phase.

---

# 15. Recommended Next GEMM Experiment

The next GEMM implementation should use:

```text
2D thread block
+
A shared-memory tile
+
B shared-memory tile
+
cooperative loading
+
tile-level synchronization
+
multiple FMAs per loaded tile
```

For example:

```text
TILE_SIZE = 16

A_tile[16][16]
B_tile[16][16]
```

The purpose of the next experiment is not simply to "use shared memory."

The real objective is to increase:

```text
arithmetic work
/
global-memory load
```

while reducing the number of synchronization phases relative to the current shared-scalar reordered implementation.

The benchmark should then compare:

```text
Naive
vs
Unrolled
vs
True Tiled
```

using the same matrix sizes:

```text
128
256
512
1024
2048
4096
```

This will show whether explicit shared-memory data reuse can move performance beyond the current `~2 TFLOPS` plateau.

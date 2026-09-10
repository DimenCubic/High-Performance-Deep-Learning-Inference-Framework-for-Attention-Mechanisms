# GEMM Performance Log


## Architectural Deep Dive

- **CPU vs. GPU Architecture:**  
  CPUs use a small number of powerful cores optimized for low-latency sequential execution and complex control flow, while GPUs use many simpler cores optimized for high-throughput parallel computation.

- **Memory Hierarchy:**  
  Data is accessed through multiple levels of storage: Registers → L1 Cache → L2/L3 Cache → Main Memory. Higher levels are faster but smaller, so performance depends heavily on keeping frequently used data close to the CPU.

- **Cache and Memory Access Patterns:**  
  Sequential row-major access provides better spatial locality and cache-line utilization, while strided or column-wise access causes more cache misses and memory delays, reducing overall throughput.



## System Detail

Hardware:
- CPU: Apple M2
- Architecture: ARM64
- OS: macOS

Complier:
- Apple Clang
- C++ 17


## Naive GEMM

### Runtime (ms)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         |11.9797| 2.3693 | 4.13849   | 3.86605   |  2.10771      |
| 256         |60.3891    | 16.236   |  19.9926  |  21.4411  |  10.3229     |
| 512         |  461.124  |  141.083  | 141.057   | 142.216   |   86.9656    |
| 1024        | 3722.56   | 1174.06   | 1172.06   |  1206.04  |   866.862    |

### Performance (GFLOPS)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         | 0.34875   | 1.76336   | 1.00953   | 1.08067   | 1.98221    |
| 256         | 0.554552   | 2.06263   |  1.67506  |  1.5619  |  3.24415     |
| 512         | 0.581565   | 1.90082   |  1.90119  |  1.88568  |  3.08367     |
| 1024        |  0.576602  | 1.82821   |  1.83133  |  1.77974  |   2.4761    |

### Profiling Data (O0)

**Profiler:** Apple Instruments CPU Profiler  
**Program:** `gemm_benchmark`  
**Kernel:** `naive_gemm`  
**Compiler Optimization:** `O0`  
**Matrix Size:** `512 × 512`  
**Thermal State:** Nominal  

#### CPU Profiling

- **Total Cycles:** ~7.92 G
- **`naive_gemm` Cycles:** ~7.88 G
- **`naive_gemm` Cycle Share:** ~99.5%

#### CPU Bottleneck Analysis

| Metric | Average |
|---|---:|
| Instruction Processing Bottleneck | 70.96% |
| Useful | 28.49% |
| Discarded Bottleneck | 5.06% |
| Instruction Delivery Bottleneck | 0.51% |

#### L1D Cache Analysis

| Metric | Value |
|---|---:|
| Load Micro-operations (Speculative) | 10,188,604,632 |
| L1D Load Misses (Speculative) | 677,846,796 |
| L1D Load Miss Rate | ~6.65% |
| Store Micro-operations (Speculative) | 1,442,036,782 |
| L1D Store Misses (Speculative) | 44,997,224 |
| L1D Store Miss Rate | ~3.12% |
| L1D Cache Writebacks (Speculative) | 1,922,701 |
| Cycles | 8,037,915,756 |




### Analysis

#### 1. Large performance gap between `-O0` and `-O1`

**Observation**

- There is a significant performance improvement from `-O0` to `-O1`.

**Interpretation**

- The generated machine code can differ significantly from the original source-code structure.

- `-O0` preserves much more of the original instruction flow and therefore introduces substantial avoidable overhead.

- Possible sources of overhead include:
  - unnecessary memory loads/stores,
  - poor register allocation,
  - redundant instructions,
  - branch overhead,
  - instruction dependencies.


---

#### 2. `-O1` to `-O2`

**Observation**

- O2 and O3 are very close

**Interpretation**

- O2 has already utilize almost all easy overhead, and left improve room is very hard for the complier such as column access to the matrix like B[0][j] which need to access bunch of discrete address space.

---
#### 3. `Ofast`

**Observation**

- Find another impressive improvement.

**Interpretation**

- floating-point reassociation / vectorization
- In order to avoid bunch of dependency on the code like sum += A[] * B[], Ofast will do something parallel calculation. Note: This action is dangerous and need to be careful used especially on the floating calculation.


---
#### 4. `GFLOPS`

**Observation**

- Not linear relationship between N and GFLOPS.
- GFLOPS will increase first than maintain steady.
- Some small matrix has lower GFLOPS
- When N reaches to the 1024, GFLOPS drops a little bit.

**Interpretation**

- When matrix is small, the percentage of overhead will be obvious and CPU also doesn't have enough time to come in sustained execution status.

- N = 1024 may reaches to the limit of the cache capacity and memory access behaviour may have sigificant influence.


---

#### 5. `CPU Profiling`

**Observation**

- `naive_gemm` takes about 99.5% of the total CPU cycles.
- The Instruction Processing Bottleneck is very high at 70.96%.
- The Instruction Delivery Bottleneck is very low at only 0.51%.
- The Useful ratio is only 28.49%.
- The thermal state remains Nominal during the profiling process.

**Interpretation**

- Most of the execution cost comes from the GEMM kernel itself, which means the overhead from matrix initialization, timing, and output is relatively small.

- The high Instruction Processing Bottleneck indicates that the CPU can deliver instructions faster than the backend can complete them.

- The low Instruction Delivery Bottleneck shows that instruction fetching and decoding are not the main performance problems.

- The processing bottleneck may come from memory access delays, floating-point dependency chains, or insufficient instruction-level parallelism.

- Since the thermal state remains Nominal, the measured performance is unlikely to be significantly affected by thermal throttling.


---

#### 6. `L1D Cache`

**Observation**

- The speculative L1D load miss rate is about 6.65%.
- The speculative L1D store miss rate is about 3.12%.
- The number of L1D load misses is much higher than the number of L1D store misses.
- Load micro-operations are also much more frequent than store micro-operations.
- The naive GEMM generates a large amount of L1D cache activity during execution.

**Interpretation**

- The relatively high number of load misses suggests that memory read behavior is an important performance factor for the naive GEMM implementation.

- One possible source is the access pattern of `B[k * N + j]`. When `k` increases, the program accesses different rows of matrix B instead of adjacent elements in memory.

- For `N = 1024`, consecutive accesses to `B[k * N + j]` are separated by approximately 4096 bytes, which results in poor spatial locality.

- Poor spatial locality reduces cache-line utilization because only a small part of each loaded cache line may be immediately useful.

- These cache misses may increase memory access latency and contribute to the high Instruction Processing Bottleneck observed in CPU profiling.

- However, the profiling data does not prove that cache misses are the only cause of the processing bottleneck. Floating-point dependencies and other backend limitations may also contribute.



### Conclusion

```text
-O0
│
│  Large source-level overhead
↓
-O1
│
│  Basic compiler optimizations applied
↓
-O2
│
│  More aggressive optimization
↓
-O3
│
│  Limited by memory access / FP dependency
↓
-Ofast
    Allows more aggressive FP transformations
    → Performance improves significantly again
```



## Reordered GEMM

### Runtime (ms)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         | 4.62467   |  0.942342  | 0.187092   | 0.149858   |  0.149017     |
| 256         |  33.8108  |  5.77711  |  1.30492  | 1.2492   |   2.10532    |
| 512         |  221.178  |  47.8456  |  13.3523  |  12.8677  |    11.6725   |
| 1024        |  1587.32  |  328.92  | 79.2432   |  81.45  |   78.9916    |

### Performance (GFLOPS)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         |  0.903398  |  4.43355  | 22.3309   |  27.8791  |   28.0366    |
| 256         |  0.99048  |  5.79683  |  26.6636  |  26.8083  |   15.9068    |
| 512         |  1.21248  |  5.60498  |  20.0844  |  20.8408  |    22.9748   |
| 1024        |  1.35224  |   6.5257 |  27.0867  |  26.3528  |   27.1729    |

### Profiling Data

**Profiler:** Apple Instruments CPU Profiler  
**Program:** `gemm_benchmark`  
**Kernel:** `reordered_gemm`  
**Compiler Optimization:** `O0`  
**Matrix Size:** `512 × 512`  
**Thermal State:** Nominal  

#### CPU Profiling

- **Total Cycles:** ~3.34 G
- **`reordered_gemm` Cycles:** ~3.29 G
- **`reordered_gemm` Cycle Share:** ~98.5%

#### CPU Bottleneck Analysis

| Metric | Average |
|---|---:|
| Useful | 67.50% |
| Instruction Processing Bottleneck | 30.89% |
| Discarded Bottleneck | 13.65% |
| Instruction Delivery Bottleneck | 1.51% |

#### L1D Cache Analysis

| Metric | Value |
|---|---:|
| Load Micro-operations (Speculative) | 9,557,579,644 |
| L1D Load Misses (Speculative) | 13,035,158 |
| L1D Load Miss Rate | ~0.14% |
| Store Micro-operations (Speculative) | 1,433,554,798 |
| L1D Store Misses (Speculative) | 37,905,054 |
| L1D Store Miss Rate | ~2.64% |
| L1D Cache Writebacks (Speculative) | 416,954 |
| Cycles | 3,394,087,825 |




## Naive GEMM vs Reordered GEMM

### `Performance Comparison (O0, N = 512)`

| Metric | Naive GEMM | Reordered GEMM | Improvement |
|---|---:|---:|---:|
| Runtime (ms) | 461.124 | 221.178 | ~52.04% lower |
| Performance (GFLOPS) | 0.581565 | 1.21248 | ~108.48% higher |
| Speedup | 1.00× | ~2.08× | ~2.08× |
| Total Cycles | ~7.92 G | ~3.34 G | ~57.83% lower |
| Useful | 28.49% | 67.50% | +39.01 percentage points |
| Instruction Processing Bottleneck | 70.96% | 30.89% | -40.07 percentage points |
| Instruction Delivery Bottleneck | 0.51% | 1.51% | +1.00 percentage point |
| L1D Load Miss Rate | ~6.65% | ~0.14% | ~97.89% lower |
| L1D Store Miss Rate | ~3.12% | ~2.64% | ~15.38% lower |
| L1D Cache Writebacks | 1,922,701 | 416,954 | ~78.31% lower |


### Analysis

#### 1. `Runtime and GFLOPS`

**Observation**

- The runtime decreases from 461.124 ms to 221.178 ms.
- The reordered GEMM is approximately 2.08× faster than the naive GEMM.
- GFLOPS increases from 0.581565 to 1.21248.
- The theoretical amount of matrix multiplication computation remains approximately the same.

**Interpretation**

- The performance improvement does not come from reducing the computational complexity because both implementations still perform approximately `O(N^3)` operations.

- Instead, the reordered implementation changes how the same computation is organized and how matrix data is accessed.

- This demonstrates that improving the memory access pattern can significantly increase performance even when the total amount of arithmetic computation remains almost unchanged.


---

#### 2. `CPU Cycles and Processing Bottleneck`

**Observation**

- Total CPU cycles decrease from approximately 7.92 G to 3.34 G.
- The Useful ratio increases from 28.49% to 67.50%.
- The Instruction Processing Bottleneck decreases significantly from 70.96% to 30.89%.
- The Instruction Delivery Bottleneck remains relatively small in both implementations.

**Interpretation**

- The large reduction in CPU cycles indicates that the reordered implementation allows the CPU to complete the same workload more efficiently.

- The increase in Useful ratio indicates that a much larger fraction of the CPU execution capacity is being used to make useful computational progress.

- The large decrease in Instruction Processing Bottleneck suggests that the CPU backend spends less time waiting for data or blocked by inefficient execution dependencies.

- Instruction fetching and decoding are not the main bottlenecks because the Instruction Delivery Bottleneck remains relatively small.


---

#### 3. `L1D Cache Miss`

**Observation**

- The speculative L1D load miss rate decreases from approximately 6.65% to only 0.14%.
- This represents approximately a 97.89% reduction in the L1D load miss rate.
- The speculative L1D store miss rate decreases from approximately 3.12% to 2.64%.
- L1D cache writebacks decrease from 1,922,701 to 416,954.

**Interpretation**

- The most significant cache improvement occurs on matrix load operations.

- In the naive `i-j-k` loop order, the innermost loop accesses matrix B using:

  `B[k * N + j]`

- Since `k` changes continuously while `j` remains fixed, the program accesses different rows of matrix B and therefore jumps across distant memory locations.

- In the reordered `i-k-j` implementation, `j` becomes the innermost loop:

  `B[k * N + j]`

  now accesses consecutive elements from the same row of matrix B.

- Because matrix B is stored in row-major order, this access pattern makes much better use of each loaded cache line.

- The dramatic reduction in L1D load misses provides strong profiling evidence that improved spatial locality is one of the main reasons for the performance improvement.


---

#### 4. `Overall Conclusion`

**Observation**

- Loop reordering alone produces approximately a 2.08× speedup under `-O0`.
- Runtime is reduced by approximately 52%.
- GFLOPS is more than doubled.
- L1D load miss rate decreases from approximately 6.65% to 0.14%.
- Instruction Processing Bottleneck decreases from 70.96% to 30.89%.
- Useful CPU execution increases from 28.49% to 67.50%.

**Interpretation**

- Loop reordering does not change the mathematical result or the asymptotic computational complexity of GEMM.

- The optimization mainly improves how the computation interacts with the CPU memory hierarchy.

- Better spatial locality allows the CPU to reuse cache-line data more efficiently and significantly reduces L1D cache misses.

- Reduced memory delays allow the CPU backend to process instructions more efficiently, which reduces CPU cycles and increases useful execution.

- This experiment demonstrates that high-performance computing is not only about reducing the number of arithmetic operations, but also about organizing computation to match the underlying hardware architecture and memory hierarchy.
 

## Unrolled GEMM

### Runtime (ms)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128 |4.60724 | 1.83141| 0.5471|0.162133 | 0.523142|
| 256 | 25.7109| 5.97956| 1.23694|1.31199 | 2.54524|
| 512 | 143.203| 28.9182|10.8994 |9.82059 |10.1542 |
| 1024 |1141.02 | 226.281| 80.9152|76.5122| 77.7861|

### Performance (GFLOPS)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128 | 0.906816| 2.28126| 7.63648| 25.7684| 7.98621|
| 256 |1.30252 |5.60056 | 27.0739|25.5252 |13.1575 |
| 512 |1.87268 | 9.27351| 24.6045|27.3072 | 26.41|
| 1024 |1.88116 |9.4857 | 26.527| 28.0535| 27.5941|

### Profiling Data

**Profiler:** Apple Instruments CPU Profiler  
**Program:** `gemm_benchmark`  
**Kernel:** `unrolled_gemm`  
**Compiler Optimization:** `O0`  
**Matrix Size:** `512 × 512`  
**Thermal State:** Nominal  

#### CPU Profiling

- **Total Cycles:** ~2.43 G
- **`unrolled_gemm` Cycles:** ~2.38 G
- **`unrolled_gemm` Cycle Share:** ~97.9%

#### CPU Bottleneck Analysis

| Metric | Average |
|---|---:|
| Instruction Processing Bottleneck | 0.60% |
| Useful | ~0% |
| Instruction Delivery Bottleneck | 58.95% |
| Discarded Bottleneck | 40.45% |

#### L1D Cache Analysis

| Metric | Value |
|---|---:|
| Load Micro-operations (Speculative) | 6,944,177,325 |
| L1D Load Misses (Speculative) | 12,672,860 |
| L1D Load Miss Rate | ~0.18% |
| Store Micro-operations (Speculative) | 932,981,022 |
| L1D Store Misses (Speculative) | 37,651,983 |
| L1D Store Miss Rate | ~4.04% |
| L1D Cache Writebacks (Speculative) | 470,824 |
| Cycles | 2,463,054,246 |



---

## Reordered GEMM vs Unrolled GEMM

### Profiling Comparison (O0, N = 512)

| Metric | Reordered GEMM | Unrolled GEMM | Change |
|---|---:|---:|---:|
| Runtime (ms) | 221.178 | 143.203 | ~35.25% lower |
| Performance (GFLOPS) | 1.21248 | 1.87268 | ~54.45% higher |
| Speedup | 1.00× | ~1.54× | ~1.54× |
| Total Cycles | ~3.34 G | ~2.43 G | ~27.25% lower |
| Kernel Cycles | ~3.29 G | ~2.38 G | ~27.66% lower |
| Kernel Cycle Share | ~98.5% | ~97.9% | Slightly lower |
| Useful | 67.50% | Not reported* | Different bottleneck classification |
| Instruction Processing Bottleneck | 30.89% | 0.60% | -30.29 percentage points |
| Instruction Delivery Bottleneck | 1.51% | 58.95% | +57.44 percentage points |
| Discarded Bottleneck | 13.65% | 40.45% | +26.80 percentage points |
| L1D Load Miss Rate | ~0.14% | ~0.18% | Slightly higher |
| L1D Store Miss Rate | ~2.64% | ~4.04% | Higher |
| L1D Cache Writebacks | 416,954 | 470,824 | Slightly higher |

\* `Useful` was not reported with a visible value in this profiling run.


### Analysis

#### 1. `Runtime and GFLOPS`

Observation

- Runtime decreases from 221.178 ms to 143.203 ms.
- This represents approximately a 35.25% reduction in runtime.
- Performance increases from 1.21248 GFLOPS to 1.87268 GFLOPS.
- This represents approximately a 54.45% increase in GFLOPS.
- The unrolled implementation achieves approximately a 1.54× speedup over the reordered implementation.

Interpretation

- The theoretical amount of matrix multiplication computation remains approximately unchanged, so the performance improvement does not come from reducing the `O(N^3)` computational complexity.

- Instead, loop unrolling allows the CPU to execute the same workload more efficiently.

- The reduced runtime and higher GFLOPS show that manual loop unrolling provides an additional performance improvement after the memory-access pattern has already been optimized by loop reordering.

- Since the L1D cache miss rate changes only slightly, the additional speedup is more strongly associated with reduced loop-control overhead and improved instruction-level parallelism than with cache-locality improvements.


---

#### 2. `CPU Cycles`

Observation

- Total CPU cycles decrease from approximately 3.34 G to 2.43 G.
- `unrolled_gemm` requires about 27% fewer CPU cycles than `reordered_gemm`.
- The theoretical GEMM computation remains approximately unchanged.

Interpretation

- Loop unrolling reduces the amount of loop-control overhead such as loop counter updates, comparisons, and branches.

- By processing multiple `j` elements in one loop iteration, the CPU performs more arithmetic work for each loop-control sequence.

- The reduction in cycles indicates that manual loop unrolling improves the efficiency of the reordered GEMM even without changing the `O(N^3)` computational complexity.

- Loop unrolling can also expose more independent operations to the CPU, increasing the available instruction-level parallelism.


---

#### 3. `Instruction Processing`

Observation

- The Instruction Processing Bottleneck decreases significantly from 30.89% to only 0.60%.
- However, the Instruction Delivery Bottleneck increases from 1.51% to 58.95%.
- The Discarded Bottleneck also increases from 13.65% to 40.45%.

Interpretation

- The very low Instruction Processing Bottleneck suggests that backend execution is no longer the main limitation in the unrolled implementation.

- Loop unrolling exposes multiple independent operations at the same time, which can reduce execution dependencies and allow the backend to process instructions more efficiently.

- However, unrolling increases the amount of machine code inside the loop body.

- A larger loop body may place more pressure on instruction delivery, instruction cache, decoding, and branch/speculative execution mechanisms.

- The bottleneck therefore appears to shift from backend instruction processing toward instruction delivery and discarded speculative work.

- This shows that optimization can move the performance bottleneck rather than completely remove it.


---

#### 4. `L1D Cache`

Observation

- The L1D load miss rate changes only slightly from approximately 0.14% to 0.18%.
- The L1D store miss rate increases from approximately 2.64% to 4.04%.
- L1D cache writebacks increase slightly from 416,954 to 470,824.
- The large cache improvement achieved by loop reordering is mostly preserved.

Interpretation

- Loop unrolling is not primarily a cache-locality optimization.

- Both `reordered_gemm` and `unrolled_gemm` still access matrix B sequentially along the innermost `j` loop.

- Therefore, the spatial locality introduced by loop reordering remains largely unchanged.

- The small increase in cache miss rates suggests that the main benefit of loop unrolling does not come from reducing L1D cache misses.

- Instead, the performance improvement mainly comes from reducing loop overhead and increasing instruction-level parallelism.


---

#### 5. `Overall Conclusion`

Observation

- Runtime decreases by approximately 35.25%.
- GFLOPS increases by approximately 54.45%.
- The unrolled implementation achieves approximately a 1.54× speedup.
- Loop unrolling reduces total CPU cycles by approximately 27%.
- The Instruction Processing Bottleneck decreases from 30.89% to 0.60%.
- L1D load miss rate remains very low and changes only slightly.
- The dominant bottleneck shifts toward Instruction Delivery and Discarded Bottleneck.

Interpretation

- Loop reordering mainly optimized memory locality, while loop unrolling mainly improves instruction execution efficiency.

- The unrolled implementation performs the same mathematical work but exposes more independent operations and reduces loop-control overhead.

- Since cache behavior changes only slightly, the additional performance improvement is mainly caused by instruction-level optimization rather than memory-locality improvement.

- The runtime, GFLOPS, and cycle measurements all consistently show that loop unrolling provides a meaningful additional optimization over the reordered GEMM.

- The profiling results also demonstrate an important performance-engineering principle: after one bottleneck is reduced, another bottleneck may become dominant.




## Blocked GEMM

### Runtime (ms)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128 | | | | | |
| 256 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |

### Performance (GFLOPS)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128 | | | | | |
| 256 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
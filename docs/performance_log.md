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
**Thermal State:** Nominal  N

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

### Block Size Tuning

Configuration:
- Matrix Size: 1024 × 1024
- Compiler Optimization: `O3`


| Block Size | Runtime (ms) | Performance (GFLOPS) |
|---:|---:|---:|
| 16 | 189.326| 11.3373|
| 32 | 153.887| 13.9481|
| 64 | 108.132 | 19.8501 |
| 128 | 93.4415| 22.9709|
| 256 |99.6236 | 21.5455|



### Runtime (ms) [128 Block Size]

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128 | 3.29818| 0.796692| 0.50485|0.499592 | 0.239992|
| 256 | 32.9728|6.20008 | 1.39276| 1.30709| 1.16832|
| 512 | 221.65| 49.5799| 10.1164| 10.2202| 10.2277|
| 1024 | 1632.68| 398.976| 92.2449| 97.0308| 92.0751|
| 2048 | 12950.2| 3251.84| 900.91| 898.786| 897.346|

### Performance (GFLOPS) [128 Block Size]

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128 | 1.26674| 5.24409| 8.27557|8.36267 | 17.4086|
| 256 | 1.01565| 5.40136| 24.045| 25.6209| 28.664|
| 512 | 1.2099| 5.40892| 26.5089| 26.2395| 26.2202|
| 1024 | 1.31467| 5.37986| 23.2689| 22.1212| 23.3118|
| 2048 |1.32629 | 5.28184| 19.0648| 19.1099| 19.1405|

## Reordered GEMM vs Blocked GEMM

### Profiling Comparison (O3, N = 2048, Block Size = 128)

| Metric | Reordered GEMM | Blocked GEMM | Change |
|---|---:|---:|---:|
| Runtime (ms) | 637.275 | 898.786 | ~41.04% higher |
| Performance (GFLOPS) | 26.9517 | 19.1099 | ~29.10% lower |
| Relative Performance | 1.00× | ~0.71× | Slower |
| CPU Counter Cycles | ~10.83 G | ~15.10 G | ~39.41% higher |
| Useful | 56.03% | 43.30% | -12.73 percentage points |
| Instruction Processing Bottleneck | 41.80% | 56.38% | +14.58 percentage points |
| Instruction Delivery Bottleneck | 0.43% | 0.09% | Slightly lower |
| Discarded Bottleneck | 1.73% | 0.23% | Lower |
| Load Micro-operations (Speculative) | 5,935,957,895 | 9,377,406,743 | ~58% higher |
| L1D Load Misses (Speculative) | 4,501,854,633 | 8,477,625,141 | Much higher |
| L1D Load Miss Rate | ~75.84% | ~90.40% | +14.56 percentage points |
| Store Micro-operations (Speculative) | 32,859,283,238 | 32,696,154,731 | Similar |
| L1D Store Misses (Speculative) | 79,484,992 | 11,818,095 | Lower |
| L1D Store Miss Rate | ~0.24% | ~0.04% | Lower |


### Analysis

#### 1. `Runtime and GFLOPS`

Observation

- Runtime increases from 637.275 ms to 898.786 ms.
- The blocked implementation is approximately 41.04% slower in runtime.
- Performance decreases from 26.9517 GFLOPS to 19.1099 GFLOPS.
- The blocked implementation achieves only about 0.71× the performance of the reordered implementation.

Interpretation

- The blocking transformation does not automatically guarantee better performance.

- Although blocking is designed to improve cache reuse, it also introduces additional loop levels, block-boundary calculations, and shorter inner loops.

- In this experiment, the additional overhead and memory behavior outweigh the expected benefit from cache reuse.

- The reordered implementation is already highly compiler-friendly because its innermost loop performs long sequential accesses to matrices B and C.

- Therefore, the compiler may optimize and vectorize the reordered version more effectively than the blocked version.


---

#### 2. `CPU Cycles`

Observation

- CPU counter cycles increase from approximately 10.83 G to 15.10 G.
- This represents approximately a 39.41% increase in cycles.
- The Useful ratio decreases from 56.03% to 43.30%.

Interpretation

- The blocked version requires significantly more CPU cycles to complete the same matrix multiplication workload.

- The additional `ii`, `kk`, and `jj` block loops introduce extra loop-control and address-calculation overhead.

- Shorter inner loops may also reduce some optimization opportunities available to the compiler.

- The decrease in Useful execution is consistent with the higher runtime and lower GFLOPS.


---

#### 3. `Instruction Processing Bottleneck`

Observation

- Instruction Processing Bottleneck increases from 41.80% to 56.38%.
- Instruction Delivery Bottleneck remains extremely low in both versions.
- Discarded Bottleneck also remains relatively small.

Interpretation

- The main limitation of the blocked implementation is still on the instruction-processing side rather than instruction delivery.

- The CPU is able to provide instructions, but the backend requires more time to complete them.

- The additional block-management operations and increased memory activity may contribute to the higher processing bottleneck.

- This result suggests that the selected blocking strategy does not reduce backend pressure for this workload.


---

#### 4. `L1D Cache`

Observation

- The speculative L1D load miss rate increases from approximately 75.84% to 90.40%.
- L1D load misses increase from approximately 4.50 billion to 8.48 billion.
- Load micro-operations also increase significantly from approximately 5.94 billion to 9.38 billion.
- The L1D store miss rate decreases, but store behavior does not appear to be the dominant problem.

Interpretation

- Contrary to the intended goal of blocking, the current blocked implementation produces worse L1D load behavior in this profiling experiment.

- One possible reason is that the current block-loop ordering and block size do not match the M2 cache hierarchy well.

- The blocked implementation repeatedly enters smaller subregions of the matrices, increasing loop and memory-access complexity.

- Meanwhile, the reordered implementation already performs long sequential accesses to matrix B and C, which gives the hardware prefetcher and compiler a very simple access pattern.

- Blocking therefore disrupts some of the advantages of the long sequential inner loop without providing enough additional cache reuse to compensate.

- This demonstrates that blocking must be tuned for a particular hardware architecture instead of assuming that any block size will improve cache performance.


---

#### 5. `Block Size Tuning`

Observation

- Performance improves as the block size increases from 16 to 128.
- `BLOCK_SIZE = 128` produces the best result among the tested values.
- Increasing the block size further to 256 slightly reduces performance.

Interpretation

- Very small blocks introduce excessive loop and block-management overhead.

- Increasing the block size reduces this overhead and provides longer inner loops.

- However, if the block size becomes too large, the working set of the active tiles may become less suitable for the cache hierarchy.

- The result suggests that `BLOCK_SIZE = 128` provides the best trade-off among the tested configurations, but even the best blocked version does not outperform the simple reordered GEMM.


---

#### 6. `Overall Conclusion`

Observation

- The best tested blocked configuration uses `BLOCK_SIZE = 128`.
- However, blocked GEMM is still slower than reordered GEMM at `N = 2048` and `O3`.
- Runtime increases by approximately 41%.
- GFLOPS decreases by approximately 29%.
- CPU cycles and the Instruction Processing Bottleneck both increase.
- L1D load behavior also becomes worse.

Interpretation

- Memory tiling is theoretically useful because it can increase data reuse, but its effectiveness strongly depends on hardware, block size, loop ordering, compiler optimization, and the existing memory-access pattern.

- In this project, loop reordering had already created highly sequential access to B and C.

- The additional blocking structure introduces overhead and interferes with some of the compiler and hardware optimizations that benefit the simpler reordered loop.

- Therefore, the current experiment shows that a theoretically reasonable optimization can reduce real performance if it is not well matched to the target architecture.

- Performance optimization must always be validated through benchmarking and profiling rather than assumed to be beneficial.




## SIMD GEMM


### Runtime (ms) 

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128         |  8.05926  | 0.396758   |   0.199742 |  0.170192  |   0.166725    |
| 256         |  31.405  |  2.21145  |  1.78384  |  2.65262  |    1.84924   |
| 512         | 213.33   |  16.7957  |  14.1433  |   14.3025 |  13.7627     |
| 1024        |   1699.35 |  102.391  |   104.592 |  106.049  | 112.62      |

### Performance (GFLOPS) 

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|---|---:|---:|---:|---:|---:|
| 128         |  0.5184  |  10.5301  |  20.9166  |  24.5438  |   25.0588    |
| 256         |  1.06635  |   15.1434 |  18.7735  |  12.6249  |    18.1095   |
| 512         |  1.25708  | 15.9668   | 18.9612   |  18.7502  |   19.4855    |
| 1024        |  1.26309  |  20.9632  |   20.5219 |   20.24 |   19.0591    |



## Reordered GEMM vs SIMD GEMM

### Profiling Comparison (O0, N = 512)

| Metric | Reordered GEMM | SIMD GEMM | Change |
|---|---:|---:|---:|
| Runtime (ms) | 221.178 | 213.330 | ~3.55% lower |
| Performance (GFLOPS) | 1.21248 | 1.25708 | ~3.68% higher |
| Speedup | 1.00× | ~1.04× | ~1.04× |
| CPU Profiler Total Cycles | ~3.34 G | ~3.63 G | Higher |
| Kernel Cycles | ~3.29 G | ~3.58 G | Higher |
| Kernel Cycle Share | ~98.5% | ~98.6% | Similar |
| Useful | 67.50% | 41.93% | -25.57 percentage points |
| Instruction Processing Bottleneck | 30.89% | 56.60% | +25.71 percentage points |
| Instruction Delivery Bottleneck | 1.51% | 1.31% | Similar |
| Discarded Bottleneck | 13.65% | 4.06% | Lower |
| Load Micro-operations (Speculative) | 9,557,579,644 | 3,321,459,935 | ~65.25% lower |
| L1D Load Misses (Speculative) | 13,035,158 | 2,079,366,535 | Much higher |
| L1D Load-Miss / Load-µop Ratio | ~0.14% | ~62.61% | Much higher |
| Store Micro-operations (Speculative) | 1,433,554,798 | 8,587,952,123 | Much higher |
| L1D Store Misses (Speculative) | 37,905,054 | 5,716,605 | Much lower |
| L1D Store-Miss / Store-µop Ratio | ~2.64% | ~0.067% | Much lower |


### Analysis

#### 1. `Runtime and GFLOPS`

Observation

- Runtime decreases from 221.178 ms to 213.330 ms.
- Performance increases from 1.21248 GFLOPS to 1.25708 GFLOPS.
- The SIMD implementation achieves approximately a 1.04× speedup.
- The measured improvement is much smaller than the theoretical 4-element SIMD width.

Interpretation

- ARM NEON allows one 128-bit vector instruction to process four `float` values at the same time.

- However, processing four values per vector instruction does not mean the whole GEMM becomes four times faster.

- SIMD only accelerates part of the execution. Memory loads, stores, address calculations, loop control, and other instructions still consume execution resources.

- The small performance improvement suggests that the current SIMD implementation is limited by factors other than scalar floating-point arithmetic alone.


---

#### 2. `SIMD and Data-Level Parallelism`

Observation

- The SIMD implementation explicitly processes four adjacent `B` and `C` elements in each vector iteration.
- Load micro-operations decrease significantly compared with the reordered scalar implementation.
- Runtime still improves slightly even under `O0`.

Interpretation

- Unlike loop unrolling, SIMD provides real data-level parallelism.

- A single NEON vector operation can perform arithmetic on four `float` values:

  `[C0 C1 C2 C3] += [a a a a] × [B0 B1 B2 B3]`

- Loop reordering makes this possible because matrix B and C are accessed sequentially along the innermost `j` dimension.

- Therefore, loop reordering provides the contiguous memory layout required for efficient SIMD vector loads and stores.


---

#### 3. `Instruction Processing`

Observation

- Instruction Processing Bottleneck increases from 30.89% to 56.60%.
- Useful decreases from 67.50% to 41.93%.
- Instruction Delivery Bottleneck remains low in both implementations.
- Discarded Bottleneck decreases from 13.65% to 4.06%.

Interpretation

- SIMD reduces the number of scalar arithmetic operations, but vector instructions are more complex and place pressure on the SIMD/FP execution units.

- The backend must perform vector loads, vector FMA operations, and vector stores.

- Therefore, the main bottleneck remains in instruction processing rather than instruction delivery.

- This suggests that simply replacing scalar arithmetic with SIMD instructions is not enough to fully utilize the CPU.

- Further optimization may require better register reuse, reduced load/store traffic, unrolling of the SIMD loop, or combining SIMD with other optimization techniques.


---

#### 4. `L1D Cache and Memory Operations`

Observation

- The speculative load micro-operation count decreases significantly.
- However, the measured speculative L1D load-miss-to-load-µop ratio becomes much higher in the SIMD profiling run.
- Store micro-operation behavior also changes significantly after vectorization.

Interpretation

- SIMD changes the type and number of micro-operations generated by the CPU, so cache-counter ratios cannot be compared as directly as between two similar scalar implementations.

- One vector load transfers multiple `float` values at once, meaning the relationship between vector instructions, micro-operations, and cache events differs from the scalar version.

- Therefore, the speculative `L1D Load Misses / Load Micro-operations` ratio should not be interpreted as an exact conventional cache miss rate.

- The more reliable conclusion from this experiment is that SIMD changes the memory-access execution behavior substantially, while the actual runtime improvement remains relatively small.


---

#### 5. `Why SIMD Is Not 4× Faster`

Observation

- NEON processes four FP32 values per 128-bit vector.
- The actual measured speedup is only approximately 1.04×.

Interpretation

- SIMD width represents how many data elements can be processed by one vector instruction, not the total program speedup.

- GEMM execution still includes vector loads and stores, address calculations, loop control, cache accesses, and instruction dependencies.

- Each inner-loop iteration currently loads `B`, loads `C`, executes an FMA, and stores `C` again.

- Therefore, memory traffic remains significant relative to the arithmetic work performed.

- The current implementation does not keep a large amount of C data in SIMD registers across multiple `k` iterations, so repeated loads and stores limit the benefit of vector arithmetic.

- This explains why SIMD alone provides only a modest improvement over the reordered scalar implementation.


---

#### 6. `Overall Conclusion`

Observation

- SIMD reduces runtime by approximately 3.55%.
- GFLOPS increases by approximately 3.68%.
- The achieved speedup is approximately 1.04×.
- The SIMD implementation substantially changes the instruction and memory-operation behavior.
- Instruction Processing Bottleneck becomes the dominant limitation.

Interpretation

- Loop reordering provides memory locality, while SIMD adds data-level parallelism on top of the reordered access pattern.

- The current SIMD implementation successfully uses vector operations but does not yet achieve a large speedup because execution is still limited by memory operations and backend processing.

- SIMD vector width should not be interpreted as expected program-level speedup.

- The next opportunity for optimization is to combine SIMD with techniques such as loop unrolling and register reuse, allowing more arithmetic work to be performed for each load/store operation.



### SIMD vs SIMD Blocked (O0, N = 512)

| Metric | SIMD GEMM | SIMD Blocked GEMM | Change |
|---|---:|---:|---:|
| Runtime (ms) | 213.330 | 299.867 | ~40.56% higher |
| Performance (GFLOPS) | 1.25708 | 0.894308 | ~28.86% lower |
| Relative Performance | 1.00× | ~0.71× | Slower |



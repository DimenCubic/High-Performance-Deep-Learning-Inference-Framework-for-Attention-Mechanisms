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
 


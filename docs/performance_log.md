# GEMM Performance Log

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




### Conclusion
 



## Reordered GEMM

### Runtime (ms)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         |    |    |    |    |       |
| 256         |    |    |    |    |       |
| 512         |    |    |    |    |       |
| 1024        |    |    |    |    |       |

### Performance (GFLOPS)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         |    |    |    |    |       |
| 256         |    |    |    |    |       |
| 512         |    |    |    |    |       |
| 1024        |    |    |    |    |       |




如何启动？（样例）

常规测试
cd build
cmake ..
cmake --build .


cmake -S . -B build-O0 -DOPT_LEVEL=O0
cmake --build build-O0

./build-O0/gemm_benchmark


cmake -S . -B build-O1 -DOPT_LEVEL=O1
cmake --build build-O1

./build-O1/gemm_benchmark



cmake -S . -B build-O2 -DOPT_LEVEL=O2
cmake --build build-O2

./build-O2/gemm_benchmark


cmake -S . -B build-O3 -DOPT_LEVEL=O3
cmake --build build-O3

./build-O3/gemm_benchmark



cmake -S . -B build-Ofast -DOPT_LEVEL=Ofast
cmake --build build-Ofast

./build-Ofast/gemm_benchmark

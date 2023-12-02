# CUDSS.jl

```julia
using CUDA
using CUDSS

# Dense linear system with one right-hand side
n = 100
A_cpu = rand(n, n)
x_cpu = rand(n)
b_cpu = rand(n)

A_gpu = CuMatrix(A_cpu)
x_gpu = CuVector(x_cpu)
b_gpu = CuVector(b_cpu)

matrix = CudssMatrix(A_gpu)
solution = CudssMatrix(x_gpu)
rhs = CudssMatrix(b_gpu)

config = CudssConfig()
data = CudssData()
cudss("analysis", config, data, matrix, solution, rhs)
cudss("factorization", config, data, matrix, solution, rhs)
cudss("solve", phase, config, data, matrix, solution, rhs)
```

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays

# Sparse linear system with multiple right-hand sides
n = 100
p = 5
A_cpu = sprand(n, n, 0.5)
A_cpu = A_cpu + A_cpu'
x_cpu = rand(n, p)
b_cpu = rand(n, p)

A_gpu = CuSparseMatrixCSR(A_cpu)
X_gpu = CuMatrix(X_cpu)
B_gpu = CuMatrix(B_cpu)

matrix = CudssMatrix(A_gpu)
solution = CudssMatrix(X_gpu)
rhs = CudssMatrix(B_gpu)

config = CudssConfig()
data = CudssData()
cudss("analysis", config, data, matrix, solution, rhs)
cudss("factorization", config, data, matrix, solution, rhs)
cudss("solve", phase, config, data, matrix, solution, rhs)
```

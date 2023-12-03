# CUDSS.jl

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays

# Solve an unsymmetric linear system with one right-hand side
T = Float64
n = 100
A_cpu = sprand(T, n, n, 0.05) + I
x_cpu = zeros(T, n)
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector(x_cpu)
b_gpu = CuVector(b_cpu)

matrix = CudssMatrix(A_gpu, 'G', 'F')
config = CudssConfig()
data = CudssData()
solver = CudssSolver(matrix, config, data)

# Note that you can replace the four previous lines by
# solver = CudssSolver(A_gpu, 'G', 'F')

cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)
cudss("solve", solver, x_gpu, b_gpu)

r_gpu = b_gpu - A_gpu * x_gpu
norm(r_gpu)
```
```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays

# Solve a symmetric linear system with multiple right-hand sides
T = Float64
n = 100
p = 5
A_cpu = sprand(n, n, 0.05) + I
A_cpu = A_cpu + A_cpu'
X_cpu = zeros(n, p)
B_cpu = rand(n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
X_gpu = CuMatrix(X_cpu)
B_gpu = CuMatrix(B_cpu)

structure = T <: Real ? 'S' : 'H'
solver = CudssSolver(A_gpu, structure, 'L')

cudss("analysis", solver, X_gpu, B_gpu)
cudss("factorization", solver, X_gpu, B_gpu)
cudss("solve", solver, X_gpu, B_gpu)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)
```
```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays

# Sparse an hermitian positive define linear system
# with multiple right-hand sides
T = ComplexF64
n = 100
p = 5
A_cpu = sprand(n, n, 0.01)
A_cpu = A_cpu * A_cpu' + I
X_cpu = zeros(n, p)
B_cpu = rand(n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
X_gpu = CuMatrix(X_cpu)
B_gpu = CuMatrix(B_cpu)

structure = T <: Real ? "SPD" : "HPD"
solver = CudssSolver(A_gpu, structure, 'U')

cudss("analysis", solver, X_gpu, B_gpu)
cudss("factorization", solver, X_gpu, B_gpu)
cudss("solve", solver, X_gpu, B_gpu)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)
```

## Iterative refinement

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.01)
A_cpu = A_cpu + I
B_cpu = rand(T, n, p)

A_gpu = CuSparseMatrixCSR(A_cpu)
B_gpu = CuMatrix(B_cpu)
X_gpu = similar(B_gpu)

solver = CudssSolver(A_gpu, "G", 'F')

# Perform one step of iterative refinement
ir = 1
cudss_set(solver, "ir_n_steps", ir)

cudss("analysis", solver, X_gpu, B_gpu)
cudss("factorization", solver, X_gpu, B_gpu)
cudss("solve", solver, X_gpu, B_gpu)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)
```

## User permutation

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays
using AMD

T = ComplexF64
n = 100
A_cpu = sprand(T, n, n, 0.01)
A_cpu = A_cpu' * A_cpu + I
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)
x_gpu = similar(b_gpu)

solver = CudssSolver(A_gpu, "HPD", 'F')

# Provide a user permutation
permutation = amd(A_cpu) |> Vector{Cint}
cudss_set(solver, "user_perm", permutation)

cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)
cudss("solve", solver, x_gpu, b_gpu)

r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
norm(r_gpu)
```

## Hybrid mode

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 100
A_cpu = sprand(T, n, n, 0.01)
A_cpu = A_cpu + A_cpu' + I
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)
x_gpu = similar(b_gpu)

solver = CudssSolver(A_gpu, "S", 'F')

# Use the hybrid mode (host and device memory)
cudss_set(solver, "hybrid_mode", 1)

cudss("analysis", solver, x_gpu, b_gpu)

# Minimal amount of device memory required in the hybrid memory mode.
nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")

# Device memory limit for the hybrid memory mode.
# Only use it if you don't want to rely on the internal default heuristic.
cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

cudss("factorization", solver, x_gpu, b_gpu)
cudss("solve", solver, x_gpu, b_gpu)

r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
norm(r_gpu)
```

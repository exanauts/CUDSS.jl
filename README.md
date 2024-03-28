# CUDSS.jl: Julia interface for NVIDIA cuDSS

[![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://exanauts.github.io/CUDSS.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://exanauts.github.io/CUDSS.jl/dev

## Overview

[CUDSS.jl](https://github.com/exanauts/CUDSS.jl) is a Julia interface to the NVIDIA [cuDSS](https://developer.nvidia.com/cudss) library.
NVIDIA cuDSS provides three factorizations (LDU, LDLᵀ, LLᵀ) for solving sparse linear systems on GPUs.

### Why CUDSS.jl?

Unlike other CUDA libraries that are commonly bundled together, cuDSS is currently in preview. For this reason, it is not included in [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
To maintain consistency with the naming conventions used for other CUDA libraries (such as CUBLAS, CUSOLVER, CUSPARSE, etc.), we have named this interface CUDSS.jl.

## Installation

CUDSS.jl can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add CUDSS
pkg> test CUDSS
```

## Examples

### Example 1: Sparse unsymmetric linear system with one right-hand side

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
n = 100
A_cpu = sprand(T, n, n, 0.05) + I
x_cpu = zeros(T, n)
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector(x_cpu)
b_gpu = CuVector(b_cpu)

solver = CudssSolver(A_gpu, "G", 'F')

cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)
cudss("solve", solver, x_gpu, b_gpu)

r_gpu = b_gpu - A_gpu * x_gpu
norm(r_gpu)

# In-place LU
d_gpu = rand(T, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cudss_set(solver, A_gpu)

c_cpu = rand(T, n)
c_gpu = CuVector(c_cpu)

cudss("factorization", solver, x_gpu, c_gpu)
cudss("solve", solver, x_gpu, c_gpu)

r_gpu = c_gpu - A_gpu * x_gpu
norm(r_gpu)
```

### Example 2: Sparse symmetric linear system with multiple right-hand sides

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.05) + I
A_cpu = A_cpu + A_cpu'
X_cpu = zeros(T, n, p)
B_cpu = rand(T, n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
X_gpu = CuMatrix(X_cpu)
B_gpu = CuMatrix(B_cpu)

structure = T <: Real ? "S" : "H"
solver = CudssSolver(A_gpu, structure, 'L')

cudss("analysis", solver, X_gpu, B_gpu)
cudss("factorization", solver, X_gpu, B_gpu)
cudss("solve", solver, X_gpu, B_gpu)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)

# In-place LDLᵀ
d_gpu = rand(R, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cudss_set(solver, A_gpu)

C_cpu = rand(T, n, p)
C_gpu = CuMatrix(C_cpu)

cudss("factorization", solver, X_gpu, C_gpu)
cudss("solve", solver, X_gpu, C_gpu)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

### Example 3: Sparse hermitian positive definite linear system with multiple right-hand sides

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = ComplexF64
R = real(T)
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.01)
A_cpu = A_cpu * A_cpu' + I
X_cpu = zeros(T, n, p)
B_cpu = rand(T, n, p)

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

# In-place LLᴴ
d_gpu = rand(R, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cudss_set(solver, A_gpu)

C_cpu = rand(T, n, p)
C_gpu = CuMatrix(C_cpu)

cudss("factorization", solver, X_gpu, C_gpu)
cudss("solve", solver, X_gpu, C_gpu)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

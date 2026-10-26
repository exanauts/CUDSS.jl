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
cudss("factorization", solver, X_gpu, B_gpu; asynchronous=false)
cudss("solve", solver, X_gpu, B_gpu; asynchronous=false)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)

# Number of iterative refinement steps performed during the last solve
cudss_get(solver, "ir_n_steps_performed")
```

## Transpose and adjoint solves

Since cuDSS 1.0, a factorization of `A` can also be used to solve `Aᵀ x = b` and `Aᴴ x = b`.
The configuration parameter `"solve_mode"` selects the operator: `'N'` (default), `'T'` (transpose) or `'C'` (conjugate transpose).
Transpose solves are not supported for Hermitian / HPD matrices, the Schur complement mode, COLAMD-family reorderings and non-uniform batches.
Conjugate-transpose solves are only supported for complex general matrices.

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = ComplexF64
n = 100
A_cpu = sprand(T, n, n, 0.02) + I
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)
x_gpu = similar(b_gpu)

solver = CudssSolver(A_gpu, "G", 'F')
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)

# Solve Aᵀx = b
cudss_set(solver, "solve_mode", 'T')
cudss("solve", solver, x_gpu, b_gpu)
norm(b_cpu - transpose(A_cpu) * Array(x_gpu))

# Solve Aᴴx = b with the same factorization
cudss_set(solver, "solve_mode", 'C')
cudss("solve", solver, x_gpu, b_gpu)
norm(b_cpu - A_cpu' * Array(x_gpu))
```

## Matching

Since cuDSS 1.0, matching with scaling is enabled by default (`"matching_alg"` = `"algo6"`, i.e. automatic) for general, symmetric and Hermitian matrices, and disabled for SPD / HPD matrices.
It improves the robustness of the factorization of indefinite and unsymmetric matrices.
Use `cudss_set(solver, "matching_alg", "none")` before the analysis to recover the behavior of cuDSS 0.8.
With matching and scaling, `cudss_get(solver, "diag")` returns the diagonal of the scaled matrix.

## Bunch-Kaufman pivoting

cuDSS 1.0 supports Bunch-Kaufman pivoting for symmetric and Hermitian indefinite matrices with the AMD (`"algo3"`) or nested dissection (`"default"`) reordering.
The automatic pivoting (`'A'`, the default) already selects it in these cases.
Note that solves with Bunch-Kaufman pivoting cannot be captured in a CUDA graph.

```julia
solver = CudssSolver(A_gpu, "S", 'F')
cudss_set(solver, "pivot_type", 'B')
cudss_set(solver, "reordering_alg", "algo3")
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)
cudss("solve", solver, x_gpu, b_gpu)
cudss_get(solver, "inertia")
```

## Reducing the memory footprint

Two options of cuDSS 1.0 reduce the memory used by the factors of double precision matrices:

- `cudss_set(solver, "downcast_factors", mode)` stores the factors in single precision. `mode` is `"keep_block_diagonal"` or `"keep_strict_diagonal"` to keep the supernode diagonal blocks or the diagonal in double precision, and `"disabled"` (default). It must be set before the factorization and not changed afterwards.
- `cudss_set(solver, "memory_optimization_setting", 1)` releases the memory that is only needed to recompute the factors. Use it when only solves follow the factorization: a later factorization requires a new analysis.

## JIT compilation of kernels

cuDSS 1.0 compiles some kernels at runtime with NVRTC and caches them on disk.
The environment variable `CUDSS_JIT_CACHE_DIR` sets the cache directory, and `CUDSS_JIT_DISABLE` disables runtime compilation.
Set them before loading CUDSS.jl.

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
cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)
cudss("solve", solver, x_gpu, b_gpu; asynchronous=false)

r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
norm(r_gpu)
```

## Hybrid memory mode

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
cudss_set(solver, "hybrid_memory_mode", 1)

cudss("analysis", solver, x_gpu, b_gpu)

# Minimal amount of device memory required in the hybrid memory mode.
nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")

# Device memory limit for the hybrid memory mode.
# Only use it if you don't want to rely on the internal default heuristic.
cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

cudss("factorization", solver, x_gpu, b_gpu; asynchronous=true)
cudss("solve", solver, x_gpu, b_gpu; asynchronous=true)

r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
norm(r_gpu)
```

## Selecting matrices in a uniform batch

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
n = 3
nbatch = 3

# Collection of unsymmetric linear systems
#        [1+λ  0   3  ]
# A(λ) = [ 4  5+λ  0  ]
#        [ 2   6  2+λ ]
nnzA = 7
rowPtr = CuVector{Cint}([1, 3, 5, 8])
colVal = CuVector{Cint}([1, 3, 1, 2, 1, 2, 3])

# List of values for λ
Λ = [1.0, 10.0, -20.0]
nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                     1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                     1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])

bλ_gpu = CuVector{T}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
xλ_gpu = CuVector{T}(undef, n * nbatch)
cudss_bλ_gpu = CudssMatrix(T, n; nbatch)
cudss_xλ_gpu = CudssMatrix(T, n; nbatch)
cudss_update(cudss_bλ_gpu, bλ_gpu)
cudss_update(cudss_xλ_gpu, xλ_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "G", 'F')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)

rλ_gpu = rand(T, nbatch)
for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
    x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

# Solve only the first linear system with the new right-hand sides
cudss_set(solver, "ubatch_index", 0)  # 0-based index of the first matrix

cλ_gpu = CuVector{T}([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
cudss_update(cudss_bλ_gpu, cλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)

for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    c_gpu = cλ_gpu[1 + (i-1) * n : i * n]
    x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
    r_gpu = c_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

# Refactorize only the first matrix of the batch
Λ = [-2.0, -10.0, 30.0]
new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                         1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                         1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
cudss_update(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)

for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    c_gpu = cλ_gpu[1 + (i-1) * n : i * n]
    x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
    r_gpu = c_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

# Process again all matrices at once in the uniform batch
cudss_set(solver, "ubatch_index", -1)
```

The data parameter `"ubatch_mask"` selects several matrices of a uniform batch at once.
It is a `Vector{Cint}` of length `"ubatch_size"` with `1` for the systems to process and `0` for the systems to skip, whose solutions are left unchanged.
The mask must be set before the analysis, and can be updated before a refactorization.

```julia
solver = CudssSolver(rowPtr, colVal, nzVal, "G", 'F')
cudss_set(solver, "ubatch_size", nbatch)
cudss_set(solver, "ubatch_mask", Cint[1, 0, 1])

cudss("analysis", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

# Process only the second system
cudss_set(solver, "ubatch_mask", Cint[0, 1, 0])
cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)
```

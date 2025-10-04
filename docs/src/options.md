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
cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

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
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

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
cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

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

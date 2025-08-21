#### Batch factorization of matrices with a common sparsity pattern

## Batch LU

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
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

cudss_bλ_gpu = CudssMatrix(T, n; nbatch)
bλ_gpu = CuVector{T}([1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 9.0])
cudss_set(cudss_bλ_gpu, bλ_gpu)

cudss_xλ_gpu = CudssMatrix(T, n; nbatch)
xλ_gpu = CuVector{T}(undef, n * nbatch)
cudss_set(cudss_xλ_gpu, xλ_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "G", 'F')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

rλ_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
    x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

# Refactorize all matrices of the uniform batch
Λ = [-2.0, -10.0, 30.0]
new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                         1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                         1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])

cudss_set(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
    x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu
```

## Batch LDLᵀ and LDLᴴ

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = ComplexF64
R = real(T)
n = 5
nbatch = 2
nrhs = 2

nnzA = 8
rowPtr = CuVector{Cint}([1, 2, 3, 6, 7, 9])
colVal = CuVector{Cint}([1, 2, 1, 2, 3, 4, 3, 5])
nzVal = CuVector{T}([4, 3, 1+im, 2-im, 5, 1, 1+im, 2,
                     2, 3, 1-im, 1+im, 6, 4, 2-im, 8])

cudss_Bs_gpu = CudssMatrix(T, n, nrhs; nbatch)
Bs_gpu = CuVector{T}([ 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im,
                      13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im])
cudss_set(cudss_Bs_gpu, Bs_gpu)

cudss_Xs_gpu = CudssMatrix(T, n, nrhs; nbatch)
Xs_gpu = CuVector{T}(undef, n * nrhs * nbatch)
cudss_set(cudss_Xs_gpu, Xs_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "H", 'L')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_Xs_gpu, cudss_Bs_gpu)
cudss("factorization", solver, cudss_Xs_gpu, cudss_Bs_gpu)
cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu)

Rs_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = reshape(Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
    X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu

new_nzVal = CuVector{T}([-4, -3,  1-im, -2+im, -5, -1, -1-im, -2,
                         -2, -3, -1+im, -1-im, -6, -4, -2+im, -8])
cudss_set(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_Xs_gpu, cudss_Bs_gpu)

new_Bs_gpu = CuVector{T}([13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im,
                           7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im])
cudss_set(cudss_Bs_gpu, new_Bs_gpu)
cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu)

Rs_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = reshape(new_Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
    X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu
```

## Batch LLᵀ and LLᴴ

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
n = 5
nbatch = 2

nnzA = 8
rowPtr = CuVector{Cint}([1, 3, 5, 7, 8, 9])
colVal = CuVector{Cint}([1, 3, 2, 3, 3, 5, 4, 5])
nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                     2, 1, 3, 1, 6, 2, 4, 8])

cudss_bs_gpu = CudssMatrix(T, n; nbatch)
bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,
                      13, 15, 29, 8, 14])
cudss_set(cudss_bs_gpu, bs_gpu)

cudss_xs_gpu = CudssMatrix(T, n; nbatch)
xs_gpu = CuVector{T}(undef, n * nbatch)
cudss_set(cudss_xs_gpu, xs_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "SPD", 'U')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_xs_gpu, cudss_bs_gpu)
cudss("factorization", solver, cudss_xs_gpu, cudss_bs_gpu)
cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu)

rs_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = bs_gpu[1 + (i-1) * n : i * n]
    x_gpu = xs_gpu[1 + (i-1) * n : i * n]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu

new_nzVal = CuVector{T}([8, 2, 6, 4, 10, 2,  2,  4,
                         6, 3, 9, 3, 18, 6, 12, 24])
cudss_set(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_xs_gpu, cudss_bs_gpu)
cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu)

rs_gpu = rand(T, nbatch)
for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = bs_gpu[1 + (i-1) * n : i * n]
    x_gpu = xs_gpu[1 + (i-1) * n : i * n]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu
```

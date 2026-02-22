#### Batch factorization of matrices with a common sparsity pattern

The batch factorization of matrices that share a common sparsity pattern enables performing the symbolic analysis only once and reusing it for the entire batch, resulting in a significant speed-up.
This phase is known to be difficult to parallelize and to port efficiently to the GPU.
In particular, the reordering step performed during symbolic analysis, whose purpose is to compute a permutation that reduces fill-in in the factors, is executed on the CPU in cuDSS.

The uniform batch solver follows the same principles as the single-matrix solver: it uses the common sparsity pattern of the batch exactly as in the single-system case.
However, the numerical values of the sparse matrices, the right-hand sides, and the solutions are stored using a strided memory layout that represents the entire batch.

For convenience (i.e., as syntactic sugar), the nonzero values, right-hand sides, and solutions may also be stored in multidimensional `CuArray`s.
This is supported provided that the batch index is the last dimension, and applying `vec` to the array produces the same long vector as the expected strided layout.

!!! note
    The cuDSS API for uniform batch factorization requires CUDSS.jl v0.5.3 or later, and the generic interface for uniform batches requires CUDSS.jl v0.6.7 or later.

!!! warning
    When calling `show`, CUDA.jl converts a `CuSparseMatrixCSR` to a `SparseMatrixCSC`. This conversion is not well defined when the `nzVal` array is longer than `colVal`, which can occur in the batched setting. For this reason, a trailing semicolon is used in the following examples of the generic interface. The matrix construction works correctly, but displaying it results in an error.

## Batch LU -- cuDSS API

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
n = 3
nbatch = 3
strided = true

# Collection of unsymmetric linear systems
#        [1+λ  0   3  ]
# A(λ) = [ 4  5+λ  0  ]
#        [ 2   6  2+λ ]
nnzA = 7
rowPtr = CuVector{Cint}([1, 3, 5, 8])
colVal = CuVector{Cint}([1, 3, 1, 2, 1, 2, 3])

# List of values for λ
Λ = [1.0, 10.0, -20.0]
if strided
    nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                         1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                         1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
else
    nzVal = CuMatrix{T}([1+Λ[1] 1+Λ[2] 1+Λ[3];
                         3      3      3     ;
                         4      4      4     ;
                         5+Λ[1] 5+Λ[2] 5+Λ[3];
                         2      2      2     ;
                         6      6      6     ;
                         2+Λ[1] 2+Λ[2] 2+Λ[3]])
end

cudss_xλ_gpu = CudssMatrix(T, n; nbatch)
cudss_bλ_gpu = CudssMatrix(T, n; nbatch)

if strided
    xλ_gpu = CuVector{T}(undef, n * nbatch)
    bλ_gpu = CuVector{T}([1.0, 2.0, 3.0,
                          4.0, 5.0, 6.0,
                          7.0, 8.0, 9.0])
else
    xλ_gpu = CuMatrix{T}(undef, n, nbatch)
    bλ_gpu = CuMatrix{T}([1.0 4.0 7.0;
                          2.0 5.0 8.0;
                          3.0 6.0 9.0])
end

cudss_update(cudss_xλ_gpu, xλ_gpu)
cudss_update(cudss_bλ_gpu, bλ_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "G", 'F')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_xλ_gpu, cudss_bλ_gpu)
cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)

rλ_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = strided ? nzVal[1 + (i-1) * nnzA : i * nnzA] : nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = strided ? bλ_gpu[1 + (i-1) * n : i * n] : bλ_gpu[:, i]
    x_gpu = strided ? xλ_gpu[1 + (i-1) * n : i * n] : xλ_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

# Refactorize all matrices of the uniform batch
Λ = [-2.0, -10.0, 30.0]
if strided
    new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                             1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                             1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
else
    new_nzVal = CuMatrix{T}([1+Λ[1] 1+Λ[2] 1+Λ[3];
                             3      3      3     ;
                             4      4      4     ;
                             5+Λ[1] 5+Λ[2] 5+Λ[3];
                             2      2      2     ;
                             6      6      6     ;
                             2+Λ[1] 2+Λ[2] 2+Λ[3]])
end

cudss_update(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)

for i = 1:nbatch
    nz = strided ? new_nzVal[1 + (i-1) * nnzA : i * nnzA] : new_nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = strided ? bλ_gpu[1 + (i-1) * n : i * n] : bλ_gpu[:, i]
    x_gpu = strided ? xλ_gpu[1 + (i-1) * n : i * n] : xλ_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu
```

## Batch LU -- generic API

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
R = real(T)
n = 3
nbatch = 3

# Batch of unsymmetric linear systems
#        [1+λ  0   3  ]
# A(λ) = [ 4  5+λ  0  ]
#        [ 2   6  2+λ ]
nnzA = 7
rowPtr = CuVector{Cint}([1, 3, 5, 8])
colVal = CuVector{Cint}([1, 3, 1, 2, 1, 2, 3])

# List of values for λ
Λ = [1.0, 10.0, -20.0]
nzVal = CuMatrix{T}([1+Λ[1] 1+Λ[2] 1+Λ[3];
                     3      3      3     ;
                     4      4      4     ;
                     5+Λ[1] 5+Λ[2] 5+Λ[3];
                     2      2      2     ;
                     6      6      6     ;
                     2+Λ[1] 2+Λ[2] 2+Λ[3]])

Aλ_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, vec(nzVal), (n,n));
F = lu(Aλ_gpu)

bλ_gpu = CuMatrix{T}([1.0 4.0 7.0;
                      2.0 5.0 8.0;
                      3.0 6.0 9.0])
xλ_gpu = CuMatrix{T}(undef, n, nbatch)
ldiv!(xλ_gpu, F, bλ_gpu)

rλ_gpu = Inf * ones(R, nbatch)
for i = 1:nbatch
    nz = nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[:, i]
    x_gpu = xλ_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

Λ = [-2.0, -10.0, 30.0]
new_nzVal = CuMatrix{T}([1+Λ[1] 1+Λ[2] 1+Λ[3];
                         3      3      3     ;
                         4      4      4     ;
                         5+Λ[1] 5+Λ[2] 5+Λ[3];
                         2      2      2     ;
                         6      6      6     ;
                         2+Λ[1] 2+Λ[2] 2+Λ[3]])

Aλ_gpu.nzVal = vec(new_nzVal)
lu!(F, Aλ_gpu)
ldiv!(xλ_gpu, F, bλ_gpu)

for i = 1:nbatch
    nz = new_nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[:, i]
    x_gpu = xλ_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu
```

## Batch LDLᵀ and LDLᴴ -- cuDSS API

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = ComplexF64
R = real(T)
n = 5
nbatch = 2
nrhs = 2
strided = false

nnzA = 8
rowPtr = CuVector{Cint}([1, 2, 3, 6, 7, 9])
colVal = CuVector{Cint}([1, 2, 1, 2, 3, 4, 3, 5])
if strided
    nzVal = CuVector{T}([4, 3, 1+im, 2-im, 5, 1, 1+im, 2,
                         2, 3, 1-im, 1+im, 6, 4, 2-im, 8])
else
    nzVal = CuMatrix{T}([4    2   ;
                         3    3   ;
                         1+im 1-im;
                         2-im 1+im;
                         5    6   ;
                         1    4   ;
                         1+im 2-im;
                         2    8   ])
end

cudss_Xs_gpu = CudssMatrix(T, n, nrhs; nbatch)
cudss_Bs_gpu = CudssMatrix(T, n, nrhs; nbatch)

if strided
    Xs_gpu = CuVector{T}(undef, n * nrhs * nbatch)
    Bs_gpu = CuVector{T}([ 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im,
                          13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im])
else
    Xs_gpu = CuArray{T}(undef, n, nrhs, nbatch)
    Bs_gpu = CuArray{T}([7  -7 ;
                         12 -12;
                         25 -25;
                         4  -4 ;
                         13 -13;;;
                         13 -13;
                         15 -15;
                         29 -29;
                         8  -8 ;
                         14 -14])
end

cudss_update(cudss_Xs_gpu, Xs_gpu)
cudss_update(cudss_Bs_gpu, Bs_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "H", 'L')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_Xs_gpu, cudss_Bs_gpu)
cudss("factorization", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)
cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)

Rs_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = strided ? nzVal[1 + (i-1) * nnzA : i * nnzA] : nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = strided ? reshape(Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs) : Bs_gpu[:, :, i]
    X_gpu = strided ? reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs) : Xs_gpu[:, :, i]
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu

if strided
    new_nzVal = CuVector{T}([-4, -3,  1-im, -2+im, -5, -1, -1-im, -2,
                             -2, -3, -1+im, -1-im, -6, -4, -2+im, -8])
else
    new_nzVal = CuMatrix{T}([-4    -2   ;
                             -3    -3   ;
                             -1-im -1+im;
                             -2+im -1-im;
                             -5    -6   ;
                             -1    -4   ;
                             -1-im -2+im;
                             -2    -8   ])
end

cudss_update(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)

if strided
new_Bs_gpu = CuVector{T}([13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im,
                           7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im])
else
new_Bs_gpu = CuArray{T}([13-im -13-im;
                         15-im -15-im;
                         29-im -29-im;
                          8-im  -8-im;
                         14-im -14-im;;;
                          7+im  -7+im;
                         12+im -12+im;
                         25+im -25+im;
                          4+im  -4+im;
                         13+im -13+im])
end

cudss_update(cudss_Bs_gpu, new_Bs_gpu)
cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)

for i = 1:nbatch
    nz = strided ? new_nzVal[1 + (i-1) * nnzA : i * nnzA] : new_nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = strided ? reshape(new_Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs) : new_Bs_gpu[:, :, i]
    X_gpu = strided ? reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs) : Xs_gpu[:, :, i]
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu
```

## Batch LDLᵀ and LDLᴴ -- generic API

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = ComplexF64
R = real(T)
n = 5
nbatch = 2
nrhs = 2
nnzA = 8
rowPtr = CuVector{Cint}([1, 2, 3, 6, 7, 9])
colVal = CuVector{Cint}([1, 2, 1, 2, 3, 4, 3, 5])
nzVal = CuMatrix{T}([4    2   ;
                     3    3   ;
                     1+im 1-im;
                     2-im 1+im;
                     5    6   ;
                     1    4   ;
                     1+im 2-im;
                     2    8   ])

As_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, vec(nzVal), (n,n));
F = ldlt(As_gpu; view='L')

Xs_gpu = CuArray{T}(undef, n, nrhs, nbatch)
Bs_gpu = CuArray{T}([7  -7 ;
                     12 -12;
                     25 -25;
                     4  -4 ;
                     13 -13;;;
                     13 -13;
                     15 -15;
                     29 -29;
                     8  -8 ;
                     14 -14])
ldiv!(Xs_gpu, F, Bs_gpu)

Rs_gpu = Inf * ones(R, nbatch)
for i = 1:nbatch
    nz = nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = Bs_gpu[:, :, i]
    X_gpu = Xs_gpu[:, :, i]
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu

new_nzVal = CuMatrix{T}([-4    -2   ;
                         -3    -3   ;
                         -1-im -1+im;
                         -2+im -1-im;
                         -5    -6   ;
                         -1    -4   ;
                         -1-im -2+im;
                         -2    -8   ])
As_gpu.nzVal = vec(new_nzVal)
ldlt!(F, As_gpu)

new_Bs_gpu = CuArray{T}([13-im -13-im;
                         15-im -15-im;
                         29-im -29-im;
                          8-im  -8-im;
                         14-im -14-im;;;
                          7+im  -7+im;
                         12+im -12+im;
                         25+im -25+im;
                          4+im  -4+im;
                         13+im -13+im])
new_Xs_gpu = copy(new_Bs_gpu)
ldiv!(F, new_Xs_gpu)

for i = 1:nbatch
    nz = new_nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = new_Bs_gpu[:, :, i]
    X_gpu = new_Xs_gpu[:, :, i]
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu
```

## Batch LLᵀ and LLᴴ -- cuDSS API

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
n = 5
nbatch = 2
strided = false

nnzA = 8
rowPtr = CuVector{Cint}([1, 3, 5, 7, 8, 9])
colVal = CuVector{Cint}([1, 3, 2, 3, 3, 5, 4, 5])
if strided
    nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                         2, 1, 3, 1, 6, 2, 4, 8])
else
    nzVal = CuMatrix{T}([4 2;
                         1 1;
                         3 3;
                         2 1;
                         5 6;
                         1 2;
                         1 4;
                         2 8])
end

cudss_xs_gpu = CudssMatrix(T, n; nbatch)
cudss_bs_gpu = CudssMatrix(T, n; nbatch)

if strided
    xs_gpu = CuVector{T}(undef, n * nbatch)
    bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,
                          13, 15, 29, 8, 14])
else
    xs_gpu = CuMatrix{T}(undef, n, nbatch)
    bs_gpu = CuMatrix{T}([7  13;
                          12 15;
                          25 29;
                          4  8 ;
                          13 14])
end

cudss_update(cudss_xs_gpu, xs_gpu)
cudss_update(cudss_bs_gpu, bs_gpu)

# Constructor for uniform batch of systems
solver = CudssSolver(rowPtr, colVal, nzVal, "SPD", 'U')

# Specify that it is a uniform batch of size "nbatch"
cudss_set(solver, "ubatch_size", nbatch)

cudss("analysis", solver, cudss_xs_gpu, cudss_bs_gpu)
cudss("factorization", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)
cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)

rs_gpu = rand(R, nbatch)
for i = 1:nbatch
    nz = strided ? nzVal[1 + (i-1) * nnzA : i * nnzA] : nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = strided ? bs_gpu[1 + (i-1) * n : i * n] : bs_gpu[:, i]
    x_gpu = strided ? xs_gpu[1 + (i-1) * n : i * n] : xs_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu

if strided
    new_nzVal = CuVector{T}([8, 2, 6, 4, 10, 2, 2 , 4 ,
                             6, 3, 9, 3, 18, 6, 12, 24])
else
    new_nzVal = CuMatrix{T}([8  6 ;
                             2  3 ;
                             6  9 ;
                             4  3 ;
                             10 18;
                             2  6 ;
                             2  12;
                             4  24])
end

cudss_update(solver, rowPtr, colVal, new_nzVal)
cudss("refactorization", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)
cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)

for i = 1:nbatch
    nz = strided ? new_nzVal[1 + (i-1) * nnzA : i * nnzA] : new_nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = strided ? bs_gpu[1 + (i-1) * n : i * n] : bs_gpu[:, i]
    x_gpu = strided ? xs_gpu[1 + (i-1) * n : i * n] : xs_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu
```

## Batch LLᵀ and LLᴴ -- generic API

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
R = real(T)
n = 5
nbatch = 2
nnzA = 8
rowPtr = CuVector{Cint}([1, 3, 5, 7, 8, 9])
colVal = CuVector{Cint}([1, 3, 2, 3, 3, 5, 4, 5])
nzVal = CuMatrix{T}([4 2;
                     1 1;
                     3 3;
                     2 1;
                     5 6;
                     1 2;
                     1 4;
                     2 8])

As_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, vec(nzVal), (n,n));
F = cholesky(As_gpu; view='U')

bs_gpu = CuMatrix{T}([ 7 13;
                      12 15;
                      25 29;
                       4  8;
                      13 14])
xs_gpu = CuMatrix{T}(undef, n, nbatch)
ldiv!(xs_gpu, F, bs_gpu)

rs_gpu = Inf * ones(R, nbatch)
for i = 1:nbatch
    nz = nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = bs_gpu[:, i]
    x_gpu = xs_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu

new_nzVal = CuMatrix{T}([8  6 ;
                         2  3 ;
                         6  9 ;
                         4  3 ;
                         10 18;
                         2  6 ;
                         2  12;
                         4  24])
As_gpu.nzVal = vec(new_nzVal)
cholesky!(F, As_gpu)
ldiv!(xs_gpu, F, bs_gpu)

for i = 1:nbatch
    nz = new_nzVal[:, i]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = bs_gpu[:, i]
    x_gpu = xs_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu
```

#### Generic interface

!!! note
    The generic interface for uniform batches requires CUDSS.jl v0.6.4 and above.

## LLᵀ and LLᴴ

```@docs
LinearAlgebra.cholesky(A::CuSparseMatrixCSR{T,INT}; view::Char='F') where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
LinearAlgebra.cholesky!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}) where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
```

!!! note
    If we only store one triangle of `A`, we can also use the wrappers `Symmetric` and `Hermitian` instead of using the keyword argument `view` in `cholesky`. For real matrices, both wrappers are allowed but only `Hermitian` can be used for complex matrices.

```julia
H = Hermitian(A, :U)
F = cholesky(H)
```

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = ComplexF64
R = real(T)
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.01)
A_cpu = A_cpu * A_cpu' + I
B_cpu = rand(T, n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
B_gpu = CuMatrix(B_cpu)
X_gpu = similar(B_gpu)

F = cholesky(A_gpu, view='U')
X_gpu = F \ B_gpu

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)

# In-place LLᴴ
d_gpu = rand(R, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cholesky!(F, A_gpu)

C_cpu = rand(T, n, p)
C_gpu = CuMatrix(C_cpu)
ldiv!(X_gpu, F, C_gpu)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

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
nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                     2, 1, 3, 1, 6, 2, 4, 8])

As_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, (n,n));
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
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    b_gpu = bs_gpu[:, i]
    x_gpu = xs_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rs_gpu[i] = norm(r_gpu)
end
rs_gpu

new_nzVal = CuVector{T}([8, 2, 6, 4, 10, 2,  2,  4,
                         6, 3, 9, 3, 18, 6, 12, 24])
As_gpu.nzVal = new_nzVal
cholesky!(F, As_gpu)
ldiv!(xs_gpu, F, bs_gpu)

for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
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

!!! note
    When solving a uniform batch of linear systems, `b` and `x` can be vectors, matrices, or tensors. Internally, they are always treated as a single long vector in memory, with elements arranged consistently with Julia's column-major order across all dimensions. This vector contains all right-hand sides of the batch in a strided layout, and the dimensions of each right-hand side as well as the number of systems are tracked automatically, allowing `b` and `x` to be passed directly to `ldiv!` without manual reshaping.

## LDLᵀ and LDLᴴ

```@docs
LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T,INT}; view::Char='F') where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
LinearAlgebra.ldlt!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}) where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
```

!!! note
    If we only store one triangle of `A_gpu`, we can also use the wrappers `Symmetric` and `Hermitian` instead of using the keyword argument `view` in `ldlt`. For real matrices, both wrappers are allowed but only `Hermitian` can be used for complex matrices.

```julia
S = Symmetric(A, :L)
F = ldlt(S)
```

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
R = real(T)
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.05) + I
A_cpu = A_cpu + A_cpu'
B_cpu = rand(T, n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
B_gpu = CuMatrix(B_cpu)
X_gpu = similar(B_gpu)

F = ldlt(A_gpu, view='L')
X_gpu = F \ B_gpu

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)

# In-place LDLᵀ
d_gpu = rand(R, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
ldlt!(F, A_gpu)

C_cpu = rand(T, n, p)
C_gpu = CuMatrix(C_cpu)
ldiv!(X_gpu, F, C_gpu)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

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
nzVal = CuVector{T}([4, 3, 1+im, 2-im, 5, 1, 1+im, 2,
                     2, 3, 1-im, 1+im, 6, 4, 2-im, 8])

As_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, (n,n));
F = ldlt(As_gpu; view='L')

Bs_gpu = CuVector{T}([ 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im,
                      13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im])
Bs_gpu = reshape(Bs_gpu, n, nrhs, nbatch)
Xs_gpu = CuArray{T}(undef, n, nrhs, nbatch)
ldiv!(Xs_gpu, F, Bs_gpu)

Rs_gpu = Inf * ones(R, nbatch)
for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    A_cpu = SparseMatrixCSC(A_gpu)
    A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
    B_gpu = Bs_gpu[:, :, i]
    X_gpu = Xs_gpu[:, :, i]
    R_gpu = B_gpu - A_gpu * X_gpu
    Rs_gpu[i] = norm(R_gpu)
end
Rs_gpu

new_nzVal = CuVector{T}([-4, -3,  1-im, -2+im, -5, -1, -1-im, -2,
                         -2, -3, -1+im, -1-im, -6, -4, -2+im, -8])
As_gpu.nzVal = new_nzVal
ldlt!(F, As_gpu)

new_Bs_gpu = CuVector{T}([13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im,
                          7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im])
new_Bs_gpu = reshape(new_Bs_gpu, n, nrhs, nbatch)
new_Xs_gpu = copy(new_Bs_gpu)
ldiv!(F, new_Xs_gpu)

for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
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

## LU

```@docs
LinearAlgebra.lu(A::CuSparseMatrixCSR{T,INT}) where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
LinearAlgebra.lu!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}) where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
```

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 100
A_cpu = sprand(T, n, n, 0.05) + I
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)

F = lu(A_gpu)
x_gpu = F \ b_gpu

r_gpu = b_gpu - A_gpu * x_gpu
norm(r_gpu)

# In-place LU
d_gpu = rand(T, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
lu!(F, A_gpu)

c_cpu = rand(T, n)
c_gpu = CuVector(c_cpu)
ldiv!(x_gpu, F, c_gpu)

r_gpu = c_gpu - A_gpu * x_gpu
norm(r_gpu)
```

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
nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                     1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                     1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])

Aλ_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, (n,n));
F = lu(Aλ_gpu)

bλ_gpu = CuMatrix{T}([1.0 4.0 7.0;
                      2.0 5.0 8.0;
                      3.0 6.0 9.0])
xλ_gpu = CuMatrix{T}(undef, n, nbatch)
ldiv!(xλ_gpu, F, bλ_gpu)

rλ_gpu = Inf * ones(R, nbatch)
for i = 1:nbatch
    nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[:, i]
    x_gpu = xλ_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu

Λ = [-2.0, -10.0, 30.0]
new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                         1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                         1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
Aλ_gpu.nzVal = new_nzVal
lu!(F, Aλ_gpu)
ldiv!(xλ_gpu, F, bλ_gpu)

for i = 1:nbatch
    nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
    A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
    b_gpu = bλ_gpu[:, i]
    x_gpu = xλ_gpu[:, i]
    r_gpu = b_gpu - A_gpu * x_gpu
    rλ_gpu[i] = norm(r_gpu)
end
rλ_gpu
```

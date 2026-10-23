#### Generic interface

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

## LDLᵀ and LDLᴴ

```@docs
LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T,INT}; view::Char='F', structure::String) where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
LinearAlgebra.ldlt!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}) where {T <: LinearAlgebra.BlasFloat, INT <: CUDSS.CudssInt}
```

!!! note
    If we only store one triangle of `A_gpu`, we can also use the wrappers `Symmetric` and `Hermitian` instead of using the keyword argument `view` in `ldlt`. For real matrices, both wrappers compute the same LDLᵀ factorization. For complex matrices, `Hermitian` computes an LDLᴴ factorization (the default of `ldlt`, structure `"H"`) while `Symmetric` treats the matrix as complex symmetric (`Aᵀ = A`) and computes an LDLᵀ factorization (structure `"S"`).

```julia
S = Symmetric(A, :L)
F = ldlt(S)
```

!!! note
    The keyword argument `structure` of `ldlt` accepts `"S"` or `"H"` only. Its default is `"S"` for real matrices and `"H"` for complex matrices, so it is only needed to factorize a complex symmetric matrix without the `Symmetric` wrapper.

```julia
F = ldlt(A_gpu, view='L', structure="S")
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

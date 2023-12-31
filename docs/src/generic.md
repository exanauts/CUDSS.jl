# Examples

## Example 1: Sparse unsymmetric linear system with one right-hand side

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
```

## Example 2: Sparse symmetric linear system with multiple right-hand sides

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.05) + I
A_cpu = A_cpu + A_cpu'
B_cpu = rand(T, n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
B_gpu = CuMatrix(B_cpu)
X_gpu = similar(B_gpu)

F = ldlt(A_gpu, view='L')
ldiv!(X_gpu, F, B_gpu)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)
```

!!! note
    If we only store one triangle of `A_gpu`, we can also use the wrappers `Symmetric` and `Hermitian`. For real matrices, both wrappers are allowed but only `Hermitian` can be used for complex matrices.

```julia
S_gpu = Symmetric(A_gpu, :L)
F = ldlt(S_gpu)
```

## Example 3: Sparse hermitian positive definite linear system with multiple right-hand sides

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = ComplexF64
n = 100
p = 5
A_cpu = sprand(T, n, n, 0.01)
A_cpu = A_cpu * A_cpu' + I
B_cpu = rand(T, n, p)

A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
B_gpu = CuMatrix(B_cpu)
X_gpu = similar(B_gpu)

F = cholesky(A_gpu, view='U')
ldiv!(X_gpu, F, B_gpu)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)
```

!!! note
    If we only store one triangle of `A_gpu`, we can also use the wrappers `Symmetric` and `Hermitian`. For real matrices, both wrappers are allowed but only `Hermitian` can be used for complex matrices.

```julia
H_gpu = Hermitian(A_gpu, :U)
F = cholesky(H_gpu)
```

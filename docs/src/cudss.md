#### Factorization of a single sparse matrix

Solving a single sparse linear system in cuDSS is performed in three distinct phases: symbolic analysis, numerical factorization, and solve.
The symbolic analysis determines the structure of the factors and computes a fill-reducing permutation based solely on the sparsity pattern of the matrix.
The numerical factorization operates on the matrix values using the symbolic information obtained in the first phase.
Once the factorization is complete, the solve phase computes the solution for one or more right-hand sides.
Both the factorization and solve phases can be efficiently repeated as long as the sparsity pattern of the matrix remains unchanged.

## LU

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
cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)
cudss("solve", solver, x_gpu, b_gpu; asynchronous=false)

r_gpu = b_gpu - A_gpu * x_gpu
norm(r_gpu)

# In-place LU
d_gpu = rand(T, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cudss_update(solver, A_gpu)

c_cpu = rand(T, n)
c_gpu = CuVector(c_cpu)

cudss("refactorization", solver, x_gpu, c_gpu; asynchronous=false)
cudss("solve", solver, x_gpu, c_gpu; asynchronous=false)

r_gpu = c_gpu - A_gpu * x_gpu
norm(r_gpu)
```

## LDLᵀ and LDLᴴ

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
cudss("factorization", solver, X_gpu, B_gpu; asynchronous=false)
cudss("solve", solver, X_gpu, B_gpu; asynchronous=false)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)

# In-place LDLᵀ
d_gpu = rand(R, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cudss_update(solver, A_gpu)

C_cpu = rand(T, n, p)
C_gpu = CuMatrix(C_cpu)

cudss("refactorization", solver, X_gpu, C_gpu; asynchronous=false)
cudss("solve", solver, X_gpu, C_gpu; asynchronous=false)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

## LLᵀ and LLᴴ

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
cudss("factorization", solver, X_gpu, B_gpu; asynchronous=false)
cudss("solve", solver, X_gpu, B_gpu; asynchronous=false)

R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
norm(R_gpu)

# In-place LLᴴ
d_gpu = rand(R, n) |> CuVector
A_gpu = A_gpu + Diagonal(d_gpu)
cudss_update(solver, A_gpu)

C_cpu = rand(T, n, p)
C_gpu = CuMatrix(C_cpu)

cudss("refactorization", solver, X_gpu, C_gpu; asynchronous=false)
cudss("solve", solver, X_gpu, C_gpu; asynchronous=false)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

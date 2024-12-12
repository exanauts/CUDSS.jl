## Batch LU

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
n = 100
nbatch = 5

batch_A_gpu = CuSparseMatrixCSR{T,Cint}[]
batch_x_gpu = CuVector{T}[]
batch_b_gpu = CuVector{T}[]

for i = 1:nbatch
    A_cpu = sprand(T, n, n, 0.05) + I
    x_cpu = zeros(T, n)
    b_cpu = rand(T, n)

    push!(batch_A_gpu, A_cpu |> CuSparseMatrixCSR)
    push!(batch_x_gpu, x_cpu |> CuVector)
    push!(batch_b_gpu, b_cpu |> CuVector)
end

solver = CudssSolver(batch_A_gpu, "G", 'F')

cudss("analysis", solver, batch_x_gpu, batch_b_gpu)
cudss("factorization", solver, batch_x_gpu, batch_b_gpu)
cudss("solve", solver, batch_x_gpu, batch_b_gpu)

batch_r_gpu = batch_b_gpu .- batch_A_gpu .* batch_x_gpu
norm.(r_gpu)

# In-place LU
for i = 1:nbatch
    d_gpu = rand(T, n) |> CuVector
    batch_A_gpu[i] = batch_A_gpu[i] + Diagonal(d_gpu)
end
cudss_set(solver, batch_A_gpu)

for i = 1:nbatch
    c_cpu = rand(T, n)
    c_gpu = CuVector(c_cpu)
    batch_b_gpu[i] = c_gpu
end

cudss("refactorization", solver, batch_x_gpu, batch_b_gpu)
cudss("solve", solver, batch_x_gpu, batch_b_gpu)

r_gpu = batch_b_gpu .- batch_A_gpu .* batch_x_gpu
norm.(r_gpu)
```

### Batch LDLᵀ and LDLᴴ

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
n = 100
p = 5
nbatch = 10

batch_A_gpu = CuSparseMatrixCSR{T,Cint}[]
batch_X_gpu = CuMatrix{T}[]
batch_B_gpu = CuMatrix{T}[]

for i = 1:nbatch
    A_cpu = sprand(T, n, n, 0.05) + I
    A_cpu = A_cpu + A_cpu'
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    push!(batch_A_gpu, A_cpu |> tril |> CuSparseMatrixCSR)
    push!(batch_X_gpu, X_cpu |> CuVector)
    push!(batch_B_gpu, B_cpu |> CuVector)
end

structure = T <: Real ? "S" : "H"
solver = CudssSolver(batch_A_gpu, structure, 'L')

cudss("analysis", solver, batch_X_gpu, batch_B_gpu)
cudss("factorization", solver, batch_X_gpu, batch_B_gpu)
cudss("solve", solver, batch_X_gpu, batch_B_gpu)

R_gpu = batch_B_gpu .- CuSparseMatrixCSR.(A_cpu) .* batch_X_gpu
norm.(R_gpu)

# In-place LDLᵀ
for i = 1:nbatch
    d_gpu = rand(R, n) |> CuVector
    batch_A_gpu[i] = batch_A_gpu[i] + Diagonal(d_gpu)
end
cudss_set(solver, A_gpu)

for i = 1:nbatch
    C_cpu = rand(T, n, p)
    C_gpu = CuMatrix(C_cpu)
    batch_B_gpu[i] = C_gpu
end

cudss("refactorization", solver, batch_X_gpu, batch_B_gpu)
cudss("solve", solver, batch_X_gpu, batch_B_gpu)

R_gpu = batch_B_gpu .- ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * batch_B_gpu
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

cudss("refactorization", solver, X_gpu, C_gpu)
cudss("solve", solver, X_gpu, C_gpu)

R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
norm(R_gpu)
```

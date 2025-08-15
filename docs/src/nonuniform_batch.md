# Batch factorization of matrices with different sparsity patterns

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

solver = CudssBatchedSolver(batch_A_gpu, "G", 'F')

cudss("analysis", solver, batch_x_gpu, batch_b_gpu)
cudss("factorization", solver, batch_x_gpu, batch_b_gpu)
cudss("solve", solver, batch_x_gpu, batch_b_gpu)

batch_r_gpu = batch_b_gpu .- batch_A_gpu .* batch_x_gpu
norm.(batch_r_gpu)

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

batch_r_gpu = batch_b_gpu .- batch_A_gpu .* batch_x_gpu
norm.(batch_r_gpu)
```

## Batch LDLᵀ and LDLᴴ

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = Float64
R = real(T)
n = 100
p = 5
nbatch = 10

batch_A_cpu = SparseMatrixCSC{T}[]
batch_A_gpu = CuSparseMatrixCSR{T,Cint}[]
batch_X_gpu = CuMatrix{T}[]
batch_B_gpu = CuMatrix{T}[]

for i = 1:nbatch
    A_cpu = sprand(T, n, n, 0.05) + I
    A_cpu = A_cpu + A_cpu'
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    push!(batch_A_cpu, A_cpu)
    push!(batch_A_gpu, A_cpu |> tril |> CuSparseMatrixCSR)
    push!(batch_X_gpu, X_cpu |> CuMatrix)
    push!(batch_B_gpu, B_cpu |> CuMatrix)
end

structure = T <: Real ? "S" : "H"
solver = CudssBatchedSolver(batch_A_gpu, structure, 'L')

cudss("analysis", solver, batch_X_gpu, batch_B_gpu)
cudss("factorization", solver, batch_X_gpu, batch_B_gpu)
cudss("solve", solver, batch_X_gpu, batch_B_gpu)

batch_R_gpu = batch_B_gpu .- CuSparseMatrixCSR.(batch_A_cpu) .* batch_X_gpu
norm.(batch_R_gpu)

# In-place LDLᵀ
d_cpu = rand(R, n)
d_gpu = CuVector(d_cpu)
for i = 1:nbatch
    batch_A_gpu[i] = batch_A_gpu[i] + Diagonal(d_gpu)
    batch_A_cpu[i] = batch_A_cpu[i] + Diagonal(d_cpu)
end
cudss_set(solver, batch_A_gpu)

for i = 1:nbatch
    C_cpu = rand(T, n, p)
    C_gpu = CuMatrix(C_cpu)
    batch_B_gpu[i] = C_gpu
end

cudss("refactorization", solver, batch_X_gpu, batch_B_gpu)
cudss("solve", solver, batch_X_gpu, batch_B_gpu)

batch_R_gpu = batch_B_gpu .- CuSparseMatrixCSR.(batch_A_cpu) .* batch_X_gpu
norm.(batch_R_gpu)
```

## Batch LLᵀ and LLᴴ

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

T = ComplexF64
R = real(T)
n = 100
p = 4
nbatch = 8

batch_A_cpu = SparseMatrixCSC{T}[]
batch_A_gpu = CuSparseMatrixCSR{T,Cint}[]
batch_X_gpu = CuMatrix{T}[]
batch_B_gpu = CuMatrix{T}[]

for i = 1:nbatch
    A_cpu = sprand(T, n, n, 0.05) + I
    A_cpu = A_cpu * A_cpu' + I
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    push!(batch_A_cpu, A_cpu)
    push!(batch_A_gpu, A_cpu |> triu |> CuSparseMatrixCSR)
    push!(batch_X_gpu, X_cpu |> CuMatrix)
    push!(batch_B_gpu, B_cpu |> CuMatrix)
end

structure = T <: Real ? "SPD" : "HPD"
solver = CudssBatchedSolver(batch_A_gpu, structure, 'U')

cudss("analysis", solver, batch_X_gpu, batch_B_gpu)
cudss("factorization", solver, batch_X_gpu, batch_B_gpu)
cudss("solve", solver, batch_X_gpu, batch_B_gpu)

batch_R_gpu = batch_B_gpu .- CuSparseMatrixCSR.(batch_A_cpu) .* batch_X_gpu
norm.(batch_R_gpu)

# In-place LLᴴ
d_cpu = rand(R, n)
d_gpu = CuVector(d_cpu)
for i = 1:nbatch
    batch_A_gpu[i] = batch_A_gpu[i] + Diagonal(d_gpu)
    batch_A_cpu[i] = batch_A_cpu[i] + Diagonal(d_cpu)
end
cudss_set(solver, batch_A_gpu)

for i = 1:nbatch
    C_cpu = rand(T, n, p)
    C_gpu = CuMatrix(C_cpu)
    batch_B_gpu[i] = C_gpu
end

cudss("refactorization", solver, batch_X_gpu, batch_B_gpu)
cudss("solve", solver, batch_X_gpu, batch_B_gpu)

batch_R_gpu = batch_B_gpu .- CuSparseMatrixCSR.(batch_A_cpu) .* batch_X_gpu
norm.(batch_R_gpu)
```

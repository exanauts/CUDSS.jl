#### Schur complement

## Dense Schur complement

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 5

# A = [A₁₁ A₁₂] where A₁₁ = [4 0], A₁₂ = [1 0 2]
#     [A₂₁ A₂₂]             [0 5]        [0 3 0]
#
# A₂₁ = [0 6] and A₂₂ = [8 0 0 ]
#       [7 0]           [0 9 1 ]
#       [0 0]           [0 1 10]
#
# The matrix A and the Schur complement S = A₂₂ − A₂₁(A₁₁)⁻¹A₁₂ are:
#
#     [4 0 1 0 2 ]
#     [0 5 0 3 0 ]          [ 8    -3.6  0  ]
# A = [0 6 8 0 0 ]  and S = [-1.75  9   -2.5]
#     [7 0 0 9 1 ]          [ 0     1    10 ]
#     [0 0 0 1 10]

rows = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5]
cols = [1, 3, 5, 2, 4, 2, 3, 1, 4, 5, 4, 5]
vals = [4.0, 1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 1.0, 1.0, 10.0]
A_cpu = sparse(rows, cols, vals, n, n)

# Right-hand side such the solution is a vector of ones
b_cpu = [7.0, 8.0, 14.0, 17.0, 11.0]

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector{Float64}(undef, n)
b_gpu = CuVector(b_cpu)
solver = CudssSolver(A_gpu, "G", 'F')

# Enable the Schur complement computation
cudss_set(solver, "schur_mode", 1)

# Rows and columns for the Schur complement of the block A₂₂
schur_indices = Cint[0, 0, 1, 1, 1]
cudss_set(solver, "user_schur_indices", schur_indices)

# Compute the Schur complement
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)

# Dimension of the Schur complement nₛ and the number of nonzeros
(nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

# Dense storage for the Schur complement
S_gpu = CuMatrix{Float64}(undef, nrows_S, ncols_S)
cudss_matrix = CudssMatrix(S_gpu)

# Update the matrix S_gpu
cudss_set(solver, "schur_matrix", cudss_matrix.matrix)
cudss_get(solver, "schur_matrix")

# [A₁₁ A₁₂] [x₁] = [b₁] ⟺ A₁₁x₁ = b₁ - A₁₂x₂
# [A₂₁ A₂₂] [x₂]   [b₂]   Sx₂ = b₂ - A₂₁(A₁₁)⁻¹b₁ = bₛ

# Compute bₛ with a partial forward solve
# bₛ is stored in the last nₛ components of b_gpu
cudss("solve_fwd_schur", solver, x_gpu, b_gpu)

# Compute x₂ with the dense LU of cuSOLVER
nₛ = 3
bₛ = x_gpu[n-nₛ+1:n]
x₂ = S_gpu \ bₛ

# Compute x₁ with a partial backward solve
# x₂ must be store the last nₛ components of x_gpu
x_gpu[n-nₛ+1:n] .= x₂
cudss("solve_bwd_schur", solver, b_gpu, x_gpu)
```

## Sparse Schur complement

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 5

# A = [A₁₁ A₁₂] where A₁₁ = [4 0], A₁₂ = [1 0 2]
#     [A₂₁ A₂₂]             [0 5]        [0 3 0]
#
# A₂₁ = [0 6] and A₂₂ = [8 0 0 ]
#       [7 0]           [0 9 1 ]
#       [0 0]           [0 1 10]
#
# The matrix A and the Schur complement S = A₂₂ − A₂₁(A₁₁)⁻¹A₁₂ are:
#
#     [4 0 1 0 2 ]
#     [0 5 0 3 0 ]          [ 8    -3.6  0  ]
# A = [0 6 8 0 0 ]  and S = [-1.75  9   -2.5]
#     [7 0 0 9 1 ]          [ 0     1    10 ]
#     [0 0 0 1 10]

rows = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5]
cols = [1, 3, 5, 2, 4, 2, 3, 1, 4, 5, 4, 5]
vals = [4.0, 1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 1.0, 1.0, 10.0]
A_cpu = sparse(rows, cols, vals, n, n)

# Right-hand side such the solution is a vector of ones
b_cpu = [7.0, 8.0, 14.0, 17.0, 11.0]

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector{Float64}(undef, n)
b_gpu = CuVector(b_cpu)
solver = CudssSolver(A_gpu, "G", 'F')

# Enable the Schur complement computation
cudss_set(solver, "schur_mode", 1)

# Rows and columns for the Schur complement of the block A₂₂
schur_indices = Cint[0, 0, 1, 1, 1]
cudss_set(solver, "user_schur_indices", schur_indices)

# Compute the Schur complement
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)

# Dimension of the Schur complement nₛ and the number of nonzeros
(nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

# Sparse storage for the Schur complement
rowPtr = CuVector{Cint}(undef, nrows_S+1)
colVal = CuVector{Cint}(undef, nnz_S)
nzVal = CuVector{T}(undef, nnz_S)
cudss_matrix = CudssMatrix(rowPtr, colVal, nzVal, "G", 'F')

# Update the vectors rowPtr, colVal and nzVal
cudss_set(solver, "schur_matrix", cudss_matrix.matrix)
cudss_get(solver, "schur_matrix")

# Schur complement stored as a CuSparseMatrixCSR
dim_S = (nrows_S, ncols_S)
S_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, dim_S)
```

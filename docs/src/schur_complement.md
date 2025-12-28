#### Schur complement

!!! danger
  With cuDSS version 0.7.1, the Schur complement functionality is partially buggy. As a result, Schur complement support in `CUDSS.jl` is currently considered experimental. Users are advised to use this feature with caution, carefully validate results, and avoid relying on it in production workflows until the issues are resolved in a future cuDSS release.

!!! info
    If the user requests sparse storage for a Hermitian Schur complement (after partial Cholesky or LDLᴴ decomposition), only one triangular part of the matrix can be recovered when dumping the matrix.

!!! warning
    When requesting sparse storage for a Hermitian Schur complement (after a partial Cholesky or LDLᴴ decomposition), the number of nonzeros reported by `cudss_get(solver, "schur_shape")` may be incorrect.

## Schur complement -- LU

```julia
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 5
dense_schur = false

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

rows = Cint[1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5]
cols = Cint[1, 3, 5, 2, 4, 2, 3, 1, 4, 5, 4, 5]
vals = T[4.0, 1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 1.0, 1.0, 10.0]
A_cpu = sparse(rows, cols, vals, n, n)

# Right-hand side such the solution is a vector of ones
b_cpu = T[7.0, 8.0, 14.0, 17.0, 11.0]

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector{T}(undef, n)
b_gpu = CuVector{T}(b_cpu)
solver = CudssSolver(A_gpu, "G", 'F')

# Enable the Schur complement computation
cudss_set(solver, "schur_mode", 1)

# Rows and columns for the Schur complement of the block A₂₂
schur_indices = Cint[0, 0, 1, 1, 1]
cudss_set(solver, "user_schur_indices", schur_indices)

# Compute the Schur complement with a partial factorization
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)

# Dimension of the Schur complement nₛ and the number of nonzeros
(nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

if dense_schur
  # Dense storage for the Schur complement
  S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
  S_cudss_dense = CudssMatrix(S_gpu)

  # Update the dense matrix S_gpu
  cudss_set(solver, "schur_matrix", S_cudss_dense.matrix)
  cudss_get(solver, "schur_matrix")
else
  # Sparse storage for the Schur complement
  S_rowPtr = CuVector{Cint}(undef, nrows_S+1)
  S_colVal = CuVector{Cint}(undef, nnz_S)
  S_nzVal = CuVector{T}(undef, nnz_S)
  dim_S = (nrows_S, ncols_S)
  S_gpu = CuSparseMatrixCSR(S_rowPtr, S_colVal, S_nzVal, dim_S)
  S_cudss_sparse = CudssMatrix(S_gpu, "G", 'F')

  # Update the sparse matrix S_gpu
  cudss_set(solver, "schur_matrix", S_cudss_sparse.matrix)
  cudss_get(solver, "schur_matrix")
end

# [A₁₁ A₁₂] [x₁] = [b₁] ⟺ A₁₁x₁ = b₁ - A₁₂x₂
# [A₂₁ A₂₂] [x₂]   [b₂]   Sx₂ = b₂ - A₂₁(A₁₁)⁻¹b₁ = bₛ
#
# Compute bₛ with a partial forward solve
# bₛ is stored in the last nₛ components of b_gpu
# nₛ = 3 is the size of the Schur complement
ns = 3
cudss("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous=false)
bs_gpu = x_gpu[n-ns+1:n]

if dense_schur
  # Compute x₂ with the dense LU of cuSOLVER
  F_gpu = copy(S_gpu)
  _, ipiv, _ = CUSOLVER.getrf!(F_gpu)
  x2_gpu = copy(bs_gpu)
  CUSOLVER.getrs!('N', F_gpu, ipiv, x2_gpu)
else
  # Compute x₂ with the sparse LU of cuDSS
  F_gpu = lu(S_gpu)
  x2_gpu = F_gpu \ bs_gpu
end

# Compute x₁ with a partial backward solve
# x₂ must be stored in the last nₛ components of x_gpu
x_gpu[n-ns+1:n] .= x2_gpu
cudss("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous=false)
x1_gpu = b_gpu[1:n-ns]
```

## Schur complement -- LDLᵀ and LDLᴴ

```julia
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 5
dense_schur = true

# A = [A₁₁ A₁₂] where A₁₁ = [4 0], A₁₂ = [1 0 2]
#     [A₂₁ A₂₂]             [0 2]        [0 3 0]
#
# A₂₁ = [1 0] and A₂₂ = [3 0 0]
#       [0 3]           [0 2 1]
#       [2 0]           [0 1 2]
#
# The matrix A and the Schur complement S = A₁₁ − A₁₂(A₂₂)⁻¹A₂₁ are:
#
#     [4 0 1 0 2]
#     [0 2 0 3 0]
# A = [1 0 3 0 0]  and S = [1   2]
#     [0 3 0 2 1]          [2  -4]
#     [2 0 0 1 2]

rows = Cint[1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
cols = Cint[1, 3, 5, 2, 4, 1, 3, 2, 4, 5, 1, 4, 5]
vals = T[4.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0]
A_cpu = sparse(rows, cols, vals, n, n) |> tril

# Right-hand side such the solution is a vector of ones
b_cpu = T[7.0, 5.0, 4.0, 6.0, 5.0]

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector{T}(undef, n)
b_gpu = CuVector{T}(b_cpu)
structure = T <: Real ? "S" : "H"
solver = CudssSolver(A_gpu, structure, 'L')

# Enable the Schur complement computation
cudss_set(solver, "schur_mode", 1)

# Rows and columns for the Schur complement of the block A₁₁
schur_indices = Cint[1, 1, 0, 0, 0]
cudss_set(solver, "user_schur_indices", schur_indices)

# Compute the Schur complement
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)

# Dimension of the Schur complement nₛ and the number of nonzeros
(nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

if dense_schur
  # Dense storage for the Schur complement
  S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
  S_cudss_dense = CudssMatrix(S_gpu)

  # Update the dense matrix S_gpu
  cudss_set(solver, "schur_matrix", S_cudss_dense.matrix)
  cudss_get(solver, "schur_matrix")
else
  # Maximum number of nonzeros in one triangle of the Schur complement
  nnz_S = min(nnz_S, nrows_S * (nrows_S + 1) ÷ 2)

  # Sparse storage for the Schur complement
  S_rowPtr = CuVector{Cint}(undef, nrows_S+1)
  S_colVal = CuVector{Cint}(undef, nnz_S)
  S_nzVal = CuVector{T}(undef, nnz_S)
  dim_S = (nrows_S, ncols_S)
  S_gpu = CuSparseMatrixCSR(S_rowPtr, S_colVal, S_nzVal, dim_S)
  S_cudss_sparse = CudssMatrix(S_gpu, structure, 'L')

  # Update the sparse matrix S_gpu
  cudss_set(solver, "schur_matrix", S_cudss_sparse.matrix)
  cudss_get(solver, "schur_matrix")
end

# [A₁₁ A₁₂] [x₁] = [b₁] ⟺ Sx₁ = b₁ - A₁₂(A₂₂)⁻¹b₂ = bₛ
# [A₂₁ A₂₂] [x₂]   [b₂]   A₂₂x₂ = b₂ - A₂₁x₁
#
# Compute bₛ with a partial forward solve
# bₛ is stored in the last nₛ components of x_gpu
# nₛ = 2 is the size of the Schur complement
ns = 2
cudss("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous=false)
cudss("solve_diag", solver, x_gpu, x_gpu; asynchronous=false)
bs_gpu = x_gpu[n-ns+1:n]

if dense_schur
  # Compute x₁ with the dense LDLᵀ / LDLᴴ of cuSOLVER
  F_gpu = copy(S_gpu)
  _, ipiv, _ = CUSOLVER.sytrf!('L', F_gpu)
  x1_gpu = copy(bs_gpu)
  CUSOLVER.sytrs!('L', F_gpu, CuVector{Int64}(ipiv), x1_gpu)
else
  # Compute x₁ with the sparse LDLᵀ / LDLᴴ of cuDSS
  F_gpu = ldlt(S_gpu, view='L')
  x1_gpu = F_gpu \ bs_gpu
end

# Compute x₂ with a partial backward solve
# x₁ must be stored in the last nₛ components of x_gpu
x_gpu[n-ns+1:n] .= x1_gpu
cudss("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous=false)
x2_gpu = b_gpu[ns+1:n]
```

## Schur complement -- LLᵀ and LLᴴ

```julia
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using CUDSS
using LinearAlgebra
using SparseArrays

T = Float64
n = 5
dense_schur = true

# A = [A₁₁ A₁₂] where A₁₁ = [2.5  1 ], A₁₂ = [1 0 0]
#     [A₂₁ A₂₂]             [ 1  2.5]        [0 1 0]
#
# A₂₁ = [1 0] and A₂₂ = [2 0 0]
#       [0 1]           [0 2 0]
#       [0 0]           [0 0 2]
#
# The matrix A and the Schur complement S = A₁₁ − A₁₂(A₂₂)⁻¹A₂₁ are:
#
#     [2.5  1   1  0  0]
#     [ 1  2.5  0  1  0]
# A = [ 1   0   2  0  0]  and S = [2 1]
#     [ 0   1   0  2  0]          [1 2]
#     [ 0   0   0  0  2]

rows = Cint[1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
cols = Cint[1, 2, 3, 1, 2, 4, 1, 3, 2, 4, 5]
vals = T[2.5, 1, 1, 1, 2.5, 1, 1, 2, 1, 2, 2]
A_cpu = sparse(rows, cols, vals, n, n) |> triu

# Right-hand side such the solution is a vector of ones
b_cpu = T[4.5, 4.5, 3.0, 3.0, 2.0]

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector{T}(undef, n)
b_gpu = CuVector{T}(b_cpu)
structure = T <: Real ? "SPD" : "HPD"
solver = CudssSolver(A_gpu, structure, 'U')

# Enable the Schur complement computation
cudss_set(solver, "schur_mode", 1)

# Rows and columns for the Schur complement of the block A₁₁
schur_indices = Cint[1, 1, 0, 0, 0]
cudss_set(solver, "user_schur_indices", schur_indices)

# Compute the Schur complement
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)

# Dimension of the Schur complement nₛ and the number of nonzeros
(nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

if dense_schur
  # Dense storage for the Schur complement
  S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
  S_cudss_dense = CudssMatrix(S_gpu)

  # Update the dense matrix S_gpu
  cudss_set(solver, "schur_matrix", S_cudss_dense.matrix)
  cudss_get(solver, "schur_matrix")
else
  # Maximum number of nonzeros in one triangle of the Schur complement
  nnz_S = min(nnz_S, nrows_S * (nrows_S + 1) ÷ 2)

  # Sparse storage for the Schur complement
  S_rowPtr = CuVector{Cint}(undef, nrows_S+1)
  S_colVal = CuVector{Cint}(undef, nnz_S)
  S_nzVal = CuVector{T}(undef, nnz_S)
  dim_S = (nrows_S, ncols_S)
  S_gpu = CuSparseMatrixCSR(S_rowPtr, S_colVal, S_nzVal, dim_S)
  S_cudss_sparse = CudssMatrix(S_gpu, structure, 'U')

  # Update the sparse matrix S_gpu
  cudss_set(solver, "schur_matrix", S_cudss_sparse.matrix)
  cudss_get(solver, "schur_matrix")
end

# [A₁₁ A₁₂] [x₁] = [b₁] ⟺ Sx₁ = b₁ - A₁₂(A₂₂)⁻¹b₂ = bₛ
# [A₂₁ A₂₂] [x₂]   [b₂]   A₂₂x₂ = b₂ - A₂₁x₁
#
# Compute bₛ with a partial forward solve
# bₛ is stored in the last nₛ components of x_gpu
# nₛ = 2 is the size of the Schur complement
ns = 2
cudss("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous=false)
bs_gpu = x_gpu[n-ns+1:n]

if dense_schur
  # Compute x₁ with the dense LLᵀ / LLᴴ of cuSOLVER
  F_gpu = copy(S_gpu)
  CUSOLVER.potrf!('U', F_gpu)
  x1_gpu = copy(bs_gpu)
  CUSOLVER.potrs!('U', F_gpu, x1_gpu)
else
  # Compute x₁ with the sparse LLᵀ / LLᴴ of cuDSS
  F_gpu = cholesky(S_gpu, view='U')
  x1_gpu = F_gpu \ bs_gpu
end

# Compute x₂ with a partial backward solve
# x₁ must be stored in the last nₛ components of x_gpu
x_gpu[n-ns+1:n] .= x1_gpu
cudss("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous=false)
x2_gpu = b_gpu[ns+1:n]
```

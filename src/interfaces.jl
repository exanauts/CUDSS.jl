export CudssSolver, cudss, cudss_set, cudss_get

"""
    solver = CudssSolver(A::CuSparseMatrixCSR{T}, structure::String, view::Char; index::Char='O')
    solver = CudssSolver(matrix::CudssMatrix{T}, config::CudssConfig, data::CudssData)

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

`CudssSolver` contains all structures required to solve linear systems with cuDSS.
One constructor of `CudssSolver` takes as input the same parameters as [`CudssMatrix`](@ref).

`structure` specifies the stucture for sparse matrices:
- `"G"`: General matrix -- LDU factorization;
- `"S"`: Real symmetric matrix -- LDLᵀ factorization;
- `"H"`: Complex Hermitian matrix -- LDLᴴ factorization;
- `"SPD"`: Symmetric positive-definite matrix -- LLᵀ factorization;
- `"HPD"`: Hermitian positive-definite matrix -- LLᴴ factorization.

`view` specifies matrix view for sparse matrices:
- `'L'`: Lower-triangular matrix and all values above the main diagonal are ignored;
- `'U'`: Upper-triangular matrix and all values below the main diagonal are ignored;
- `'F'`: Full matrix.

`index` specifies indexing base for sparse matrix indices:
- `'Z'`: 0-based indexing;
- `'O'`: 1-based indexing.

`CudssSolver` can be also constructed from the three structures `CudssMatrix`, `CudssConfig` and `CudssData` if needed.
"""
mutable struct CudssSolver{T}
  matrix::CudssMatrix{T}
  config::CudssConfig
  data::CudssData

  function CudssSolver(matrix::CudssMatrix{T}, config::CudssConfig, data::CudssData) where T <: BlasFloat
    return new{T}(matrix, config, data)
  end

  function CudssSolver(A::CuSparseMatrixCSR{T}, structure::String, view::Char; index::Char='O') where T <: BlasFloat
    matrix = CudssMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return new{T}(matrix, config, data)
  end
end

"""
    cudss_set(matrix::CudssMatrix{T}, v::CuVector{T})
    cudss_set(matrix::CudssMatrix{T}, A::CuMatrix{T})
    cudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T})
    cudss_set(data::CudssSolver, param::String, value)
    cudss_set(config::CudssConfig, param::String, value)
    cudss_set(data::CudssData, param::String, value)

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

The available configuration parameters are:
- `"reordering_alg"`: Algorithm for the reordering phase;
- `"factorization_alg"`: Algorithm for the factorization phase;
- `"solve_alg"`: Algorithm for the solving phase;
- `"matching_type"`: Type of matching;
- `"solve_mode"`: Potential modificator on the system matrix (transpose or adjoint);
- `"ir_n_steps"`: Number of steps during the iterative refinement;
- `"ir_tol"`: Iterative refinement tolerance;
- `"pivot_type"`: Type of pivoting (`'C'`, `'R'` or `'N'`);
- `"pivot_threshold"`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `"pivot_epsilon"`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `"max_lu_nnz"`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices.

The available data parameter is:
- `"user_perm"`: User permutation to be used instead of running the reordering algorithms.
"""
function cudss_set end

function cudss_set(matrix::CudssMatrix{T}, v::CuVector{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, v)
end

function cudss_set(matrix::CudssMatrix{T}, A::CuMatrix{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, A)
end

function cudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T}) where T <: BlasFloat
  cudssMatrixSetCsrPointers(matrix, A.rowPtr, CU_NULL, A.colVal, A.nzVal)
end

function cudss_set(solver::CudssSolver, param::String, value)
  (param ∈ CUDSS_CONFIG_PARAMETERS) && cudss_set(solver.config, param, value)
  (param ∈ CUDSS_DATA_PARAMETERS) && cudss_set(solver.data, param, value)
end

function cudss_set(data::CudssData, param::String, value)
  type = CUDSS_TYPES[param]
  val = Ref{type}(value)
  nbytes = sizeof(val)
  cudssDataSet(handle(), data, param, val, nbytes)
end

function cudss_set(config::CudssConfig, param::String, value)
  type = CUDSS_TYPES[param]
  val = Ref{type}(value)
  nbytes = sizeof(val)
  cudssConfigSet(config, param, val, nbytes)
end

"""
    value = cudss_get(data::CudssSolver, param::String)
    value = cudss_get(config::CudssConfig, param::String)
    value = cudss_get(data::CudssData, param::String)

The available configuration parameters are:
- `"reordering_alg"`: Algorithm for the reordering phase;
- `"factorization_alg"`: Algorithm for the factorization phase;
- `"solve_alg"`: Algorithm for the solving phase;
- `"matching_type"`: Type of matching;
- `"solve_mode"`: Potential modificator on the system matrix (transpose or adjoint);
- `"ir_n_steps"`: Number of steps during the iterative refinement;
- `"ir_tol"`: Iterative refinement tolerance;
- `"pivot_type"`: Type of pivoting (`'C'`, `'R'` or `'N'`);
- `"pivot_threshold"`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `"pivot_epsilon"`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `"max_lu_nnz"`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices.

The available data parameters are:
- `"info"`: Device-side error information;
- `"lu_nnz"`: Number of non-zero entries in LU factors;
- `"npivots"`: Number of pivots encountered during factorization;
- `"inertia"`: Tuple of positive and negative indices of inertia for symmetric and hermitian non positive-definite matrix types;
- `"perm_reorder"`: Reordering permutation;
- `"perm_row"`: Final row permutation (which includes effects of both reordering and pivoting);
- `"perm_col"`: Final column permutation (which includes effects of both reordering and pivoting);
- `"diag"`: Diagonal of the factorized matrix.

The data parameters `"info"`, `"lu_nnz"` and `"perm_reorder"` require the phase `"analyse"` performed by [`cudss`](@ref).
The data parameters `"npivots"`, `"inertia"` and `"diag"` require the phases `"analyse"` and `"factorization"` performed by [`cudss`](@ref).
The data parameters `"perm_row"` and `"perm_col"` are available but not yet functional.
"""
function cudss_get end

function cudss_get(solver::CudssSolver, param::String)
  (param ∈ CUDSS_CONFIG_PARAMETERS) && cudss_get(solver.config, param)
  (param ∈ CUDSS_DATA_PARAMETERS) && cudss_get(solver.data, param)
end

function cudss_get(data::CudssData, param::String)
  type = CUDSS_TYPES[param]
  val = Ref{type}()
  nbytes = sizeof(val)
  nbytes_written = Ref{Cint}()
  cudssDataGet(handle(), data, param, val, nbytes, nbytes_written)
  return val[]
end

function cudss_get(config::CudssConfig, param::String)
  type = CUDSS_TYPES[param]
  val = Ref{type}()
  nbytes = sizeof(val)
  nbytes_written = Ref{Cint}()
  cudssConfigGet(config, param, val, nbytes, nbytes_written)
  return val[]
end

"""
    cudss(phase::String, solver::CudssSolver{T}, x::CuVector{T}, b::CuVector{T})
    cudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T})
    cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T})

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

The available phases are `"analysis"`, `"factorization"`, `"refactorization"` and `"solve"`.
The phases `"solve_fwd"`, `"solve_diag"` and `"solve_bwd"` are available but not yet functional.
"""
function cudss end

function cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T}) where T <: BlasFloat
  cudssExecute(handle(), phase, solver.config, solver.data, solver.matrix, X, B)
end

function cudss(phase::String, solver::CudssSolver{T}, x::CuVector{T}, b::CuVector{T}) where T <: BlasFloat
  solution = CudssMatrix(x)
  rhs = CudssMatrix(b)
  cudss(phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T}) where T <: BlasFloat
  solution = CudssMatrix(X)
  rhs = CudssMatrix(B)
  cudss(phase, solver, solution, rhs)
end

export CudssSolver, cudss, cudss_set, cudss_get

mutable struct CudssSolver
  matrix::CudssMatrix
  config::CudssConfig
  data::CudssData

  function CudssSolver(matrix::CudssMatrix, config::CudssConfig, data::CudssData)
    return new(matrix, config, data)
  end

  function CudssSolver(A::CuSparseMatrixCSR, structure::Union{Char, String}, view::Char; index::Char='O')
    matrix = CudssMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return new(matrix, config, data)
  end
end

"""
    cudss_set(matrix::CudssMatrix, A::CuVector)
    cudss_set(matrix::CudssMatrix, A::CuMatrix)
    cudss_set(matrix::CudssMatrix, A::CuSparseMatrixCSR)
    cudss_set(data::CudssSolver, param::String, value)
    cudss_set(config::CudssConfig, param::String, value)
    cudss_set(data::CudssData, param::String, value)

The available config parameters are:
"reordering_alg"
"factorization_alg"
"solve_alg"
"matching_type"
"solve_mode"
"ir_n_steps"
"ir_tol"
"pivot_type"
"pivot_threshold"
"pivot_epsilon"
"max_lu_nnz"

The available data parameters are:
"info"
"lu_nnz"
"npivots"
"inertia"
"perm_reorder"
"perm_row"
"perm_col"
"diag"
"user_perm"
"""
function cudss_set end

function cudss_set(matrix::CudssMatrix, v::CuVector)
  cudssMatrixSetValues(matrix, v)
end

function cudss_set(matrix::CudssMatrix, A::CuMatrix)
  cudssMatrixSetValues(matrix, A)
end

function cudss_set(matrix::CudssMatrix, A::CuSparseMatrixCSR)
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

The available config parameters are:
"reordering_alg"
"factorization_alg"
"solve_alg"
"matching_type"
"solve_mode"
"ir_n_steps"
"ir_tol"
"pivot_type"
"pivot_threshold"
"pivot_epsilon"
"max_lu_nnz"

The available data parameters are:
"info"
"lu_nnz"
"npivots"
"inertia"
"perm_reorder"
"perm_row"
"perm_col"
"diag"
"user_perm"
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
    cudss(phase::String, solver::CudssSolver, x::CuVector, b::CuVector)
    cudss(phase::String, solver::CudssSolver, X::CuMatrix, B::CuMatrix)
    cudss(phase::String, solver::CudssSolver, X::CudssMatrix, B::CudssMatrix)

The available phases are "analysis", "factorization", "refactorization" and "solve".
The phases "solve_fwd", "solve_diag" and "solve_bwd" are available but not yet functional.
"""
function cudss end

function cudss(phase::String, solver::CudssSolver, x::CuVector, b::CuVector)
  solution = CudssMatrix(x)
  rhs = CudssMatrix(b)
  cudssExecute(handle(), phase, solver.config, solver.data, solver.matrix, solution, rhs)
end

function cudss(phase::String, solver::CudssSolver, X::CuMatrix, B::CuMatrix)
  solution = CudssMatrix(X)
  rhs = CudssMatrix(B)
  cudssExecute(handle(), phase, solver.config, solver.data, solver.matrix, solution, rhs)
end

function cudss(phase::String, solver::CudssSolver, X::CudssMatrix, B::CudssMatrix)
  cudssExecute(handle(), phase, solver.config, solver.data, solver.matrix, X, B)
end

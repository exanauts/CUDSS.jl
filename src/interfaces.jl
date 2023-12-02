export cudss, cudss_set, cudss_get

"""
  cudss_set(matrix::CudssMatrix, A::CuVector)
  cudss_set(matrix::CudssMatrix, A::CuMatrix)
  cudss_set(matrix::CudssMatrix, A::CuSparseMatrixCSR)

  cudss_set(config::CudssConfig, param::String, value)

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

  cudss_set(data::CudssData, param::String, value)

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
  cudssMatrixSetCsrPointers(matrix, A.rowPtr[1:end-1],
                            A.rowPtr[2:end], A.colVal, A.nzVal)
end

function cudss_set(data::CudssData, param::String, value)
  type = cudss_types[param]
  val = Ref{type}(value)
  nbytes = sizeof(val)
  cudssDataSet(handle(), data, param, val, nbytes)
end

function cudss_set(config::CudssConfig, param::String, value)
  type = cudss_types[param]
  val = Ref{type}(value)
  nbytes = sizeof(val)
  cudssConfigSet(config, param, val, nbytes)
end

"""
  value = cudss_get(config::CudssConfig, param::String)

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

  value = cudss_get(data::CudssData, param::String)

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

function cudss_get(data::CudssData, param::String)
  type = cudss_types[param]
  val = Ref{type}()
  nbytes = sizeof(val)
  nbytes_written = Ref{Cint}()
  cudssDataGet(handle(), data, param, val, nbytes, nbytes_written)
  return val[]
end

function cudss_get(config::CudssConfig, param::String)
  type = cudss_types[param]
  val = Ref{type}()
  nbytes = sizeof(val)
  nbytes_written = Ref{Cint}()
  cudssConfigGet(config, param, val, nbytes, nbytes_written)
  return val[]
end

"""
  cudss(phase::String, config::CudssConfig, data::CudssData, matrix::CudssMatrix, solution::CudssMatrix, rhs::CudssMatrix)

The available phases are:
"analysis"
"factorization"
"refactorization"
"solve"
"solve_fwd"
"solve_diag"
"solve_bwd"
"""
function cudss(phase::String, config::CudssConfig, data::CudssData, matrix::CudssMatrix, solution::CudssMatrix, rhs::CudssMatrix)
  cudssExecute(handle(), phase, config, data, matrix, solution, rhs)
end

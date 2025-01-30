export CudssSolver, cudss, cudss_set, cudss_get

"""
    solver = CudssSolver(A::CuSparseMatrixCSR{T,Cint}, structure::String, view::Char; index::Char='O')
    solver = CudssSolver(A::Vector{CuSparseMatrixCSR{T,Cint}}, structure::String, view::Char; index::Char='O')
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

  function CudssSolver(A::CuSparseMatrixCSR{T,Cint}, structure::String, view::Char; index::Char='O') where T <: BlasFloat
    matrix = CudssMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return new{T}(matrix, config, data)
  end

  function CudssSolver(A::Vector{CuSparseMatrixCSR{T,Cint}}, structure::String, view::Char; index::Char='O') where T <: BlasFloat
    matrix = CudssMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return new{T}(matrix, config, data)
  end
end

"""
    cudss_set(matrix::CudssMatrix{T}, v::CuVector{T})
    cudss_set(matrix::CudssMatrix{T}, A::CuMatrix{T})
    cudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T,Cint})
    cudss_set(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})
    cudss_set(matrix::CudssMatrix{T}, v::Vector{CuVector{T}})
    cudss_set(matrix::CudssMatrix{T}, A::Vector{CuMatrix{T}})
    cudss_set(matrix::CudssMatrix{T}, A::Vector{CuSparseMatrixCSR{T,Cint}})
    cudss_set(solver::CudssSolver{T}, A::Vector{CuSparseMatrixCSR{T,Cint}})
    cudss_set(solver::CudssSolver, parameter::String, value)
    cudss_set(config::CudssConfig, parameter::String, value)
    cudss_set(data::CudssData, parameter::String, value)

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

The available configuration parameters are:
- `"reordering_alg"`: Algorithm for the reordering phase (`"default"`, `"algo1"`, `"algo2"` or `"algo3"`);
- `"factorization_alg"`: Algorithm for the factorization phase (`"default"`, `"algo1"`, `"algo2"` or `"algo3"`);
- `"solve_alg"`: Algorithm for the solving phase (`"default"`, `"algo1"`, `"algo2"` or `"algo3"`);
- `"matching_type"`: Type of matching;
- `"solve_mode"`: Potential modificator on the system matrix (transpose or adjoint);
- `"ir_n_steps"`: Number of steps during the iterative refinement;
- `"ir_tol"`: Iterative refinement tolerance;
- `"pivot_type"`: Type of pivoting (`'C'`, `'R'` or `'N'`);
- `"pivot_threshold"`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `"pivot_epsilon"`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `"max_lu_nnz"`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;
- `"hybrid_mode"`: Memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"hybrid_device_memory_limit"`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `"use_cuda_register_memory"`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode.

The available data parameters are:
- `"user_perm"`: User permutation to be used instead of running the reordering algorithms;
- `"comm"`: Communicator for Multi-GPU multi-node mode.
"""
function cudss_set end

function cudss_set(matrix::CudssMatrix{T}, v::CuVector{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, v)
end

function cudss_set(matrix::CudssMatrix{T}, A::CuMatrix{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, A)
end

function cudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: BlasFloat
  cudssMatrixSetCsrPointers(matrix, A.rowPtr, CU_NULL, A.colVal, A.nzVal)
end

function cudss_set(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: BlasFloat
  cudss_set(solver.matrix, A)
end

function cudss_set(matrix::CudssMatrix{T}, v::Vector{<:CuVector{T}}) where T <: BlasFloat
  vptrs = unsafe_cudss_batch(v)
  cudssMatrixSetBatchValues(matrix, vptrs)
  # unsafe_free!(vptrs)
end

function cudss_set(matrix::CudssMatrix{T}, A::Vector{<:CuMatrix{T}}) where T <: BlasFloat
  Aptrs = unsafe_cudss_batch(A)
  cudssMatrixSetBatchValues(matrix, Aptrs)
  # unsafe_free!(Aptrs)
end

function cudss_set(matrix::CudssMatrix{T}, A::Vector{CuSparseMatrixCSR{T,Cint}}) where T <: BlasFloat
  rowPtrs, colVals, nzVals = unsafe_cudss_batch(A)
  cudssMatrixSetBatchCsrPointers(matrix, rowPtrs, CUPTR_C_NULL, colVals, nzVals)
  # unsafe_free!(rowPtrs)
  # unsafe_free!(colVals)
  # unsafe_free!(nzVals)
end

function cudss_set(solver::CudssSolver{T}, A::Vector{CuSparseMatrixCSR{T,Cint}}) where T <: BlasFloat
  cudss_set(solver.matrix, A)
end

function cudss_set(solver::CudssSolver, parameter::String, value)
  if parameter ∈ CUDSS_CONFIG_PARAMETERS
    cudss_set(solver.config, parameter, value)
  elseif parameter ∈ CUDSS_DATA_PARAMETERS
    cudss_set(solver.data, parameter, value)
  else
    throw(ArgumentError("Unknown data or config parameter $parameter."))
  end
end

function cudss_set(data::CudssData, parameter::String, value)
  (parameter ∈ CUDSS_DATA_PARAMETERS) || throw(ArgumentError("Unknown data parameter $parameter."))
  (parameter == "user_perm") || (parameter == "comm") || throw(ArgumentError("Only the data parameters \"user_perm\" and \"comm\" can be set."))
  (value isa Vector{Cint} || value isa CuVector{Cint}) || throw(ArgumentError("The permutation is neither a Vector{Cint} nor a CuVector{Cint}."))
  nbytes = sizeof(value)
  cudssDataSet(data.handle, data, parameter, value, nbytes)
end

function cudss_set(config::CudssConfig, parameter::String, value)
  (parameter ∈ CUDSS_CONFIG_PARAMETERS) || throw(ArgumentError("Unknown config parameter $parameter."))
  type = CUDSS_TYPES[parameter]
  val = Ref{type}(value)
  nbytes = sizeof(val)
  cudssConfigSet(config, parameter, val, nbytes)
end

"""
    value = cudss_get(solver::CudssSolver, parameter::String)
    value = cudss_get(config::CudssConfig, parameter::String)
    value = cudss_get(data::CudssData, parameter::String)

The available configuration parameters are:
- `"reordering_alg"`: Algorithm for the reordering phase;
- `"factorization_alg"`: Algorithm for the factorization phase;
- `"solve_alg"`: Algorithm for the solving phase;
- `"matching_type"`: Type of matching;
- `"solve_mode"`: Potential modificator on the system matrix (transpose or adjoint);
- `"ir_n_steps"`: Number of steps during the iterative refinement;
- `"ir_tol"`: Iterative refinement tolerance;
- `"pivot_type"`: Type of pivoting;
- `"pivot_threshold"`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `"pivot_epsilon"`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `"max_lu_nnz"`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;
- `"hybrid_mode"`: Memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"hybrid_device_memory_limit"`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `"use_cuda_register_memory"`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode.

The available data parameters are:
- `"info"`: Device-side error information;
- `"lu_nnz"`: Number of non-zero entries in LU factors;
- `"npivots"`: Number of pivots encountered during factorization;
- `"inertia"`: Tuple of positive and negative indices of inertia for symmetric and hermitian non positive-definite matrix types;
- `"perm_reorder_row"`: Reordering permutation for the rows;
- `"perm_reorder_col"`: Reordering permutation for the columns;
- `"perm_row"`: Final row permutation (which includes effects of both reordering and pivoting);
- `"perm_col"`: Final column permutation (which includes effects of both reordering and pivoting);
- `"diag"`: Diagonal of the factorized matrix;
- `"hybrid_device_memory_min"`: Minimal amount of device memory (number of bytes) required in the hybrid memory mode;
- `"memory_estimates"`: Memory estimates (in bytes) for host and device memory required for the chosen memory mode.

The data parameters `"info"`, `"lu_nnz"`, `"perm_reorder_row"`, `"perm_reorder_col"`, `"hybrid_device_memory_min"` and `"memory_estimates"` require the phase `"analyse"` performed by [`cudss`](@ref).
The data parameters `"npivots"`, `"inertia"` and `"diag"` require the phases `"analyse"` and `"factorization"` performed by [`cudss`](@ref).
The data parameters `"perm_row"` and `"perm_col"` are available but not yet functional.
"""
function cudss_get end

function cudss_get(solver::CudssSolver, parameter::String)
  if parameter ∈ CUDSS_CONFIG_PARAMETERS
    cudss_get(solver.config, parameter)
  elseif parameter ∈ CUDSS_DATA_PARAMETERS
    cudss_get(solver.data, parameter)
  else
    throw(ArgumentError("Unknown data or config parameter $parameter."))
  end
end

function cudss_get(data::CudssData, parameter::String)
  (parameter ∈ CUDSS_DATA_PARAMETERS) || throw(ArgumentError("Unknown data parameter $parameter."))
  if (parameter == "user_perm") || (parameter == "comm")
    throw(ArgumentError("The data parameter \"$parameter\" cannot be retrieved."))
  end
  if (parameter == "perm_reorder_row") || (parameter == "perm_reorder_col") || (parameter == "perm_row") || (parameter == "perm_col") || (parameter == "diag") || (parameter == "memory_estimates")
    throw(ArgumentError("The data parameter \"$parameter\" is not supported by CUDSS.jl."))
  end
  type = CUDSS_TYPES[parameter]
  val = Ref{type}()
  nbytes = sizeof(val)
  nbytes_written = Ref{Csize_t}()
  cudssDataGet(handle(), data, parameter, val, nbytes, nbytes_written)
  return val[]
end

function cudss_get(config::CudssConfig, parameter::String)
  (parameter ∈ CUDSS_CONFIG_PARAMETERS) || throw(ArgumentError("Unknown config parameter $parameter."))
  type = CUDSS_TYPES[parameter]
  val = Ref{type}()
  nbytes = sizeof(val)
  nbytes_written = Ref{Csize_t}()
  cudssConfigGet(config, parameter, val, nbytes, nbytes_written)
  return val[]
end

"""
    cudss(phase::String, solver::CudssSolver{T}, x::CuVector{T}, b::CuVector{T})
    cudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T})
    cudss(phase::String, solver::CudssSolver{T}, x::Vector{CuVector{T}}, b::Vector{CuVector{T}})
    cudss(phase::String, solver::CudssSolver{T}, X::Vector{CuMatrix{T}}, B::Vector{CuMatrix{T}})
    cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T})

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

The available phases are `"analysis"`, `"factorization"`, `"refactorization"` and `"solve"`.
The phases `"solve_fwd"`, `"solve_diag"` and `"solve_bwd"` are available but not yet functional.
"""
function cudss end

function cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T}) where T <: BlasFloat
  cudssExecute(solver.data.handle, phase, solver.config, solver.data, solver.matrix, X, B)
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

function cudss(phase::String, solver::CudssSolver{T}, x::Vector{<:CuVector{T}}, b::Vector{<:CuVector{T}}) where T <: BlasFloat
  solution = CudssMatrix(x)
  rhs = CudssMatrix(b)
  cudss(phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssSolver{T}, X::Vector{<:CuMatrix{T}}, B::Vector{<:CuMatrix{T}}) where T <: BlasFloat
  solution = CudssMatrix(X)
  rhs = CudssMatrix(B)
  cudss(phase, solver, solution, rhs)
end

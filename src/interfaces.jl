export CudssSolver, CudssBatchedSolver, cudss, cudss_set, cudss_get

"""
    solver = CudssSolver(A::CuSparseMatrixCSR{T,Cint}, structure::String, view::Char; index::Char='O')
    solver = CudssSolver(matrix::CudssMatrix{T}, config::CudssConfig, data::CudssData)

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

`CudssSolver` contains all structures required to solve a linear system with cuDSS.
One constructor of `CudssSolver` takes as input the same parameters as [`CudssMatrix`](@ref).

`structure` specifies the stucture for the sparse matrix:
- `"G"`: General matrix -- LDU factorization;
- `"S"`: Real symmetric matrix -- LDLᵀ factorization;
- `"H"`: Complex Hermitian matrix -- LDLᴴ factorization;
- `"SPD"`: Symmetric positive-definite matrix -- LLᵀ factorization;
- `"HPD"`: Hermitian positive-definite matrix -- LLᴴ factorization.

`view` specifies matrix view for the sparse matrix:
- `'L'`: Lower-triangular matrix and all values above the main diagonal are ignored;
- `'U'`: Upper-triangular matrix and all values below the main diagonal are ignored;
- `'F'`: Full matrix.

`index` specifies indexing base for the sparse matrix:
- `'Z'`: 0-based indexing;
- `'O'`: 1-based indexing.

`CudssSolver` can be also constructed from the three structures [`CudssMatrix`](@ref), [`CudssConfig`](@ref) and [`CudssData`](@ref) if needed.
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
end

"""
    solver = CudssBatchedSolver(A::CuSparseMatrixCSR{T,Cint}, structure::String, view::Char; index::Char='O')
    solver = CudssBatchedSolver(matrix::CudssBatchedMatrix{T}, config::CudssConfig, data::CudssData)

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

`CudssBatchedSolver` contains all structures required to solve a batch of linear systems with cuDSS.
One constructor of `CudssBatchedSolver` takes as input the same parameters as [`CudssBatchedMatrix`](@ref).

`structure` specifies the stucture for the sparse matrices:
- `"G"`: General matrix -- LDU factorization;
- `"S"`: Real symmetric matrix -- LDLᵀ factorization;
- `"H"`: Complex Hermitian matrix -- LDLᴴ factorization;
- `"SPD"`: Symmetric positive-definite matrix -- LLᵀ factorization;
- `"HPD"`: Hermitian positive-definite matrix -- LLᴴ factorization.

`view` specifies matrix view for the sparse matrices:
- `'L'`: Lower-triangular matrix and all values above the main diagonal are ignored;
- `'U'`: Upper-triangular matrix and all values below the main diagonal are ignored;
- `'F'`: Full matrix.

`index` specifies indexing base for the sparse matrices:
- `'Z'`: 0-based indexing;
- `'O'`: 1-based indexing.

`CudssBatchedSolver` can be also constructed from the three structures [`CudssBatchedMatrix`](@ref), [`CudssConfig`](@ref) and [`CudssData`](@ref) if needed.
"""
mutable struct CudssBatchedSolver{T,M}
  matrix::CudssBatchedMatrix{T,M}
  config::CudssConfig
  data::CudssData

  function CudssBatchedSolver(matrix::CudssBatchedMatrix{T}, config::CudssConfig, data::CudssData) where T <: BlasFloat
    M = typeof(matrix.Mptrs)
    return new{T,M}(matrix, config, data)
  end

  function CudssBatchedSolver(A::Vector{CuSparseMatrixCSR{T,Cint}}, structure::String, view::Char; index::Char='O') where T <: BlasFloat
    matrix = CudssBatchedMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    M = typeof(matrix.Mptrs)
    return new{T,M}(matrix, config, data)
  end
end

"""
    cudss_set(solver::CudssSolver, parameter::String, value)
    cudss_set(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})
    cudss_set(solver::CudssBatchedSolver, parameter::String, value)
    cudss_set(solver::CudssBatchedSolver{T}, A::Vector{CuSparseMatrixCSR{T,Cint}})
    cudss_set(config::CudssConfig, parameter::String, value)
    cudss_set(data::CudssData, parameter::String, value)
    cudss_set(matrix::CudssMatrix{T}, b::CuVector{T})
    cudss_set(matrix::CudssMatrix{T}, B::CuMatrix{T})
    cudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T,Cint})
    cudss_set(matrix::CudssBatchedMatrix{T}, b::Vector{CuVector{T}})
    cudss_set(matrix::CudssBatchedMatrix{T}, B::Vector{CuMatrix{T}})
    cudss_set(matrix::CudssBatchedMatrix{T}, A::Vector{CuSparseMatrixCSR{T,Cint}})

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

The available configuration parameters are:
- `"reordering_alg"`: Algorithm for the reordering phase (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, or `"algo5"`);
- `"factorization_alg"`: Algorithm for the factorization phase (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, or `"algo5"`);
- `"solve_alg"`: Algorithm for the solving phase (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, or `"algo5"`);
- `"use_matching"`: A flag to enable (`1`) or disable (`0`) the matching;
- `"matching_alg"`: Algorithm for the matching;
- `"solve_mode"`: Potential modificator on the system matrix (transpose or adjoint);
- `"ir_n_steps"`: Number of steps during the iterative refinement;
- `"ir_tol"`: Iterative refinement tolerance;
- `"pivot_type"`: Type of pivoting (`'C'`, `'R'` or `'N'`);
- `"pivot_threshold"`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `"pivot_epsilon"`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `"max_lu_nnz"`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;
- `"hybrid_mode"`: Memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"hybrid_device_memory_limit"`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `"use_cuda_register_memory"`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode;
- `"host_nthreads"`: Number of threads to be used by cuDSS in multi-threaded mode;
- `"hybrid_execute_mode"`: Hybrid execute mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"pivot_epsilon_alg"`: Algorithm for the pivot epsilon calculation;
- `"nd_nlevels"`: Minimum number of levels for the nested dissection reordering;
- `"ubatch_size"`: The number of matrices in a uniform batch of systems to be processed by cuDSS;
- `"ubatch_index"`: Specify cuDSS to process all matrices in the uniform batch at once.

The available data parameters are:
- `"info"`: Device-side error information;
- `"user_perm"`: User permutation to be used instead of running the reordering algorithms;
- `"comm"`: Communicator for Multi-GPU multi-node mode.

The data parameter `"info"` must be restored to `0` if a Cholesky factorization fails
due to indefiniteness and refactorization is performed on an updated matrix.
"""
function cudss_set end

function cudss_set(matrix::CudssMatrix{T}, b::CuVector{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, b)
end

function cudss_set(matrix::CudssMatrix{T}, B::CuMatrix{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, B)
end

function cudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: BlasFloat
  cudssMatrixSetCsrPointers(matrix, A.rowPtr, CU_NULL, A.colVal, A.nzVal)
end

function cudss_set(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: BlasFloat
  cudss_set(solver.matrix, A)
end

function cudss_set(matrix::CudssBatchedMatrix{T}, b::Vector{<:CuVector{T}}) where T <: BlasFloat
  Mptrs = unsafe_cudss_batch(b)
  copyto!(matrix.Mptrs, Mptrs)
  cudssMatrixSetBatchValues(matrix, matrix.Mptrs)
  unsafe_free!(Mptrs)
end

function cudss_set(matrix::CudssBatchedMatrix{T}, B::Vector{<:CuMatrix{T}}) where T <: BlasFloat
  Mptrs = unsafe_cudss_batch(B)
  copyto!(matrix.Mptrs, Mptrs)
  cudssMatrixSetBatchValues(matrix, matrix.Mptrs)
  unsafe_free!(Mptrs)
end

function cudss_set(matrix::CudssBatchedMatrix{T}, A::Vector{CuSparseMatrixCSR{T,Cint}}) where T <: BlasFloat
  rowPtrs, colVals, nzVals = unsafe_cudss_batch(A)
  copyto!(matrix.Mptrs[1], rowPtrs)
  copyto!(matrix.Mptrs[2], colVals)
  copyto!(matrix.Mptrs[3], nzVals)
  cudssMatrixSetBatchCsrPointers(matrix, matrix.Mptrs[1], CUPTR_C_NULL, matrix.Mptrs[2], matrix.Mptrs[3])
  unsafe_free!(rowPtrs)
  unsafe_free!(colVals)
  unsafe_free!(nzVals)
end

function cudss_set(solver::CudssBatchedSolver{T}, A::Vector{CuSparseMatrixCSR{T,Cint}}) where T <: BlasFloat
  cudss_set(solver.matrix, A)
end

function cudss_set(solver::Union{CudssSolver,CudssBatchedSolver}, parameter::String, value)
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
  (parameter == "comm") && throw(ArgumentError("The data parameter \"$parameter\" is not supported by CUDSS.jl."))
  if parameter == "info"
    val = Ref{Cint}(value)
    nbytes = sizeof(val)
    cudssDataSet(data.handle, data, parameter, val, nbytes)
  else
    (parameter == "user_perm") || throw(ArgumentError("Only the data parameters \"info\" and \"user_perm\" can be set."))
    (value isa Vector{Cint} || value isa CuVector{Cint}) || throw(ArgumentError("The permutation is neither a Vector{Cint} nor a CuVector{Cint}."))
    nbytes = sizeof(value)
    cudssDataSet(data.handle, data, parameter, value, nbytes)
  end
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
    value = cudss_get(solver::CudssBatchedSolver, parameter::String)
    value = cudss_get(config::CudssConfig, parameter::String)
    value = cudss_get(data::CudssData, parameter::String)

The available configuration parameters are:
- `"reordering_alg"`: Algorithm for the reordering phase;
- `"factorization_alg"`: Algorithm for the factorization phase;
- `"solve_alg"`: Algorithm for the solving phase;
- `"use_matching"`: A flag to enable (`1`) or disable (`0`) the matching;
- `"matching_alg"`: Algorithm for the matching;
- `"solve_mode"`: Potential modificator on the system matrix (transpose or adjoint);
- `"ir_n_steps"`: Number of steps during the iterative refinement;
- `"ir_tol"`: Iterative refinement tolerance;
- `"pivot_type"`: Type of pivoting;
- `"pivot_threshold"`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `"pivot_epsilon"`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `"max_lu_nnz"`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;
- `"hybrid_mode"`: Memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"hybrid_device_memory_limit"`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `"use_cuda_register_memory"`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode;
- `"host_nthreads"`: Number of threads to be used by cuDSS in multi-threaded mode;
- `"hybrid_execute_mode"`: Hybrid execute mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"pivot_epsilon_alg"`: Algorithm for the pivot epsilon calculation;
- `"nd_nlevels"`: Minimum number of levels for the nested dissection reordering;
- `"ubatch_size"`: The number of matrices in a uniform batch of systems to be processed by cuDSS;
- `"ubatch_index"`: Specify cuDSS to process all matrices in the uniform batch at once.

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

function cudss_get(solver::Union{CudssSolver,CudssBatchedSolver}, parameter::String)
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
  if (parameter == "perm_reorder_row") || (parameter == "perm_reorder_col") ||
     (parameter == "perm_row") || (parameter == "perm_col") || (parameter == "diag") ||
     (parameter == "perm_matching") || (parameter == "scale_row") || (parameter == "scale_col")
    throw(ArgumentError("The data parameter \"$parameter\" is not supported by CUDSS.jl."))
  end
  if parameter == "memory_estimates"
    val = zeros(Int64, 16)
  else
    type = CUDSS_TYPES[parameter]
    val = Ref{type}()
  end
  nbytes = sizeof(val)
  nbytes_written = Ref{Csize_t}()
  cudssDataGet(handle(), data, parameter, val, nbytes, nbytes_written)
  parameter_value = (parameter == "memory_estimates") ? val : val[]
  return parameter_value
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
    cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T})
    cudss(phase::String, solver::CudssBatchedSolver{T}, x::Vector{CuVector{T}}, b::Vector{CuVector{T}})
    cudss(phase::String, solver::CudssBatchedSolver{T}, X::Vector{CuMatrix{T}}, B::Vector{CuMatrix{T}})
    cudss(phase::String, solver::CudssBatchedSolver{T}, X::CudssBatchedMatrix{T}, B::CudssBatchedMatrix{T})

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

The available phases are `"reordering"`, `"symbolic_factorization"`, `"analysis"`, `"factorization"`, `"refactorization"` and `"solve"`.
The phases `"solve_fwd"`, `"solve_diag"` and `"solve_bwd"` are available but not yet functional.
"""
function cudss end

function cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  cudss_phase = convert(cudssPhase_t, phase)
  cudssExecute(solver.data.handle, cudss_phase, solver.config, solver.data, solver.matrix, X, B)
end

function cudss(phase::String, solver::CudssSolver{T}, x::CuVector{T}, b::CuVector{T}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  solution = CudssMatrix(x)
  rhs = CudssMatrix(b)
  cudss_phase = convert(cudssPhase_t, phase)
  cudss(cudss_phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  solution = CudssMatrix(X)
  rhs = CudssMatrix(B)
  cudss_phase = convert(cudssPhase_t, phase)
  cudss(cudss_phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssBatchedSolver{T}, X::CudssBatchedMatrix{T}, B::CudssBatchedMatrix{T}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  cudss_phase = convert(cudssPhase_t, phase)
  cudssExecute(solver.data.handle, cudss_phase, solver.config, solver.data, solver.matrix, X, B)
end

function cudss(phase::String, solver::CudssBatchedSolver{T}, x::Vector{<:CuVector{T}}, b::Vector{<:CuVector{T}}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  solution = CudssBatchedMatrix(x)
  rhs = CudssBatchedMatrix(b)
  cudss_phase = convert(cudssPhase_t, phase)
  cudss(cudss_phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssBatchedSolver{T}, X::Vector{<:CuMatrix{T}}, B::Vector{<:CuMatrix{T}}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  solution = CudssBatchedMatrix(X)
  rhs = CudssBatchedMatrix(B)
  cudss_phase = convert(cudssPhase_t, phase)
  cudss(cudss_phase, solver, solution, rhs)
end

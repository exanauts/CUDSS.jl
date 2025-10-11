export CudssSolver, CudssBatchedSolver, cudss, cudss_set, cudss_get, cudss_update

"""
    solver = CudssSolver(A::CuSparseMatrixCSR{T,INT}, structure::String, view::Char; index::Char='O')
    solver = CudssSolver(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}, structure::String, view::Char; index::Char='O')
    solver = CudssSolver(matrix::CudssMatrix{T,INT}, config::CudssConfig, data::CudssData)

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

`CudssSolver` contains all structures required to solve a linear system with cuDSS.
It can also be used to solve a batch of linear systems sharing the same sparsity pattern.
Two constructors of `CudssSolver` take as input the same parameters as [`CudssMatrix`](@ref).

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
mutable struct CudssSolver{T,INT} <: AbstractCudssSolver{T,INT}
  matrix::CudssMatrix{T,INT}
  config::CudssConfig
  data::CudssData
  pointer::PtrOrCuPtr{Cvoid}
  ref_cint::Base.RefValue{Cint}
  ref_int64::Base.RefValue{Int64}
  ref_float64::Base.RefValue{Float64}
  ref_inertia::Base.RefValue{Tuple{INT,INT}}
  ref_schur::Base.RefValue{Tuple{Int64,Int64,Int64}}
  ref_algo::Base.RefValue{cudssAlgType_t}
  ref_pivot::Base.RefValue{cudssPivotType_t}
  ref_matrix::Base.RefValue{cudssMatrix_t}
  nbytes_provided::Csize_t
  nbytes_written::Base.RefValue{Csize_t}

  function CudssSolver(matrix::CudssMatrix{T,INT}, config::CudssConfig, data::CudssData) where {T <: BlasFloat, INT <: CudssInt}
    pointer = Base.unsafe_convert(PtrOrCuPtr{Cvoid}, C_NULL)
    ref_cint = Ref{Cint}()
    ref_int64 = Ref{Int64}()
    ref_float64 = Ref{Float64}()
    ref_inertia = Ref{Tuple{INT,INT}}()
    ref_schur = Ref{Tuple{Int64,Int64,Int64}}()
    ref_algo = Ref{cudssAlgType_t}()
    ref_pivot = Ref{cudssPivotType_t}()
    ref_matrix = Ref{cudssMatrix_t}()
    nbytes_provided = Csize_t(0)
    nbytes_written = Ref{Csize_t}()
    return new{T,INT}(matrix, config, data, pointer, ref_cint, ref_int64, ref_float64, ref_inertia,
                      ref_schur, ref_algo, ref_pivot, ref_matrix, nbytes_provided, nbytes_written)
  end

  function CudssSolver(A::CuSparseMatrixCSR{T,INT}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
    matrix = CudssMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return CudssSolver(matrix, config, data)
  end

  function CudssSolver(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
    matrix = CudssMatrix(rowPtr, colVal, nzVal, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return CudssSolver(matrix, config, data)
  end
end

"""
    solver = CudssBatchedSolver(A::Vector{CuSparseMatrixCSR{T,INT}}, structure::String, view::Char; index::Char='O')
    solver = CudssBatchedSolver(matrix::CudssBatchedMatrix{T,INT}, config::CudssConfig, data::CudssData)

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

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
mutable struct CudssBatchedSolver{T,INT,M} <: AbstractCudssSolver{T,INT}
  matrix::CudssBatchedMatrix{T,INT,M}
  config::CudssConfig
  data::CudssData
  pointer::PtrOrCuPtr{Cvoid}
  ref_cint::Base.RefValue{Cint}
  ref_int64::Base.RefValue{Int64}
  ref_float64::Base.RefValue{Float64}
  ref_inertia::Base.RefValue{Tuple{INT,INT}}
  ref_schur::Base.RefValue{Tuple{Int64,Int64,Int64}}
  ref_algo::Base.RefValue{cudssAlgType_t}
  ref_pivot::Base.RefValue{cudssPivotType_t}
  ref_matrix::Base.RefValue{cudssMatrix_t}
  nbytes_provided::Csize_t
  nbytes_written::Base.RefValue{Csize_t}

  function CudssBatchedSolver(matrix::CudssBatchedMatrix{T,INT}, config::CudssConfig, data::CudssData) where {T <: BlasFloat, INT <: CudssInt}
    pointer = Base.unsafe_convert(PtrOrCuPtr{Cvoid}, C_NULL)
    ref_cint = Ref{Cint}()
    ref_int64 = Ref{Int64}()
    ref_float64 = Ref{Float64}()
    ref_inertia = Ref{Tuple{INT,INT}}()
    ref_schur = Ref{Tuple{Int64,Int64,Int64}}()
    ref_algo = Ref{cudssAlgType_t}()
    ref_pivot = Ref{cudssPivotType_t}()
    ref_matrix = Ref{cudssMatrix_t}()
    nbytes_provided = Csize_t(0)
    nbytes_written = Ref{Csize_t}()
    M = typeof(matrix.Mptrs)
    return new{T,INT,M}(matrix, config, data, pointer, ref_cint, ref_int64, ref_float64, ref_inertia,
                        ref_schur, ref_algo, ref_pivot, ref_matrix, nbytes_provided, nbytes_written)
  end

  function CudssBatchedSolver(A::Vector{CuSparseMatrixCSR{T,INT}}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
    matrix = CudssBatchedMatrix(A, structure, view; index)
    config = CudssConfig()
    data = CudssData()
    return CudssBatchedSolver(matrix, config, data)
  end
end

"""
    cudss_update(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT})
    cudss_update(solver::CudssSolver{T,INT}, rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T})
    cudss_update(solver::CudssBatchedSolver{T,INT}, A::Vector{CuSparseMatrixCSR{T,INT}})
    cudss_update(matrix::CudssMatrix{T}, b::CuVector{T})
    cudss_update(matrix::CudssMatrix{T}, B::CuMatrix{T})
    cudss_update(matrix::CudssMatrix{T,INT}, A::CuSparseMatrixCSR{T,INT})
    cudss_update(matrix::CudssMatrix{T,INT}, rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T})
    cudss_update(matrix::CudssBatchedMatrix{T}, b::Vector{CuVector{T}})
    cudss_update(matrix::CudssBatchedMatrix{T}, B::Vector{CuMatrix{T}})
    cudss_update(matrix::CudssBatchedMatrix{T,INT}, A::Vector{CuSparseMatrixCSR{T,INT}})

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

Update the contents of a `CudssMatrix` -- `CudssBatchedMatrix` or `CudssSolver` -- `CudssBatchedSolver` with new numerical values.
"""
function cudss_update end

function cudss_update(matrix::CudssMatrix{T}, b::CuVector{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, b)
end

function cudss_update(matrix::CudssMatrix{T}, B::CuMatrix{T}) where T <: BlasFloat
  cudssMatrixSetValues(matrix, B)
end

function cudss_update(matrix::CudssMatrix{T,INT}, A::CuSparseMatrixCSR{T,INT}) where {T <: BlasFloat, INT <: CudssInt}
  cudssMatrixSetCsrPointers(matrix, A.rowPtr, CU_NULL, A.colVal, A.nzVal)
end

function cudss_update(matrix::CudssMatrix{T,INT}, rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}) where {T <: BlasFloat, INT <: CudssInt}
  cudssMatrixSetCsrPointers(matrix, rowPtr, CU_NULL, colVal, nzVal)
end

function cudss_update(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}) where {T <: BlasFloat, INT <: CudssInt}
  cudss_update(solver.matrix, A)
end

function cudss_update(solver::CudssSolver{T,INT}, rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}) where {T <: BlasFloat, INT <: CudssInt}
  cudss_update(solver.matrix, rowPtr, colVal, nzVal)
end

function cudss_update(matrix::CudssBatchedMatrix{T}, b::Vector{<:CuVector{T}}) where T <: BlasFloat
  Mptrs = unsafe_cudss_batch(b)
  copyto!(matrix.Mptrs, Mptrs)
  cudssMatrixSetBatchValues(matrix, matrix.Mptrs)
  unsafe_free!(Mptrs)
end

function cudss_update(matrix::CudssBatchedMatrix{T}, B::Vector{<:CuMatrix{T}}) where T <: BlasFloat
  Mptrs = unsafe_cudss_batch(B)
  copyto!(matrix.Mptrs, Mptrs)
  cudssMatrixSetBatchValues(matrix, matrix.Mptrs)
  unsafe_free!(Mptrs)
end

function cudss_update(matrix::CudssBatchedMatrix{T,INT}, A::Vector{CuSparseMatrixCSR{T,INT}}) where {T <: BlasFloat, INT <: CudssInt}
  rowPtrs, colVals, nzVals = unsafe_cudss_batch(A)
  copyto!(matrix.Mptrs[1], rowPtrs)
  copyto!(matrix.Mptrs[2], colVals)
  copyto!(matrix.Mptrs[3], nzVals)
  cudssMatrixSetBatchCsrPointers(matrix, matrix.Mptrs[1], CUPTR_C_NULL, matrix.Mptrs[2], matrix.Mptrs[3])
  unsafe_free!(rowPtrs)
  unsafe_free!(colVals)
  unsafe_free!(nzVals)
end

function cudss_update(solver::CudssBatchedSolver{T,INT}, A::Vector{CuSparseMatrixCSR{T,INT}}) where {T <: BlasFloat, INT <: CudssInt}
  cudss_update(solver.matrix, A)
end

"""
    cudss_set(solver::CudssSolver, parameter::String, value)
    cudss_set(solver::CudssBatchedSolver, parameter::String, value)

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
- `"hybrid_memory_mode"`: Hybrid memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"hybrid_device_memory_limit"`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `"use_cuda_register_memory"`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode;
- `"host_nthreads"`: Number of threads to be used by cuDSS in multi-threaded mode;
- `"hybrid_execute_mode"`: Hybrid execute mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"pivot_epsilon_alg"`: Algorithm for the pivot epsilon calculation;
- `"nd_nlevels"`: Minimum number of levels for the nested dissection reordering;
- `"ubatch_size"`: The number of matrices in a uniform batch of systems to be processed by cuDSS;
- `"ubatch_index"`: Use `-1` (default) to process all matrices in the uniform batch, or a 0-based index to process a single matrix during the factorization or solve phase;
- `"use_superpanels"`: Use superpanel optimization -- `1` (default = enabled) or `0` (disabled);
- `"device_count"`: Device count in case of multiple device;
- `"device_indices"`: A list of device indices as an integer array;
- `"schur_mode"`: Schur complement mode -- `0` (default = disabled) or `1` (enabled);
- `"deterministic_mode"`: Enable deterministic mode -- `0` (default = disabled) or `1` (enabled).

The available data parameters are:
- `"info"`: Device-side error information;
- `"user_perm"`: User permutation to be used instead of running the reordering algorithms;
- `"comm"`: Communicator for Multi-GPU multi-node mode;
- `"user_elimination_tree"`: User provided elimination tree information, which is used instead of running the reordering algorithm;
- `"user_schur_indices"`: User-provided Schur complement indices. The provided buffer should be an integer array of size `n`, where `n` is the dimension of the matrix. The values should be equal to `1` for the rows / columns which are part of the Schur complement and `0` for the rest;
- `"user_host_interrupt"`: User-provided host interrupt pointer;
- `"schur_matrix"`: Schur complement matrix passed as a `cudssMatrix_t` object.

The data parameter `"info"` must be restored to `0` if a Cholesky factorization fails
due to indefiniteness and refactorization is performed on an updated matrix.

Note that for the data parameters `"perm_reorder_row"`, `"perm_row"`, `"scale_row"`, `"perm_reorder_col"`, `"perm_col"`, `"scale_col"`,
`"perm_matching"`, `"diag"`, and `"memory_estimates"`, this function only specifies which vector to update for a subsequent call to [`cudss_get`](@ref).
"""
function cudss_set end

function cudss_set(solver::AbstractCudssSolver, parameter::String, value)
  if parameter ∈ CUDSS_CONFIG_PARAMETERS
    cudss_set_config(solver, parameter, value)
  elseif parameter ∈ CUDSS_DATA_PARAMETERS
    cudss_set_data(solver, parameter, value)
  else
    throw(ArgumentError("Unknown data or config parameter \"$parameter\"."))
  end
  return
end

function cudss_set_data(solver::AbstractCudssSolver{T,INT}, parameter::String, value) where {T <: BlasFloat, INT <: CudssInt}
  if parameter == "info"
    solver.ref_cint[] = value
    cudssDataSet(solver.data.handle, solver.data, parameter, solver.ref_cint, 4)
  elseif parameter == "user_perm" || parameter == "user_schur_indices"
    (value isa Vector{INT} || value isa CuVector{INT}) || throw(ArgumentError("The vector is neither a Vector{$INT} nor a CuVector{$INT}."))
    cudssDataSet(solver.data.handle, solver.data, parameter, value, sizeof(value))
  elseif parameter == "schur_matrix"
    solver.ref_matrix[] = value
  elseif parameter == "perm_reorder_row" || parameter == "perm_row" || parameter == "scale_row" ||
         parameter == "perm_reorder_col" || parameter == "perm_col" || parameter == "scale_col" ||
         parameter == "perm_matching" || parameter == "diag" || parameter == "memory_estimates"
    solver.pointer = Base.unsafe_convert(PtrOrCuPtr{Cvoid}, value)
    solver.nbytes_provided = sizeof(value)
  elseif parameter == "comm" || parameter == "user_elimination_tree" || parameter == "user_host_interrupt"
    throw(ArgumentError("The data parameter \"$parameter\" is not yet supported by CUDSS.jl."))
  elseif parameter == "lu_nnz" || parameter == "npivots" || parameter == "inertia" || parameter == "hybrid_device_memory_min" ||
         parameter == "nsuperpanels" || parameter == "schur_shape" || parameter == "elimination_tree"
    throw(ArgumentError("The data parameter \"$parameter\" can't be set."))
  else
    throw(ArgumentError("Unknown data parameter \"$parameter\"."))
  end
  return
end

function cudss_set_config(solver::AbstractCudssSolver, parameter::String, value)
  if parameter == "reordering_alg" || parameter == "factorization_alg" || parameter == "solve_alg" ||
     parameter == "matching_alg" || parameter == "pivot_epsilon_alg"
    solver.ref_algo[] = value
    cudssConfigSet(solver.config, parameter, solver.ref_algo, 4)
  elseif parameter == "pivot_type"
    solver.ref_pivot[] = value
    cudssConfigSet(solver.config, parameter, solver.ref_pivot, 4)
  elseif parameter == "ir_tol" || parameter == "pivot_threshold" || parameter == "pivot_epsilon"
    solver.ref_float64[] = value
    cudssConfigSet(solver.config, parameter, solver.ref_float64, 8)
  elseif parameter == "max_lu_nnz" || parameter == "hybrid_device_memory_limit"
    solver.ref_int64[] = value
    cudssConfigSet(solver.config, parameter, solver.ref_int64, 8)
  elseif parameter == "hybrid_memory_mode" || parameter == "hybrid_execute_mode" || parameter == "solve_mode" ||
         parameter == "deterministic_mode" || parameter == "schur_mode" || parameter == "use_cuda_register_memory" ||
         parameter == "use_matching" || parameter == "use_superpanels" || parameter == "ir_n_steps" ||
         parameter == "host_nthreads" || parameter == "device_count" || parameter == "nd_nlevels" ||
         parameter == "ubatch_size" || parameter == "ubatch_index"
    solver.ref_cint[] = value
    cudssConfigSet(solver.config, parameter, solver.ref_cint, 4)
  elseif parameter == "device_indices"
    throw(ArgumentError("The config parameter \"device_indices\" is not supported by CUDSS.jl."))
  else
    throw(ArgumentError("Unknown config parameter \"$parameter\"."))
  end
  return
end

"""
    value = cudss_get(solver::CudssSolver, parameter::String)
    value = cudss_get(solver::CudssBatchedSolver, parameter::String)

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
- `"hybrid_memory_mode"`: Hybrid memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"hybrid_device_memory_limit"`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `"use_cuda_register_memory"`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode;
- `"host_nthreads"`: Number of threads to be used by cuDSS in multi-threaded mode;
- `"hybrid_execute_mode"`: Hybrid execute mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `"pivot_epsilon_alg"`: Algorithm for the pivot epsilon calculation;
- `"nd_nlevels"`: Minimum number of levels for the nested dissection reordering;
- `"ubatch_size"`: The number of matrices in a uniform batch of systems to be processed by cuDSS;
- `"ubatch_index"`: Use `-1` (default) to process all matrices in the uniform batch, or a 0-based index to process a single matrix during the factorization or solve phase;
- `"use_superpanels"`: Use superpanel optimization -- `1` (default = enabled) or `0` (disabled);
- `"device_count"`: Device count in case of multiple device;
- `"device_indices"`: A list of device indices as an integer array;
- `"schur_mode"`: Schur complement mode -- `0` (default = disabled) or `1` (enabled);
- `"deterministic_mode"`: Enable deterministic mode -- `0` (default = disabled) or `1` (enabled).

The available data parameters are:
- `"info"`: Device-side error information;
- `"lu_nnz"`: Number of non-zero entries in LU factors;
- `"npivots"`: Number of pivots encountered during factorization;
- `"inertia"`: Tuple of positive and negative indices of inertia for symmetric / hermitian indefinite matrices;
- `"perm_reorder_row"`: Reordering permutation for the rows;
- `"perm_reorder_col"`: Reordering permutation for the columns;
- `"perm_row"`: Final row permutation (which includes effects of both reordering and pivoting);
- `"perm_col"`: Final column permutation (which includes effects of both reordering and pivoting);
- `"perm_matching"`: Matching (column) permutation Q such that A[:,Q] is reordered and then factorized;
- `"scale_row"`: A vector of scaling factors applied to the rows of the factorized matrix;
- `"scale_col"`: A vector of scaling factors applied to the columns of the factorized matrix;
- `"diag"`: Diagonal of the factorized matrix;
- `"hybrid_device_memory_min"`: Minimal amount of device memory (number of bytes) required in the hybrid memory mode;
- `"memory_estimates"`: Memory estimates (in bytes) for host and device memory required for the chosen memory mode;
- `"nsuperpanels"`: Number of superpanels in the matrix;
- `"schur_shape"`: Shape of the Schur complement matrix as a triplet (nrows, ncols, nnz);
- `"schur_matrix"`: Retrieve the Schur complement matrix;
- `"elimination_tree"`: User provided elimination tree information, which is used instead of running the reordering algorithm. It must be used in combination with `"user_perm"` to have an effect.

The data parameters `"info"`, `"lu_nnz"`, `"perm_reorder_row"`, `"perm_reorder_col"`, `"perm_matching"`, `"scale_row"`, `"scale_col"`, `"hybrid_device_memory_min"` and `"memory_estimates"` require the phase `"analyse"` performed by [`cudss`](@ref).
The data parameters `"npivots"`, `"inertia"` and `"diag"` require the phases `"analyse"` and `"factorization"` performed by [`cudss`](@ref).
The data parameters `"perm_matching"`, `"scale_row"`, and `"scale_col"` require matching to be enabled (the configuration parameter `"use_matching"` must be set to `1`).

Note that for the data parameters `"perm_reorder_row"`, `"perm_row"`, `"scale_row"`, `"perm_reorder_col"`, `"perm_col"`, `"scale_col"`,
`"perm_matching"`, `"diag"`, and `"memory_estimates"`, a call to [`cudss_set`](@ref) is required beforehand to specify which vector to update.
"""
function cudss_get end

function cudss_get(solver::AbstractCudssSolver, parameter::String)
  if parameter ∈ CUDSS_CONFIG_PARAMETERS
    cudss_get_config(solver, parameter)
  elseif parameter ∈ CUDSS_DATA_PARAMETERS
    cudss_get_data(solver, parameter)
  else
    throw(ArgumentError("Unknown data or config parameter \"$parameter\"."))
  end
end

function cudss_get_data(solver::AbstractCudssSolver{T,INT}, parameter::String) where {T <: BlasFloat, INT <: CudssInt}
  if parameter == "info"
    cudssDataGet(solver.data.handle, solver.data, parameter, solver.ref_cint, 4, solver.nbytes_written)
    return solver.ref_cint[]
  elseif parameter == "nsuperpanels" || parameter == "npivots"
    ref_INT = (INT == Cint) ? solver.ref_cint : solver.ref_int64
    cudssDataGet(solver.data.handle, solver.data, parameter, ref_INT, sizeof(INT), solver.nbytes_written)
    return ref_INT[]
  elseif parameter == "lu_nnz" || parameter == "hybrid_device_memory_min"
    cudssDataGet(solver.data.handle, solver.data, parameter, solver.ref_int64, 8, solver.nbytes_written)
    return solver.ref_int64[]
  elseif parameter == "inertia"
    cudssDataGet(solver.data.handle, solver.data, parameter, solver.ref_inertia, 2 * sizeof(INT), solver.nbytes_written)
    return solver.ref_inertia[]
  elseif parameter == "schur_shape"
    cudssDataGet(solver.data.handle, solver.data, parameter, solver.ref_schur, 24, solver.nbytes_written)
    return solver.ref_schur[]
  elseif parameter == "schur_matrix"
    cudssDataGet(solver.data.handle, solver.data, parameter, solver.ref_matrix, 8, solver.nbytes_written)
    return nothing
  elseif parameter == "perm_reorder_row" || parameter == "perm_row" || parameter == "scale_row" ||
         parameter == "perm_reorder_col" || parameter == "perm_col" || parameter == "scale_col" ||
         parameter == "perm_matching" || parameter == "diag" || parameter == "memory_estimates"
    cudssDataGet(solver.data.handle, solver.data, parameter, solver.pointer, solver.nbytes_provided, solver.nbytes_written)
    return nothing
  elseif parameter == "user_perm" || parameter == "user_elimination_tree" || parameter == "user_host_interrupt" ||
         parameter == "user_schur_indices" || parameter == "comm" || parameter == "elimination_tree"
    throw(ArgumentError("The data parameter \"$parameter\" can't be retrieved."))
  else
    throw(ArgumentError("Unknown data parameter \"$parameter\"."))
  end
end

function cudss_get_config(solver::AbstractCudssSolver, parameter::String)
  if parameter == "reordering_alg" || parameter == "factorization_alg" || parameter == "solve_alg" ||
     parameter == "matching_alg" || parameter == "pivot_epsilon_alg"
    cudssConfigGet(solver.config, parameter, solver.ref_algo, 4, solver.nbytes_written)
    return solver.ref_algo[]
  elseif parameter == "pivot_type"
    cudssConfigGet(solver.config, parameter, solver.ref_pivot, 4, solver.nbytes_written)
    return solver.ref_pivot[]
  elseif parameter == "ir_tol" || parameter == "pivot_threshold" || parameter == "pivot_epsilon"
    cudssConfigGet(solver.config, parameter, solver.ref_float64, 8, solver.nbytes_written)
    return solver.ref_float64[]
  elseif parameter == "max_lu_nnz" || parameter == "hybrid_device_memory_limit"
    cudssConfigGet(solver.config, parameter, solver.ref_int64, 8, solver.nbytes_written)
    return solver.ref_int64[]
  elseif parameter == "hybrid_memory_mode" || parameter == "hybrid_execute_mode" || parameter == "solve_mode" ||
         parameter == "deterministic_mode" || parameter == "schur_mode" || parameter == "use_cuda_register_memory" ||
         parameter == "use_matching" || parameter == "use_superpanels" || parameter == "ir_n_steps" ||
         parameter == "host_nthreads" || parameter == "device_count" || parameter == "nd_nlevels" ||
         parameter == "ubatch_size" || parameter == "ubatch_index"
    cudssConfigGet(solver.config, parameter, solver.ref_cint, 4, solver.nbytes_written)
    return solver.ref_cint[]
  elseif parameter == "device_indices"
    throw(ArgumentError("The config parameter \"device_indices\" is not supported by CUDSS.jl."))
  else
    throw(ArgumentError("Unknown config parameter \"$parameter\"."))
  end
end

"""
    cudss(phase::String, solver::CudssSolver{T}, x::CuVector{T}, b::CuVector{T})
    cudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T})
    cudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T})
    cudss(phase::String, solver::CudssBatchedSolver{T}, x::Vector{CuVector{T}}, b::Vector{CuVector{T}})
    cudss(phase::String, solver::CudssBatchedSolver{T}, X::Vector{CuMatrix{T}}, B::Vector{CuMatrix{T}})
    cudss(phase::String, solver::CudssBatchedSolver{T}, X::CudssBatchedMatrix{T}, B::CudssBatchedMatrix{T})

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`.

The available phases are:
- `"reordering"`: Reordering;
- `"symbolic_factorization"`: Symbolic factorization;
- `"analysis"`: Reordering and symbolic factorization combined;
- `"factorization"`: Numerical factorization;
- `"refactorization"`: Numerical re-factorization;
- `"solve_fwd_perm"`: Applying reordering permutation to the right hand side before the forward substitution;
- `"solve_fwd"`: Forward substitution sub-step of the solving phase, including the local permutation due to partial pivoting;
- `"solve_diag"`: Diagonal solve sub-step of the solving phase (only needed for symmetric / hermitian indefinite matrices);
- `"solve_bwd"`: Backward substitution sub-step of the solving phase, including the local permutation due to partial pivoting;
- `"solve_bwd_perm"`: Applying inverse reordering permutation to the intermediate solution after the backward substitution. If matching (and scaling) is enabled, this phase also includes applying the inverse matching permutation and inverse scaling (as the matching permutation and scalings were used to modify the matrix before the factorization);
- `"solve_refinement"`: Iterative refinement;
- `"solve"`: Full solving phase, combining all sub-phases and (optional) iterative refinement.

When the Schur complement mode is enabled (option `"schur_mode"` set to `1`), a specific combination of phases is required.
For that reason, we added shorthand phases:
- `"solve_fwd_schur"`: combines the phases `"solve_fwd_perm"`, `"solve_fwd"`, and `"solve_diag"`;
- `"solve_bwd_schur"`: combines the phases `"solve_bwd"` and `"solve_bwd_perm"`.
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
  cudss(phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  solution = CudssMatrix(X)
  rhs = CudssMatrix(B)
  cudss(phase, solver, solution, rhs)
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
  cudss(phase, solver, solution, rhs)
end

function cudss(phase::String, solver::CudssBatchedSolver{T}, X::Vector{<:CuMatrix{T}}, B::Vector{<:CuMatrix{T}}) where T <: BlasFloat
  (phase == "refactorization") && cudss_set(solver, "info", 0)
  solution = CudssBatchedMatrix(X)
  rhs = CudssBatchedMatrix(B)
  cudss(phase, solver, solution, rhs)
end

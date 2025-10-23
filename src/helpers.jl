# cuDSS helper functions
export CudssMatrix, CudssBatchedMatrix, CudssData, CudssConfig

## Matrix

"""
    matrix = CudssMatrix(::Type{T}, n::Integer; nbatch::Integer=1)
    matrix = CudssMatrix(::Type{T}, m::Integer, n::Integer; nbatch::Integer=1)
    matrix = CudssMatrix(b::CuVector{T})
    matrix = CudssMatrix(B::CuMatrix{T})
    matrix = CudssMatrix(A::CuSparseMatrixCSR{T,INT}, struture::String, view::Char; index::Char='O')
    matrix = CudssMatrix(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}, struture::String, view::Char; index::Char='O')

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

`CudssMatrix` is a wrapper for `CuVector`, `CuMatrix` and `CuSparseMatrixCSR`.
`CudssMatrix` can also represent a batch of `CuSparseMatrixCSR` sharing the same sparsity pattern.
`CudssMatrix` is used to pass the matrix of the sparse linear system, as well as solution and right-hand side.

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
"""
mutable struct CudssMatrix{T,INT} <: AbstractCudssMatrix{T,INT}
    value_type::Type{T}
    index_type::Type{INT}
    matrix::cudssMatrix_t
    nrows::Int64
    ncols::Int64
    nz::Int64

    function CudssMatrix(::Type{T}, n::Integer; nbatch::Integer=1) where T <: BlasFloat
        nz = n * nbatch
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, n, 1, n, CU_NULL, T, 'C')
        obj = new{T,Cint}(T, Cint, matrix_ref[], n, 1, nz)
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(::Type{T}, m::Integer, n::Integer; nbatch::Integer=1, transposed::Bool=false) where T <: BlasFloat
        nz = n * m * nbatch
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, CU_NULL, T, 'R')
            obj = new{T,Cint}(T, Cint, matrix_ref[], n, m, nz)
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, CU_NULL, T, 'C')
            obj = new{T,Cint}(T, Cint, matrix_ref[], m, n, nz)
        end
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(b::CuVector{T}) where T <: BlasFloat
        m = length(b)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, m, 1, m, b, T, 'C')
        obj = new{T,Cint}(T, Cint, matrix_ref[], m, 1, m)
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(B::CuMatrix{T}; transposed::Bool=false) where T <: BlasFloat
        m,n = size(B)
        nz = m * n
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, B, T, 'R')
            obj = new{T,Cint}(T, Cint, matrix_ref[], n, m, nz)
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, B, T, 'C')
            obj = new{T,Cint}(T, Cint, matrix_ref[], m, n, nz)
        end
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(A::CuSparseMatrixCSR{T,INT}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        m,n = size(A)
        nz = nnz(A)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateCsr(matrix_ref, m, n, nz, A.rowPtr, CU_NULL,
                             A.colVal, A.nzVal, INT, T, structure,
                             view, index)
        obj = new{T,INT}(T, INT, matrix_ref[], m, n, nz)
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        n = length(rowPtr) - 1
        nz = length(nzVal)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateCsr(matrix_ref, n, n, length(colVal), rowPtr, CU_NULL,
                             colVal, nzVal, INT, T, structure,
                             view, index)
        obj = new{T,INT}(T, INT, matrix_ref[], n, n, nz)
        finalizer(cudssMatrixDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssMatrix_t}, matrix::CudssMatrix) = matrix.matrix

"""
    matrix = CudssBatchedMatrix(b::Vector{CuVector{T}})
    matrix = CudssBatchedMatrix(B::Vector{CuMatrix{T}})
    matrix = CudssBatchedMatrix(A::Vector{CuSparseMatrixCSR{T,INT}}, struture::String, view::Char; index::Char='O')

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

`CudssBatchedMatrix` is a wrapper for `Vector{CuVector}`, `Vector{CuMatrix}` and `Vector{CuSparseMatrixCSR}`.
`CudssBatchedMatrix` is used to pass the matrices of the sparse linear systems, as well as solutions and right-hand sides.

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
"""
mutable struct CudssBatchedMatrix{T,INT,M} <: AbstractCudssMatrix{T,INT}
    value_type::Type{T}
    index_type::Type{INT}
    matrix::cudssMatrix_t
    nbatch::Int64
    nrows::Vector{INT}
    ncols::Vector{INT}
    nnzA::Vector{INT}
    Mptrs::M

    function CudssBatchedMatrix(b::Vector{<:CuVector{T}}) where T <: BlasFloat
        matrix_ref = Ref{cudssMatrix_t}()
        nbatch = length(b)
        nrows = Cint[length(bᵢ) for bᵢ in b]
        ncols = Cint[1 for i = 1:nbatch]
        ld = nrows
        Mptrs = unsafe_cudss_batch(b)
        M = typeof(Mptrs)
        cudssMatrixCreateBatchDn(matrix_ref, nbatch, nrows, ncols, ld, Mptrs, Cint, T, 'C')
        obj = new{T,Cint,M}(T, Cint, matrix_ref[], nbatch, nrows, ncols, Cint[], Mptrs)
        finalizer(cudssBatchedMatrixDestroy, obj)
        obj
    end

    function CudssBatchedMatrix(B::Vector{<:CuMatrix{T}}; transposed::Bool=false) where T <: BlasFloat
        matrix_ref = Ref{cudssMatrix_t}()
        nbatch = length(B)
        nrows = Cint[size(Bᵢ,1) for Bᵢ in B]
        ncols = Cint[size(Bᵢ,2) for Bᵢ in B]
        ld = nrows
        Mptrs = unsafe_cudss_batch(B)
        M = typeof(Mptrs)
        if transposed
            cudssMatrixCreateBatchDn(matrix_ref, nbatch, ncols, nrows, ld, Mptrs, Cint, T, 'R')
        else
            cudssMatrixCreateBatchDn(matrix_ref, nbatch, nrows, ncols, ld, Mptrs, Cint, T, 'C')
        end
        obj = new{T,Cint,M}(T, Cint, matrix_ref[], nbatch, nrows, ncols, Cint[], Mptrs)
        finalizer(cudssBatchedMatrixDestroy, obj)
        obj
    end

    function CudssBatchedMatrix(A::Vector{CuSparseMatrixCSR{T,INT}}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        matrix_ref = Ref{cudssMatrix_t}()
        nbatch = length(A)
        nrows = INT[size(Aᵢ,1) for Aᵢ in A]
        ncols = INT[size(Aᵢ,2) for Aᵢ in A]
        nnzA = INT[nnz(Aᵢ) for Aᵢ in A]
        rowPtrs, colVals, nzVals = unsafe_cudss_batch(A)
        cudssMatrixCreateBatchCsr(matrix_ref, nbatch, nrows, ncols, nnzA, rowPtrs,
                                  CUPTR_C_NULL, colVals, nzVals, INT, T, structure,
                                  view, index)
        Mptrs = (rowPtrs, colVals, nzVals)
        M = typeof(Mptrs)
        obj = new{T,INT,M}(T, INT, matrix_ref[], nbatch, nrows, ncols, nnzA, Mptrs)
        finalizer(cudssBatchedMatrixDestroy, obj)
        obj
    end
end

function cudssBatchedMatrixDestroy(matrix::CudssBatchedMatrix)
    cudssMatrixDestroy(matrix.matrix)
    if matrix.Mptrs isa Tuple
        # sparse matrix
        unsafe_free!(matrix.Mptrs[1])  # rowPtrs
        unsafe_free!(matrix.Mptrs[2])  # colVals
        unsafe_free!(matrix.Mptrs[3])  # nzVals
    else
        # dense vector or matrix
        unsafe_free!(matrix.Mptrs)
    end
end

Base.unsafe_convert(::Type{cudssMatrix_t}, matrix::CudssBatchedMatrix) = matrix.matrix

## Data

"""
    data = CudssData()
    data = CudssData(device_indices::Vector{Cint})
    data = CudssData(cudss_handle::cudssHandle_t)

`CudssData` holds internal data (e.g., LU factors arrays).

When called without arguments, the data is initialized for the current CUDA device.
To prepare data for multiple devices, provide their indices in `device_indices` (starting from 0).
Note: When using `device_indices`, you should also call `CUDSS.devices!(device_indices)` to configure
the task-local device selection before creating the CudssData object.
Alternatively, you can create a `CudssData` object directly from an existing `cudssHandle_t`
returned by `CUDSS.handle()` or `CUDSS.mg_handle()`.
"""
mutable struct CudssData
    handle::cudssHandle_t
    data::cudssData_t

    function CudssData(cudss_handle::cudssHandle_t)
        data_ref = Ref{cudssData_t}()
        cudssDataCreate(cudss_handle, data_ref)
        obj = new(cudss_handle, data_ref[])
        finalizer(cudssDataDestroy, obj)
        obj
    end

    function CudssData()
        cudss_handle = handle()
        CudssData(cudss_handle)
    end

    function CudssData(device_indices::Vector{Cint})
        # Set the devices for the current task
        devices!(device_indices)
        cudss_handle = mg_handle()
        CudssData(cudss_handle)
    end
end

Base.unsafe_convert(::Type{cudssData_t}, data::CudssData) = data.data

function cudssDataDestroy(data::CudssData)
    # Protect against context destruction during finalization
    # This is especially important for multi-GPU handles
    try
        cudssDataDestroy(data.handle, data.data)
    catch e
        # Silently ignore errors during finalization if context is destroyed
        # This can happen during Julia shutdown with multi-GPU handles
        if !(e isa CUDSSError || e isa CUDA.CuError)
            rethrow(e)
        end
    end
end

## Configuration

"""
    config = CudssConfig()
    config = CudssConfig(device_indices::Vector{Cint})

`CudssConfig` stores configuration parameters for the solver.

When called without arguments, the configuration is initialized for the current CUDA device.
To configure the solver for multiple devices, provide their indices in `device_indices` (starting from 0).
This can also be done directly with [`cudss_set`](@ref) by specifying both `"device_count"` and `"device_indices"`.
"""
mutable struct CudssConfig
    config::cudssConfig_t

    function CudssConfig()
        config_ref = Ref{cudssConfig_t}()
        cudssConfigCreate(config_ref)
        obj = new(config_ref[])
        finalizer(_safe_cudssConfigDestroy, obj)
        obj
    end

    function CudssConfig(device_indices::Vector{Cint})
        config = CudssConfig()
        device_count = length(device_indices)
        cudssConfigSet(config, "device_count", Ref{Cint}(device_count), 4)
        cudssConfigSet(config, "device_indices", device_indices, 4 * device_count)
        return config
    end
end

Base.unsafe_convert(::Type{cudssConfig_t}, config::CudssConfig) = config.config

function _safe_cudssConfigDestroy(config::CudssConfig)
    # Protect against context destruction during finalization
    try
        cudssConfigDestroy(config.config)
    catch e
        # Silently ignore errors during finalization if context is destroyed
        if !(e isa CUDSSError || e isa CUDA.CuError)
            rethrow(e)
        end
    end
end

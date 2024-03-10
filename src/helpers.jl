# cuDSS helper functions
export CudssMatrix, CudssData, CudssConfig

## Matrix

"""
    matrix = CudssMatrix(v::CuVector{T})
    matrix = CudssMatrix(A::CuMatrix{T})
    matrix = CudssMatrix(A::CuSparseMatrixCSR{T,Cint}, struture::String, view::Char; index::Char='O')

The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

`CudssMatrix` is a wrapper for `CuVector`, `CuMatrix` and `CuSparseMatrixCSR`.
`CudssMatrix` is used to pass matrix of the linear system, as well as solution and right-hand side.

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
"""
mutable struct CudssMatrix{T}
    type::Type{T}
    matrix::cudssMatrix_t

    function CudssMatrix(::Type{T}, n::Integer) where T <: BlasFloat
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, n, 1, n, CU_NULL, T, 'C')
        obj = new{T}(T, matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(::Type{T}, m::Integer, n::Integer; transposed::Bool=false) where T <: BlasFloat
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, CU_NULL, T, 'R')
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, CU_NULL, T, 'C')
        end
        obj = new{T}(T, matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(v::CuVector{T}) where T <: BlasFloat
        m = length(v)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, m, 1, m, v, T, 'C')
        obj = new{T}(T, matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(A::CuMatrix{T}; transposed::Bool=false) where T <: BlasFloat
        m,n = size(A)
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, A, T, 'R')
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, A, T, 'C')
        end
        obj = new{T}(T, matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(A::CuSparseMatrixCSR{T,Cint}, structure::String, view::Char; index::Char='O') where T <: BlasFloat
        m,n = size(A)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateCsr(matrix_ref, m, n, nnz(A), A.rowPtr, CU_NULL,
                             A.colVal, A.nzVal, eltype(A.rowPtr), T, structure,
                             view, index)
        obj = new{T}(T, matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssMatrix_t}, matrix::CudssMatrix) = matrix.matrix


## Data

"""
    data = CudssData()

`CudssData` holds internal data (e.g., LU factors arrays).
"""
mutable struct CudssData
    data::cudssData_t

    function CudssData()
        data_ref = Ref{cudssData_t}()
        cudssDataCreate(handle(), data_ref)
        obj = new(data_ref[])
        finalizer(cudssDataDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssData_t}, data::CudssData) = data.data

function cudssDataDestroy(data::CudssData)
    cudssDataDestroy(handle(), data)
end

## Configuration

"""
    config = CudssConfig()

`CudssConfig` stores configuration settings for the solver.
"""
mutable struct CudssConfig
    config::cudssConfig_t

    function CudssConfig()
        config_ref = Ref{cudssConfig_t}()
        cudssConfigCreate(config_ref)
        obj = new(config_ref[])
        finalizer(cudssConfigDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssConfig_t}, config::CudssConfig) = config.config

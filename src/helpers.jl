# cuDSS helper functions
export CudssMatrix, CudssData, CudssConfig

## Matrix

"""
    matrix = CudssMatrix(v::CuVector)
    matrix = CudssMatrix(A::CuMatrix)
    matrix = CudssMatrix(A::CuSparseMatrixCSR, struture::Union{Char, String}, view::Char; index::Char='O')

`CudssMatrix` is a wrapper for `CuVector`, `CuMatrix` and `CuSparseMatrixCSR`.
`CudssMatrix` is used to pass matrix of the linear system, as well as solution and right-hand side.

`structure` specifies the stucture for sparse matrices:
- `'G'` or `"G"`: General matrix -- LDU factorization;
- `'S'` or `"S"`: Real symmetric matrix -- LDLᵀ factorization;
- `'H'` or `"H"`: Complex Hermitian matrix -- LDLᴴ factorization;
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
mutable struct CudssMatrix
    matrix::cudssMatrix_t

    function CudssMatrix(v::CuVector) 
        m = length(v)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, m, 1, m, v, eltype(v), 'C')
        obj = new(matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(A::CuMatrix; transposed::Bool=false) 
        m,n = size(A)
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, A, eltype(A), 'R')
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, A, eltype(A), 'C')
        end
        obj = new(matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(A::CuSparseMatrixCSR, structure::Union{Char, String}, view::Char; index::Char='O')
        m,n = size(A)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateCsr(matrix_ref, m, n, nnz(A), A.rowPtr, CU_NULL,
                             A.colVal, A.nzVal, eltype(A.rowPtr), eltype(A.nzVal), structure,
                             view, index)
        obj = new(matrix_ref[])
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

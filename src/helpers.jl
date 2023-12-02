# cuDSS helper functions
export CudssMatrix, CudssData, CudssConfig

## Matrix

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
            cudssMatrixCreateDn(matrix_ref, m, n, m, A, eltype(A), 'R')
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
        cudssMatrixCreateCsr(matrix_ref, m, n, nnz(A), A.rowPtr[1:end-1], A.rowPtr[2:end],
                             A.colVal, A.nzVal, eltype(A.rowPtr), eltype(A.nzVal), structure,
                             view, index)
        obj = new(matrix_ref[])
        finalizer(cudssMatrixDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssMatrix_t}, matrix::CudssMatrix) = matrix.matrix


## Data

mutable struct CudssData
    data::cudssData_t

    function CudssData()
        data_ref = Ref{cudssData_t}()
        cudssDataCreate(handle(), data_ref)
        obj = new(data_ref[])
        finalizer(CudssDataDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssData_t}, data::CudssData) = data.data

function CudssDataDestroy(data::CudssData)
    cudssDataDestroy(handle(), data)
end

## Configuration

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

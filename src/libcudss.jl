using CEnum

mutable struct cudssContext end

const cudssHandle_t = Ptr{cudssContext}

mutable struct cudssMatrix end

const cudssMatrix_t = Ptr{cudssMatrix}

mutable struct cudssData end

const cudssData_t = Ptr{cudssData}

mutable struct cudssConfig end

const cudssConfig_t = Ptr{cudssConfig}

@cenum cudssConfigParam_t::UInt32 begin
    CUDSS_CONFIG_REORDERING_ALG = 0
    CUDSS_CONFIG_FACTORIZATION_ALG = 1
    CUDSS_CONFIG_SOLVE_ALG = 2
    CUDSS_CONFIG_MATCHING_TYPE = 3
    CUDSS_CONFIG_SOLVE_MODE = 4
    CUDSS_CONFIG_IR_N_STEPS = 5
    CUDSS_CONFIG_IR_TOL = 6
    CUDSS_CONFIG_PIVOT_TYPE = 7
    CUDSS_CONFIG_PIVOT_THRESHOLD = 8
    CUDSS_CONFIG_PIVOT_EPSILON = 9
    CUDSS_CONFIG_MAX_LU_NNZ = 10
end

@cenum cudssDataParam_t::UInt32 begin
    CUDSS_DATA_INFO = 0
    CUDSS_DATA_LU_NNZ = 1
    CUDSS_DATA_NPIVOTS = 2
    CUDSS_DATA_INERTIA = 3
    CUDSS_DATA_PERM_REORDER = 4
    CUDSS_DATA_PERM_ROW = 5
    CUDSS_DATA_PERM_COL = 6
    CUDSS_DATA_DIAG = 7
    CUDSS_DATA_USER_PERM = 8
end

@cenum cudssPhase_t::UInt32 begin
    CUDSS_PHASE_ANALYSIS = 1
    CUDSS_PHASE_FACTORIZATION = 2
    CUDSS_PHASE_REFACTORIZATION = 4
    CUDSS_PHASE_SOLVE = 8
    CUDSS_PHASE_SOLVE_FWD = 16
    CUDSS_PHASE_SOLVE_DIAG = 32
    CUDSS_PHASE_SOLVE_BWD = 64
end

@cenum cudssStatus_t::UInt32 begin
    CUDSS_STATUS_SUCCESS = 0
    CUDSS_STATUS_NOT_INITIALIZED = 1
    CUDSS_STATUS_ALLOC_FAILED = 2
    CUDSS_STATUS_INVALID_VALUE = 3
    CUDSS_STATUS_NOT_SUPPORTED = 4
    CUDSS_STATUS_ARCH_MISMATCH = 5
    CUDSS_STATUS_EXECUTION_FAILED = 6
    CUDSS_STATUS_INTERNAL_ERROR = 7
    CUDSS_STATUS_ZERO_PIVOT = 8
end

@cenum cudssMatrixType_t::UInt32 begin
    CUDSS_MTYPE_GENERAL = 0
    CUDSS_MTYPE_SYMMETRIC = 1
    CUDSS_MTYPE_HERMITIAN = 2
    CUDSS_MTYPE_SPD = 3
    CUDSS_MTYPE_HPD = 4
end

@cenum cudssMatrixViewType_t::UInt32 begin
    CUDSS_MVIEW_FULL = 0
    CUDSS_MVIEW_LOWER = 1
    CUDSS_MVIEW_UPPER = 2
end

@cenum cudssIndexBase_t::UInt32 begin
    CUDSS_BASE_ZERO = 0
    CUDSS_BASE_ONE = 1
end

@cenum cudssLayout_t::UInt32 begin
    CUDSS_LAYOUT_COL_MAJOR = 0
    CUDSS_LAYOUT_ROW_MAJOR = 1
end

@cenum cudssAlgType_t::UInt32 begin
    CUDSS_ALG_DEFAULT = 0
    CUDSS_ALG_1 = 1
    CUDSS_ALG_2 = 2
    CUDSS_ALG_3 = 3
end

@cenum cudssPivotType_t::UInt32 begin
    CUDSS_PIVOT_COL = 0
    CUDSS_PIVOT_ROW = 1
    CUDSS_PIVOT_NONE = 2
end

@cenum cudssMatrixFormat_t::UInt32 begin
    CUDSS_MFORMAT_DENSE = 0
    CUDSS_MFORMAT_CSR = 1
end

@checked function cudssConfigSet(config, param, value, sizeInBytes)
    initialize_context()
    @ccall libcudss.cudssConfigSet(config::cudssConfig_t, param::cudssConfigParam_t,
                                   value::Ptr{Cvoid}, sizeInBytes::Cint)::cudssStatus_t
end

@checked function cudssConfigGet(config, param, value, sizeInBytes, sizeWritten)
    initialize_context()
    @ccall libcudss.cudssConfigGet(config::cudssConfig_t, param::cudssConfigParam_t,
                                   value::Ptr{Cvoid}, sizeInBytes::Cint,
                                   sizeWritten::Ptr{Cint})::cudssStatus_t
end

@checked function cudssDataSet(handle, data, param, value, sizeInBytes)
    initialize_context()
    @ccall libcudss.cudssDataSet(handle::cudssHandle_t, data::cudssData_t,
                                 param::cudssDataParam_t, value::Ptr{Cvoid},
                                 sizeInBytes::Cint)::cudssStatus_t
end

@checked function cudssDataGet(handle, data, param, value, sizeInBytes, sizeWritten)
    initialize_context()
    @ccall libcudss.cudssDataGet(handle::cudssHandle_t, data::cudssData_t,
                                 param::cudssDataParam_t, value::Ptr{Cvoid},
                                 sizeInBytes::Cint, sizeWritten::Ptr{Cint})::cudssStatus_t
end

@checked function cudssExecute(handle, phase, solverConfig, solverData, inputMatrix,
                               solution, rhs)
    initialize_context()
    @ccall libcudss.cudssExecute(handle::cudssHandle_t, phase::cudssPhase_t,
                                 solverConfig::cudssConfig_t, solverData::cudssData_t,
                                 inputMatrix::cudssMatrix_t, solution::cudssMatrix_t,
                                 rhs::cudssMatrix_t)::cudssStatus_t
end

@checked function cudssSetStream(handle, stream)
    initialize_context()
    @ccall libcudss.cudssSetStream(handle::cudssHandle_t, stream::Cint)::cudssStatus_t
end

@checked function cudssConfigCreate(solverConfig)
    initialize_context()
    @ccall libcudss.cudssConfigCreate(solverConfig::Ptr{cudssConfig_t})::cudssStatus_t
end

@checked function cudssConfigDestroy(solverConfig)
    initialize_context()
    @ccall libcudss.cudssConfigDestroy(solverConfig::cudssConfig_t)::cudssStatus_t
end

@checked function cudssDataCreate(handle, solverData)
    initialize_context()
    @ccall libcudss.cudssDataCreate(handle::cudssHandle_t,
                                    solverData::Ptr{cudssData_t})::cudssStatus_t
end

@checked function cudssDataDestroy(handle, solverData)
    initialize_context()
    @ccall libcudss.cudssDataDestroy(handle::cudssHandle_t,
                                     solverData::cudssData_t)::cudssStatus_t
end

@checked function cudssCreate(handle)
    initialize_context()
    @ccall libcudss.cudssCreate(handle::Ptr{cudssHandle_t})::cudssStatus_t
end

@checked function cudssDestroy(handle)
    initialize_context()
    @ccall libcudss.cudssDestroy(handle::cudssHandle_t)::cudssStatus_t
end

@checked function cudssGetProperty(propertyType, value)
    initialize_context()
    @ccall libcudss.cudssGetProperty(propertyType::libraryPropertyType,
                                     value::Ptr{Cint})::cudssStatus_t
end

@checked function cudssMatrixCreateDn(matrix, nrows, ncols, ld, values, valueType, layout)
    initialize_context()
    @ccall libcudss.cudssMatrixCreateDn(matrix::Ptr{cudssMatrix_t}, nrows::Int64,
                                        ncols::Int64, ld::Int64, values::CuPtr{Cvoid},
                                        valueType::cudaDataType_t,
                                        layout::cudssLayout_t)::cudssStatus_t
end

@checked function cudssMatrixCreateCsr(matrix, nrows, ncols, nnz, rowStart, rowEnd,
                                       colIndices, values, indexType, valueType, mtype,
                                       mview, indexBase)
    initialize_context()
    @ccall libcudss.cudssMatrixCreateCsr(matrix::Ptr{cudssMatrix_t}, nrows::Int64,
                                         ncols::Int64, nnz::Int64, rowStart::CuPtr{Cvoid},
                                         rowEnd::CuPtr{Cvoid}, colIndices::CuPtr{Cvoid},
                                         values::CuPtr{Cvoid}, indexType::cudaDataType_t,
                                         valueType::cudaDataType_t,
                                         mtype::cudssMatrixType_t,
                                         mview::cudssMatrixViewType_t,
                                         indexBase::cudssIndexBase_t)::cudssStatus_t
end

@checked function cudssMatrixDestroy(matrix)
    initialize_context()
    @ccall libcudss.cudssMatrixDestroy(matrix::cudssMatrix_t)::cudssStatus_t
end

@checked function cudssMatrixGetDn(matrix, nrows, ncols, ld, values, type, layout)
    initialize_context()
    @ccall libcudss.cudssMatrixGetDn(matrix::cudssMatrix_t, nrows::Ptr{Int64},
                                     ncols::Ptr{Int64}, ld::Ptr{Int64},
                                     values::Ptr{CuPtr{Cvoid}}, type::Ptr{cudaDataType_t},
                                     layout::Ptr{cudssLayout_t})::cudssStatus_t
end

@checked function cudssMatrixGetCsr(matrix, nrows, ncols, nnz, rowStart, rowEnd, colIndices,
                                    values, indexType, valueType, mtype, mview, indexBase)
    initialize_context()
    @ccall libcudss.cudssMatrixGetCsr(matrix::cudssMatrix_t, nrows::Ptr{Int64},
                                      ncols::Ptr{Int64}, nnz::Ptr{Int64},
                                      rowStart::Ptr{CuPtr{Cvoid}}, rowEnd::Ptr{CuPtr{Cvoid}},
                                      colIndices::Ptr{CuPtr{Cvoid}}, values::Ptr{CuPtr{Cvoid}},
                                      indexType::Ptr{cudaDataType_t},
                                      valueType::Ptr{cudaDataType_t},
                                      mtype::Ptr{cudssMatrixType_t},
                                      mview::Ptr{cudssMatrixViewType_t},
                                      indexBase::Ptr{cudssIndexBase_t})::cudssStatus_t
end

@checked function cudssMatrixSetValues(matrix, values)
    initialize_context()
    @ccall libcudss.cudssMatrixSetValues(matrix::cudssMatrix_t,
                                         values::CuPtr{Cvoid})::cudssStatus_t
end

@checked function cudssMatrixSetCsrPointers(matrix, rowOffsets, rowEnd, colIndices, values)
    initialize_context()
    @ccall libcudss.cudssMatrixSetCsrPointers(matrix::cudssMatrix_t, rowOffsets::CuPtr{Cvoid},
                                              rowEnd::CuPtr{Cvoid}, colIndices::CuPtr{Cvoid},
                                              values::CuPtr{Cvoid})::cudssStatus_t
end

@checked function cudssMatrixGetFormat(matrix, format)
    initialize_context()
    @ccall libcudss.cudssMatrixGetFormat(matrix::cudssMatrix_t,
                                         format::Ptr{cudssMatrixFormat_t})::cudssStatus_t
end

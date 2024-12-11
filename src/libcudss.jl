using CEnum

# CUDSS uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

const cudaDataType_t = cudaDataType

@cenum cudssOpType_t::UInt32 begin
    CUDSS_SUM = 0
    CUDSS_MAX = 1
    CUDSS_MIN = 2
end

struct cudssDistributedInterface_t
    cudssCommRank::Ptr{Cvoid}
    cudssCommSize::Ptr{Cvoid}
    cudssSend::Ptr{Cvoid}
    cudssRecv::Ptr{Cvoid}
    cudssBcast::Ptr{Cvoid}
    cudssReduce::Ptr{Cvoid}
    cudssAllreduce::Ptr{Cvoid}
    cudssScatterv::Ptr{Cvoid}
    cudssCommSplit::Ptr{Cvoid}
    cudssCommFree::Ptr{Cvoid}
end

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
    CUDSS_CONFIG_HYBRID_MODE = 11
    CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT = 12
    CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY = 13
end

@cenum cudssDataParam_t::UInt32 begin
    CUDSS_DATA_INFO = 0
    CUDSS_DATA_LU_NNZ = 1
    CUDSS_DATA_NPIVOTS = 2
    CUDSS_DATA_INERTIA = 3
    CUDSS_DATA_PERM_REORDER_ROW = 4
    CUDSS_DATA_PERM_REORDER_COL = 5
    CUDSS_DATA_PERM_ROW = 6
    CUDSS_DATA_PERM_COL = 7
    CUDSS_DATA_DIAG = 8
    CUDSS_DATA_USER_PERM = 9
    CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN = 10
    CUDSS_DATA_COMM = 11
    CUDSS_DATA_MEMORY_ESTIMATES = 12
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
    CUDSS_STATUS_EXECUTION_FAILED = 5
    CUDSS_STATUS_INTERNAL_ERROR = 6
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

struct cudssDeviceMemHandler_t
    ctx::Ptr{Cvoid}
    device_alloc::Ptr{Cvoid}
    device_free::Ptr{Cvoid}
    name::NTuple{64,Cchar}
end

@checked function cudssConfigSet(config, param, value, sizeInBytes)
    initialize_context()
    @gcsafe_ccall libcudss.cudssConfigSet(config::cudssConfig_t, param::cudssConfigParam_t,
                                          value::Ptr{Cvoid},
                                          sizeInBytes::Csize_t)::cudssStatus_t
end

@checked function cudssConfigGet(config, param, value, sizeInBytes, sizeWritten)
    initialize_context()
    @gcsafe_ccall libcudss.cudssConfigGet(config::cudssConfig_t, param::cudssConfigParam_t,
                                          value::Ptr{Cvoid}, sizeInBytes::Csize_t,
                                          sizeWritten::Ptr{Csize_t})::cudssStatus_t
end

@checked function cudssDataSet(handle, data, param, value, sizeInBytes)
    initialize_context()
    @gcsafe_ccall libcudss.cudssDataSet(handle::cudssHandle_t, data::cudssData_t,
                                        param::cudssDataParam_t, value::PtrOrCuPtr{Cvoid},
                                        sizeInBytes::Csize_t)::cudssStatus_t
end

@checked function cudssDataGet(handle, data, param, value, sizeInBytes, sizeWritten)
    initialize_context()
    @gcsafe_ccall libcudss.cudssDataGet(handle::cudssHandle_t, data::cudssData_t,
                                        param::cudssDataParam_t, value::PtrOrCuPtr{Cvoid},
                                        sizeInBytes::Csize_t,
                                        sizeWritten::Ptr{Csize_t})::cudssStatus_t
end

@checked function cudssExecute(handle, phase, solverConfig, solverData, inputMatrix,
                               solution, rhs)
    initialize_context()
    @gcsafe_ccall libcudss.cudssExecute(handle::cudssHandle_t, phase::cudssPhase_t,
                                        solverConfig::cudssConfig_t,
                                        solverData::cudssData_t, inputMatrix::cudssMatrix_t,
                                        solution::cudssMatrix_t,
                                        rhs::cudssMatrix_t)::cudssStatus_t
end

@checked function cudssSetStream(handle, stream)
    initialize_context()
    @gcsafe_ccall libcudss.cudssSetStream(handle::cudssHandle_t,
                                          stream::cudaStream_t)::cudssStatus_t
end

@checked function cudssSetCommLayer(handle, commLibFileName)
    initialize_context()
    @gcsafe_ccall libcudss.cudssSetCommLayer(handle::cudssHandle_t,
                                             commLibFileName::Cstring)::cudssStatus_t
end

@checked function cudssConfigCreate(solverConfig)
    initialize_context()
    @gcsafe_ccall libcudss.cudssConfigCreate(solverConfig::Ptr{cudssConfig_t})::cudssStatus_t
end

@checked function cudssConfigDestroy(solverConfig)
    initialize_context()
    @gcsafe_ccall libcudss.cudssConfigDestroy(solverConfig::cudssConfig_t)::cudssStatus_t
end

@checked function cudssDataCreate(handle, solverData)
    initialize_context()
    @gcsafe_ccall libcudss.cudssDataCreate(handle::cudssHandle_t,
                                           solverData::Ptr{cudssData_t})::cudssStatus_t
end

@checked function cudssDataDestroy(handle, solverData)
    initialize_context()
    @gcsafe_ccall libcudss.cudssDataDestroy(handle::cudssHandle_t,
                                            solverData::cudssData_t)::cudssStatus_t
end

@checked function cudssCreate(handle)
    initialize_context()
    @gcsafe_ccall libcudss.cudssCreate(handle::Ptr{cudssHandle_t})::cudssStatus_t
end

@checked function cudssDestroy(handle)
    initialize_context()
    @gcsafe_ccall libcudss.cudssDestroy(handle::cudssHandle_t)::cudssStatus_t
end

@checked function cudssGetProperty(propertyType, value)
    @gcsafe_ccall libcudss.cudssGetProperty(propertyType::libraryPropertyType,
                                            value::Ptr{Cint})::cudssStatus_t
end

@checked function cudssMatrixCreateDn(matrix, nrows, ncols, ld, values, valueType, layout)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixCreateDn(matrix::Ptr{cudssMatrix_t}, nrows::Int64,
                                               ncols::Int64, ld::Int64,
                                               values::CuPtr{Cvoid},
                                               valueType::cudaDataType_t,
                                               layout::cudssLayout_t)::cudssStatus_t
end

@checked function cudssMatrixCreateCsr(matrix, nrows, ncols, nnz, rowStart, rowEnd,
                                       colIndices, values, indexType, valueType, mtype,
                                       mview, indexBase)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixCreateCsr(matrix::Ptr{cudssMatrix_t}, nrows::Int64,
                                                ncols::Int64, nnz::Int64,
                                                rowStart::CuPtr{Cvoid},
                                                rowEnd::CuPtr{Cvoid},
                                                colIndices::CuPtr{Cvoid},
                                                values::CuPtr{Cvoid},
                                                indexType::cudaDataType_t,
                                                valueType::cudaDataType_t,
                                                mtype::cudssMatrixType_t,
                                                mview::cudssMatrixViewType_t,
                                                indexBase::cudssIndexBase_t)::cudssStatus_t
end

@checked function cudssMatrixCreateBatchDn(matrix, batchCount, nrows, ncols, ld, values,
                                           valueType, layout)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixCreateBatchDn(matrix::Ptr{cudssMatrix_t},
                                                    batchCount::Int64, nrows::Ptr{Cvoid},
                                                    ncols::Ptr{Cvoid}, ld::Ptr{Cvoid},
                                                    values::Ptr{CuPtr{Cvoid}},
                                                    valueType::cudaDataType_t,
                                                    layout::cudssLayout_t)::cudssStatus_t
end

@checked function cudssMatrixCreateBatchCsr(matrix, batchCount, nrows, ncols, nnz, rowStart,
                                            rowEnd, colIndices, values, indexType,
                                            valueType, mtype, mview, indexBase)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixCreateBatchCsr(matrix::Ptr{cudssMatrix_t},
                                                     batchCount::Int64, nrows::Ptr{Cvoid},
                                                     ncols::Ptr{Cvoid}, nnz::Ptr{Cvoid},
                                                     rowStart::Ptr{CuPtr{Cvoid}},
                                                     rowEnd::Ptr{CuPtr{Cvoid}},
                                                     colIndices::Ptr{CuPtr{Cvoid}},
                                                     values::Ptr{CuPtr{Cvoid}},
                                                     indexType::cudaDataType_t,
                                                     valueType::cudaDataType_t,
                                                     mtype::cudssMatrixType_t,
                                                     mview::cudssMatrixViewType_t,
                                                     indexBase::cudssIndexBase_t)::cudssStatus_t
end

@checked function cudssMatrixDestroy(matrix)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixDestroy(matrix::cudssMatrix_t)::cudssStatus_t
end

@checked function cudssMatrixGetDn(matrix, nrows, ncols, ld, values, type, layout)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixGetDn(matrix::cudssMatrix_t, nrows::Ptr{Int64},
                                            ncols::Ptr{Int64}, ld::Ptr{Int64},
                                            values::Ptr{CuPtr{Cvoid}},
                                            type::Ptr{cudaDataType_t},
                                            layout::Ptr{cudssLayout_t})::cudssStatus_t
end

@checked function cudssMatrixGetCsr(matrix, nrows, ncols, nnz, rowStart, rowEnd, colIndices,
                                    values, indexType, valueType, mtype, mview, indexBase)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixGetCsr(matrix::cudssMatrix_t, nrows::Ptr{Int64},
                                             ncols::Ptr{Int64}, nnz::Ptr{Int64},
                                             rowStart::Ptr{CuPtr{Cvoid}},
                                             rowEnd::Ptr{CuPtr{Cvoid}},
                                             colIndices::Ptr{CuPtr{Cvoid}},
                                             values::Ptr{CuPtr{Cvoid}},
                                             indexType::Ptr{cudaDataType_t},
                                             valueType::Ptr{cudaDataType_t},
                                             mtype::Ptr{cudssMatrixType_t},
                                             mview::Ptr{cudssMatrixViewType_t},
                                             indexBase::Ptr{cudssIndexBase_t})::cudssStatus_t
end

@checked function cudssMatrixSetValues(matrix, values)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixSetValues(matrix::cudssMatrix_t,
                                                values::CuPtr{Cvoid})::cudssStatus_t
end

@checked function cudssMatrixSetCsrPointers(matrix, rowOffsets, rowEnd, colIndices, values)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixSetCsrPointers(matrix::cudssMatrix_t,
                                                     rowOffsets::CuPtr{Cvoid},
                                                     rowEnd::CuPtr{Cvoid},
                                                     colIndices::CuPtr{Cvoid},
                                                     values::CuPtr{Cvoid})::cudssStatus_t
end

@checked function cudssMatrixGetBatchDn(matrix, batchCount, nrows, ncols, ld, values, type,
                                        layout)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixGetBatchDn(matrix::cudssMatrix_t,
                                                 batchCount::Ptr{Int64},
                                                 nrows::Ptr{Ptr{Cvoid}},
                                                 ncols::Ptr{Ptr{Cvoid}},
                                                 ld::Ptr{Ptr{Cvoid}},
                                                 values::Ptr{Ptr{CuPtr{Cvoid}}},
                                                 type::Ptr{cudaDataType_t},
                                                 layout::Ptr{cudssLayout_t})::cudssStatus_t
end

@checked function cudssMatrixGetBatchCsr(matrix, batchCount, nrows, ncols, nnz, rowStart,
                                         rowEnd, colIndices, values, indexType, valueType,
                                         mtype, mview, indexBase)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixGetBatchCsr(matrix::cudssMatrix_t,
                                                  batchCount::Ptr{Int64},
                                                  nrows::Ptr{Ptr{Cvoid}},
                                                  ncols::Ptr{Ptr{Cvoid}},
                                                  nnz::Ptr{Ptr{Cvoid}},
                                                  rowStart::Ptr{Ptr{CuPtr{Cvoid}}},
                                                  rowEnd::Ptr{Ptr{CuPtr{Cvoid}}},
                                                  colIndices::Ptr{Ptr{CuPtr{Cvoid}}},
                                                  values::Ptr{Ptr{CuPtr{Cvoid}}},
                                                  indexType::Ptr{cudaDataType_t},
                                                  valueType::Ptr{cudaDataType_t},
                                                  mtype::Ptr{cudssMatrixType_t},
                                                  mview::Ptr{cudssMatrixViewType_t},
                                                  indexBase::Ptr{cudssIndexBase_t})::cudssStatus_t
end

@checked function cudssMatrixSetBatchValues(matrix, values)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixSetBatchValues(matrix::cudssMatrix_t,
                                                     values::Ptr{CuPtr{Cvoid}})::cudssStatus_t
end

@checked function cudssMatrixSetBatchCsrPointers(matrix, rowOffsets, rowEnd, colIndices,
                                                 values)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixSetBatchCsrPointers(matrix::cudssMatrix_t,
                                                          rowOffsets::Ptr{CuPtr{Cvoid}},
                                                          rowEnd::Ptr{CuPtr{Cvoid}},
                                                          colIndices::Ptr{CuPtr{Cvoid}},
                                                          values::Ptr{CuPtr{Cvoid}})::cudssStatus_t
end

@checked function cudssMatrixGetFormat(matrix, format)
    initialize_context()
    @gcsafe_ccall libcudss.cudssMatrixGetFormat(matrix::cudssMatrix_t,
                                                format::Ptr{cudssMatrixFormat_t})::cudssStatus_t
end

@checked function cudssGetDeviceMemHandler(handle, handler)
    initialize_context()
    @gcsafe_ccall libcudss.cudssGetDeviceMemHandler(handle::cudssHandle_t,
                                                    handler::Ptr{cudssDeviceMemHandler_t})::cudssStatus_t
end

@checked function cudssSetDeviceMemHandler(handle, handler)
    initialize_context()
    @gcsafe_ccall libcudss.cudssSetDeviceMemHandler(handle::cudssHandle_t,
                                                    handler::Ptr{cudssDeviceMemHandler_t})::cudssStatus_t
end

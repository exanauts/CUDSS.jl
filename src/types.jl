# cuDSS types

const CUDSS_DATA_PARAMETERS = ("info", "lu_nnz", "npivots", "inertia", "perm_reorder_row",
                               "perm_reorder_col", "perm_row", "perm_col", "diag", "user_perm",
                               "hybrid_device_memory_min", "comm", "memory_estimates")

const CUDSS_CONFIG_PARAMETERS = ("reordering_alg", "factorization_alg", "solve_alg", "matching_type",
                                 "solve_mode", "ir_n_steps", "ir_tol", "pivot_type", "pivot_threshold",
                                 "pivot_epsilon", "max_lu_nnz", "hybrid_mode", "hybrid_device_memory_limit",
                                 "use_cuda_register_memory")

const CUDSS_TYPES = Dict{String, DataType}(
    # data type
    "info" => Cint,
    "lu_nnz" => Int64,
    "npivots" => Cint,
    "inertia" => Tuple{Cint, Cint},
    "perm_reorder_row" => Vector{Cint},
    "perm_reorder_col" => Vector{Cint},
    "perm_row" => Vector{Cint},
    "perm_col" => Vector{Cint},
    "diag" => Vector{Float64},
    "user_perm" => Vector{Cint},
    "hybrid_device_memory_min" => Int64,
    "comm" => Ptr{Cvoid},
    "memory_estimates" => Vector{Int64},
    # config type
    "reordering_alg" => cudssAlgType_t,
    "factorization_alg" => cudssAlgType_t,
    "solve_alg" => cudssAlgType_t,
    "matching_type" => Cint,
    "solve_mode" => Cint,
    "ir_n_steps" => Cint,
    "ir_tol" => Float64,
    "pivot_type" => cudssPivotType_t,
    "pivot_threshold" => Float64,
    "pivot_epsilon" => Float64,
    "max_lu_nnz" => Int64,
    "hybrid_mode" => Cint,
    "hybrid_device_memory_limit" => Int64,
    "use_cuda_register_memory" => Cint
)

## config type

function Base.convert(::Type{cudssConfigParam_t}, config::String)
    if config == "reordering_alg"
        return CUDSS_CONFIG_REORDERING_ALG
    elseif config == "factorization_alg"
        return CUDSS_CONFIG_FACTORIZATION_ALG
    elseif config == "solve_alg"
        return CUDSS_CONFIG_SOLVE_ALG
    elseif config == "matching_type"
        return CUDSS_CONFIG_MATCHING_TYPE
    elseif config == "solve_mode"
        return CUDSS_CONFIG_SOLVE_MODE
    elseif config == "ir_n_steps"
        return CUDSS_CONFIG_IR_N_STEPS
    elseif config == "ir_tol"
        return CUDSS_CONFIG_IR_TOL
    elseif config == "pivot_type"
        return CUDSS_CONFIG_PIVOT_TYPE
    elseif config == "pivot_threshold"
        return CUDSS_CONFIG_PIVOT_THRESHOLD
    elseif config == "pivot_epsilon"
        return CUDSS_CONFIG_PIVOT_EPSILON
    elseif config == "max_lu_nnz"
        return CUDSS_CONFIG_MAX_LU_NNZ
    elseif config == "hybrid_mode"
        return CUDSS_CONFIG_HYBRID_MODE
    elseif config == "hybrid_device_memory_limit"
        return CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT
    elseif config == "use_cuda_register_memory"
        return CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY
    else
        throw(ArgumentError("Unknown config parameter $config"))
    end
end

## data type

function Base.convert(::Type{cudssDataParam_t}, data::String)
    if data == "info"
        return CUDSS_DATA_INFO
    elseif data == "lu_nnz"
        return CUDSS_DATA_LU_NNZ
    elseif data == "npivots"
        return CUDSS_DATA_NPIVOTS
    elseif data == "inertia"
        return CUDSS_DATA_INERTIA
    elseif data == "perm_reorder_row"
        return CUDSS_DATA_PERM_REORDER_ROW
    elseif data == "perm_reorder_col"
        return CUDSS_DATA_PERM_REORDER_COL
    elseif data == "perm_row"
        return CUDSS_DATA_PERM_ROW
    elseif data == "perm_col"
        return CUDSS_DATA_PERM_COL
    elseif data == "diag"
        return CUDSS_DATA_DIAG
    elseif data == "user_perm"
        return CUDSS_DATA_USER_PERM
    elseif data == "hybrid_device_memory_min"
        return CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN
    elseif data == "comm"
        return CUDSS_DATA_COMM
    elseif data == "memory_estimates"
        return CUDSS_DATA_MEMORY_ESTIMATES
    else
        throw(ArgumentError("Unknown data parameter $data"))
    end
end

## phase type

function Base.convert(::Type{cudssPhase_t}, phase::String)
    if phase == "analysis"
        return CUDSS_PHASE_ANALYSIS
    elseif phase == "factorization"
        return CUDSS_PHASE_FACTORIZATION
    elseif phase == "refactorization"
        return CUDSS_PHASE_REFACTORIZATION
    elseif phase == "solve"
        return CUDSS_PHASE_SOLVE
    elseif phase == "solve_fwd"
        return CUDSS_PHASE_SOLVE_FWD
    elseif phase == "solve_diag"
        return CUDSS_PHASE_SOLVE_DIAG
    elseif phase == "solve_bwd"
        return CUDSS_PHASE_SOLVE_BWD
    else
        throw(ArgumentError("Unknown phase $phase"))
    end
end

## matrix structure type

function Base.convert(::Type{cudssMatrixType_t}, structure::String)
    if structure == "G"
        return CUDSS_MTYPE_GENERAL
    elseif structure == "S"
        return CUDSS_MTYPE_SYMMETRIC
    elseif structure == "H"
        return CUDSS_MTYPE_HERMITIAN
    elseif structure == "SPD"
        return CUDSS_MTYPE_SPD
    elseif structure == "HPD"
        return CUDSS_MTYPE_HPD
    else
        throw(ArgumentError("Unknown structure $structure"))
    end
end

## view type

function Base.convert(::Type{cudssMatrixViewType_t}, view::Char)
    if view == 'F'
        return CUDSS_MVIEW_FULL
    elseif view == 'L'
        return CUDSS_MVIEW_LOWER
    elseif view == 'U'
        return CUDSS_MVIEW_UPPER
    else
        throw(ArgumentError("Unknown view $view"))
    end
end

## index base

function Base.convert(::Type{cudssIndexBase_t}, index::Char)
    if index == 'Z'
        return CUDSS_BASE_ZERO
    elseif index == 'O'
        return CUDSS_BASE_ONE
    else
        throw(ArgumentError("Unknown index $index"))
    end
end

## layout type

function Base.convert(::Type{cudssLayout_t}, layout::Char)
    if layout == 'R'
        CUDSS_LAYOUT_ROW_MAJOR
    elseif layout == 'C'
        CUDSS_LAYOUT_COL_MAJOR
    else
        throw(ArgumentError("Unknown layout $layout"))
    end
end

## algorithm type

function Base.convert(::Type{cudssAlgType_t}, algorithm::String)
    if algorithm == "default"
        CUDSS_ALG_DEFAULT
    elseif algorithm == "algo1"
        CUDSS_ALG_1
    elseif algorithm == "algo2"
        CUDSS_ALG_2
    elseif algorithm == "algo3"
        CUDSS_ALG_3
    else
        throw(ArgumentError("Unknown algorithm $algorithm"))
    end
end

## pivot type

function Base.convert(::Type{cudssPivotType_t}, pivoting::Char)
    if pivoting == 'C'
        return CUDSS_PIVOT_COL
    elseif pivoting == 'R'
        return CUDSS_PIVOT_ROW
    elseif pivoting == 'N'
        return CUDSS_PIVOT_NONE
    else
        throw(ArgumentError("Unknown pivoting $pivoting"))
    end
end

# matrix format type

function Base.convert(::Type{cudssMatrixFormat_t}, format::Char)
    if format == 'D'
        return CUDSS_MFORMAT_DENSE
    elseif format == 'S'
        return CUDSS_MFORMAT_CSR
    else
        throw(ArgumentError("Unknown format $format"))
    end
end

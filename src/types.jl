# cuDSS types

const CUDSS_DATA_PARAMETERS = ("info", "lu_nnz", "npivots", "inertia", "perm_reorder_row",
                               "perm_reorder_col", "perm_row", "perm_col", "diag", "user_perm",
                               "hybrid_device_memory_min", "comm", "memory_estimates",
                               "perm_matching", "scale_row", "scale_col", "nsuperpanels",
                               "user_schur_indices", "schur_shape", "schur_matrix",
                               "user_elimination_tree", "elimination_tree", "user_host_interrupt")

const CUDSS_CONFIG_PARAMETERS = ("reordering_alg", "factorization_alg", "solve_alg", "use_matching",
                                 "matching_alg", "solve_mode", "ir_n_steps", "ir_tol", "pivot_type",
                                 "pivot_threshold", "pivot_epsilon", "max_lu_nnz", "hybrid_mode",
                                 "hybrid_device_memory_limit", "use_cuda_register_memory", "host_nthreads",
                                 "hybrid_execute_mode", "pivot_epsilon_alg", "nd_nlevels", "ubatch_size",
                                 "ubatch_index", "use_superpanels", "device_count", "device_indices",
                                 "schur_mode" "deterministic_mode")

const CUDSS_TYPES = Dict{String, Type}(
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
    "perm_matching" => Vector{Cint},
    "scale_row" => Vector{Float64},
    "scale_col" => Vector{Float64},
    "nsuperpanels" => Cint,
    "user_schur_indices" => Cint,
    "schur_shape" => Tuple{Int64, Int64, Int64},
    "schur_matrix" => cudssMatrix_t
    "user_elimination_tree" => Vector{Cint},
    "elimination_tree" => Vector{Cint},
    "user_host_interrupt" => Vector{Cint},
    # config type
    "reordering_alg" => cudssAlgType_t,
    "factorization_alg" => cudssAlgType_t,
    "solve_alg" => cudssAlgType_t,
    "use_matching" => Cint,
    "matching_alg" => cudssAlgType_t,
    "solve_mode" => Cint,
    "ir_n_steps" => Cint,
    "ir_tol" => Float64,
    "pivot_type" => cudssPivotType_t,
    "pivot_threshold" => Float64,
    "pivot_epsilon" => Float64,
    "max_lu_nnz" => Int64,
    "hybrid_mode" => Cint,
    "hybrid_device_memory_limit" => Int64,
    "use_cuda_register_memory" => Cint,
    "host_nthreads" => Cint,
    "hybrid_execute_mode" => Cint,
    "pivot_epsilon_alg" => cudssAlgType_t,
    "nd_nlevels" => Cint,
    "ubatch_size" => Cint,
    "ubatch_index" => Cint,
    "use_superpanels" => Cint,
    "device_count" => Cint,
    "device_indices" => Vector{Cint},
    "schur_mode" => Cint,
    "deterministic_mode" => Cint,
)

## config type

function Base.convert(::Type{cudssConfigParam_t}, config::String)
    if config == "reordering_alg"
        return CUDSS_CONFIG_REORDERING_ALG
    elseif config == "factorization_alg"
        return CUDSS_CONFIG_FACTORIZATION_ALG
    elseif config == "solve_alg"
        return CUDSS_CONFIG_SOLVE_ALG
    elseif config == "use_matching"
        return CUDSS_CONFIG_USE_MATCHING
    elseif config == "matching_alg"
        return CUDSS_CONFIG_MATCHING_ALG
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
    elseif config == "host_nthreads"
        return CUDSS_CONFIG_HOST_NTHREADS
    elseif config == "hybrid_execute_mode"
        return CUDSS_CONFIG_HYBRID_EXECUTE_MODE
    elseif config == "pivot_epsilon_alg"
        return CUDSS_CONFIG_PIVOT_EPSILON_ALG
    elseif config == "nd_nlevels"
        return CUDSS_CONFIG_ND_NLEVELS
    elseif config == "ubatch_size"
        return CUDSS_CONFIG_UBATCH_SIZE
    elseif config == "ubatch_index"
        return CUDSS_CONFIG_UBATCH_INDEX
    elseif config == "use_superpanels"
        return CUDSS_CONFIG_USE_SUPERPANELS
    elseif config == "device_count"
        return CUDSS_CONFIG_DEVICE_COUNT
    elseif config == "device_indices"
        return CUDSS_CONFIG_DEVICE_INDICES
    elseif config == "schur_mode"
        return CUDSS_CONFIG_SCHUR_MODE
    elseif config == "deterministic_mode"
        return CUDSS_CONFIG_DETERMINISTIC_MODE
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
    elseif data == "perm_matching"
        return CUDSS_DATA_PERM_MATCHING
    elseif data == "scale_row"
        return CUDSS_DATA_SCALE_ROW
    elseif data == "scale_col"
        return CUDSS_DATA_SCALE_COL
    elseif data == "nsuperpanels"
        return CUDSS_DATA_NSUPERPANELS
    elseif data == "user_schur_indices"
        return CUDSS_DATA_USER_SCHUR_INDICES
    elseif data == "schur_shape"
        return CUDSS_DATA_SCHUR_SHAPE
    elseif data == "schur_matrix"
        return CUDSS_DATA_SCHUR_MATRIX
    elseif data == "user_elimination_tree"
        return CUDSS_DATA_USER_ELIMINATION_TREE
    elseif data == "elimination_tree"
        return CUDSS_DATA_ELIMINATION_TREE
    elseif data == "user_host_interrupt"
        return CUDSS_DATA_USER_HOST_INTERRUPT
    else
        throw(ArgumentError("Unknown data parameter $data"))
    end
end

## phase type

function Base.convert(::Type{cudssPhase_t}, phase::String)
    if phase == "reordering"
        return CUDSS_PHASE_REORDERING
    elseif phase == "symbolic_factorization"
        return CUDSS_PHASE_SYMBOLIC_FACTORIZATION
    elseif phase == "analysis"
        return CUDSS_PHASE_ANALYSIS
    elseif phase == "factorization"
        return CUDSS_PHASE_FACTORIZATION
    elseif phase == "refactorization"
        return CUDSS_PHASE_REFACTORIZATION
    elseif phase == "solve_fwd_perm"
        return CUDSS_PHASE_SOLVE_FWD_PERM
    elseif phase == "solve_fwd"
        return CUDSS_PHASE_SOLVE_FWD
    elseif phase == "solve_diag"
        return CUDSS_PHASE_SOLVE_DIAG
    elseif phase == "solve_bwd"
        return CUDSS_PHASE_SOLVE_BWD
    elseif phase == "solve_bwd_perm"
        return CUDSS_PHASE_SOLVE_BWD_PERM
    elseif phase == "solve_refinement"
        return CUDSS_PHASE_SOLVE_REFINEMENT
    elseif phase == "solve"
        return CUDSS_PHASE_SOLVE
    else
        throw(ArgumentError("Unknown phase $phase"))
    end
end

Base.convert(::Type{Cint}, phase::cudssPhase_t) = Cint(phase)

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
        return CUDSS_LAYOUT_ROW_MAJOR
    elseif layout == 'C'
        return CUDSS_LAYOUT_COL_MAJOR
    else
        throw(ArgumentError("Unknown layout $layout"))
    end
end

## algorithm type

function Base.convert(::Type{cudssAlgType_t}, algorithm::String)
    if algorithm == "default"
        return CUDSS_ALG_DEFAULT
    elseif algorithm == "algo1"
        return CUDSS_ALG_1
    elseif algorithm == "algo2"
        return CUDSS_ALG_2
    elseif algorithm == "algo3"
        return CUDSS_ALG_3
    elseif algorithm == "algo4"
        return CUDSS_ALG_4
    elseif algorithm == "algo5"
        return CUDSS_ALG_5
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

function Base.convert(::Type{cudssMatrixFormat_t}, format::String)
    if format == "DENSE"
        return CUDSS_MFORMAT_DENSE
    elseif format == "CSR"
        return CUDSS_MFORMAT_CSR
    elseif format == "BATCH"
        return CUDSS_MFORMAT_BATCH
    elseif format == "DISTRIBUTED"
        return CUDSS_MFORMAT_DISTRIBUTED
    else
        throw(ArgumentError("Unknown format $format"))
    end
end

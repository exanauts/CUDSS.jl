var documenterSearchIndex = {"docs":
[{"location":"options/#Iterative-refinement","page":"Options","title":"Iterative refinement","text":"","category":"section"},{"location":"options/","page":"Options","title":"Options","text":"using CUDA, CUDA.CUSPARSE\nusing CUDSS\nusing LinearAlgebra\nusing SparseArrays\n\nT = Float64\nn = 100\np = 5\nA_cpu = sprand(T, n, n, 0.01)\nA_cpu = A_cpu + I\nB_cpu = rand(T, n, p)\n\nA_gpu = CuSparseMatrixCSR(A_cpu)\nB_gpu = CuMatrix(B_cpu)\nX_gpu = similar(B_gpu)\n\nsolver = CudssSolver(A_gpu, \"G\", 'F')\n\n# Perform one step of iterative refinement\nir = 1\ncudss_set(solver, \"ir_n_steps\", ir)\n\ncudss(\"analysis\", solver, X_gpu, B_gpu)\ncudss(\"factorization\", solver, X_gpu, B_gpu)\ncudss(\"solve\", solver, X_gpu, B_gpu)\n\nR_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu\nnorm(R_gpu)","category":"page"},{"location":"options/#User-permutation","page":"Options","title":"User permutation","text":"","category":"section"},{"location":"options/","page":"Options","title":"Options","text":"using CUDA, CUDA.CUSPARSE\nusing CUDSS\nusing LinearAlgebra\nusing SparseArrays\nusing AMD\n\nT = ComplexF64\nn = 100\nA_cpu = sprand(T, n, n, 0.01)\nA_cpu = A_cpu' * A_cpu + I\nb_cpu = rand(T, n)\n\nA_gpu = CuSparseMatrixCSR(A_cpu)\nb_gpu = CuVector(b_cpu)\nx_gpu = similar(b_gpu)\n\nsolver = CudssSolver(A_gpu, \"HPD\", 'F')\n\n# Provide a user permutation\npermutation = amd(A_cpu) |> Vector{Cint}\ncudss_set(solver, \"user_perm\", permutation)\n\ncudss(\"analysis\", solver, x_gpu, b_gpu)\ncudss(\"factorization\", solver, x_gpu, b_gpu)\ncudss(\"solve\", solver, x_gpu, b_gpu)\n\nr_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu\nnorm(r_gpu)","category":"page"},{"location":"options/#Hybrid-mode","page":"Options","title":"Hybrid mode","text":"","category":"section"},{"location":"options/","page":"Options","title":"Options","text":"using CUDA, CUDA.CUSPARSE\nusing CUDSS\nusing LinearAlgebra\nusing SparseArrays\n\nT = Float64\nn = 100\nA_cpu = sprand(T, n, n, 0.01)\nA_cpu = A_cpu + A_cpu' + I\nb_cpu = rand(T, n)\n\nA_gpu = CuSparseMatrixCSR(A_cpu)\nb_gpu = CuVector(b_cpu)\nx_gpu = similar(b_gpu)\n\nsolver = CudssSolver(A_gpu, \"S\", 'F')\n\n# Use the hybrid mode (host and device memory)\ncudss_set(solver, \"hybrid_mode\", 1)\n\ncudss(\"analysis\", solver, x_gpu, b_gpu)\n\n# Minimal amount of device memory required in the hybrid memory mode.\nnbytes = cudss_get(solver, \"hybrid_device_memory_min\")\n\n# Device memory limit for the hybrid memory mode.\n# Only use it if you don't want to rely on the internal default heuristic.\ncudss_set(solver, \"hybrid_device_memory_limit\", nbytes_gpu)\n\ncudss(\"factorization\", solver, x_gpu, b_gpu)\ncudss(\"solve\", solver, x_gpu, b_gpu)\n\nr_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu\nnorm(r_gpu)","category":"page"},{"location":"#Home","page":"Home","title":"CUDSS.jl documentation","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CUDSS.jl is a Julia interface to the NVIDIA cuDSS library. NVIDIA cuDSS provides three factorizations (LDU, LDLᵀ, LLᵀ) for solving sparse linear systems on GPUs. For more details on using cuDSS, refer to the official cuDSS documentation.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> ]\npkg> add CUDSS\npkg> test CUDSS","category":"page"},{"location":"#Types","page":"Home","title":"Types","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CudssMatrix\nCudssConfig\nCudssData\nCudssSolver","category":"page"},{"location":"#CUDSS.CudssMatrix","page":"Home","title":"CUDSS.CudssMatrix","text":"matrix = CudssMatrix(v::CuVector{T})\nmatrix = CudssMatrix(A::CuMatrix{T})\nmatrix = CudssMatrix(A::CuSparseMatrixCSR{T,Cint}, struture::String, view::Char; index::Char='O')\n\nThe type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nCudssMatrix is a wrapper for CuVector, CuMatrix and CuSparseMatrixCSR. CudssMatrix is used to pass matrix of the linear system, as well as solution and right-hand side.\n\nstructure specifies the stucture for sparse matrices:\n\n\"G\": General matrix – LDU factorization;\n\"S\": Real symmetric matrix – LDLᵀ factorization;\n\"H\": Complex Hermitian matrix – LDLᴴ factorization;\n\"SPD\": Symmetric positive-definite matrix – LLᵀ factorization;\n\"HPD\": Hermitian positive-definite matrix – LLᴴ factorization.\n\nview specifies matrix view for sparse matrices:\n\n'L': Lower-triangular matrix and all values above the main diagonal are ignored;\n'U': Upper-triangular matrix and all values below the main diagonal are ignored;\n'F': Full matrix.\n\nindex specifies indexing base for sparse matrix indices:\n\n'Z': 0-based indexing;\n'O': 1-based indexing.\n\n\n\n\n\n","category":"type"},{"location":"#CUDSS.CudssConfig","page":"Home","title":"CUDSS.CudssConfig","text":"config = CudssConfig()\n\nCudssConfig stores configuration settings for the solver.\n\n\n\n\n\n","category":"type"},{"location":"#CUDSS.CudssData","page":"Home","title":"CUDSS.CudssData","text":"data = CudssData()\ndata = CudssData(cudss_handle::cudssHandle_t)\n\nCudssData holds internal data (e.g., LU factors arrays).\n\n\n\n\n\n","category":"type"},{"location":"#CUDSS.CudssSolver","page":"Home","title":"CUDSS.CudssSolver","text":"solver = CudssSolver(A::CuSparseMatrixCSR{T,Cint}, structure::String, view::Char; index::Char='O')\nsolver = CudssSolver(matrix::CudssMatrix{T}, config::CudssConfig, data::CudssData)\n\nThe type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nCudssSolver contains all structures required to solve linear systems with cuDSS. One constructor of CudssSolver takes as input the same parameters as CudssMatrix.\n\nstructure specifies the stucture for sparse matrices:\n\n\"G\": General matrix – LDU factorization;\n\"S\": Real symmetric matrix – LDLᵀ factorization;\n\"H\": Complex Hermitian matrix – LDLᴴ factorization;\n\"SPD\": Symmetric positive-definite matrix – LLᵀ factorization;\n\"HPD\": Hermitian positive-definite matrix – LLᴴ factorization.\n\nview specifies matrix view for sparse matrices:\n\n'L': Lower-triangular matrix and all values above the main diagonal are ignored;\n'U': Upper-triangular matrix and all values below the main diagonal are ignored;\n'F': Full matrix.\n\nindex specifies indexing base for sparse matrix indices:\n\n'Z': 0-based indexing;\n'O': 1-based indexing.\n\nCudssSolver can be also constructed from the three structures CudssMatrix, CudssConfig and CudssData if needed.\n\n\n\n\n\n","category":"type"},{"location":"#Functions","page":"Home","title":"Functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"cudss_set\ncudss_get\ncudss","category":"page"},{"location":"#CUDSS.cudss_set","page":"Home","title":"CUDSS.cudss_set","text":"cudss_set(matrix::CudssMatrix{T}, v::CuVector{T})\ncudss_set(matrix::CudssMatrix{T}, A::CuMatrix{T})\ncudss_set(matrix::CudssMatrix{T}, A::CuSparseMatrixCSR{T,Cint})\ncudss_set(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})\ncudss_set(solver::CudssSolver, parameter::String, value)\ncudss_set(config::CudssConfig, parameter::String, value)\ncudss_set(data::CudssData, parameter::String, value)\n\nThe type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nThe available configuration parameters are:\n\n\"reordering_alg\": Algorithm for the reordering phase (\"default\", \"algo1\", \"algo2\" or \"algo3\");\n\"factorization_alg\": Algorithm for the factorization phase (\"default\", \"algo1\", \"algo2\" or \"algo3\");\n\"solve_alg\": Algorithm for the solving phase (\"default\", \"algo1\", \"algo2\" or \"algo3\");\n\"matching_type\": Type of matching;\n\"solve_mode\": Potential modificator on the system matrix (transpose or adjoint);\n\"ir_n_steps\": Number of steps during the iterative refinement;\n\"ir_tol\": Iterative refinement tolerance;\n\"pivot_type\": Type of pivoting ('C', 'R' or 'N');\n\"pivot_threshold\": Pivoting threshold which is used to determine if digonal element is subject to pivoting;\n\"pivot_epsilon\": Pivoting epsilon, absolute value to replace singular diagonal elements;\n\"max_lu_nnz\": Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;\n\"hybrid_mode\": Memory mode – 0 (default = device-only) or 1 (hybrid = host/device);\n\"hybrid_device_memory_limit\": User-defined device memory limit (number of bytes) for the hybrid memory mode;\n\"use_cuda_register_memory\": A flag to enable (1) or disable (0) usage of cudaHostRegister() by the hybrid memory mode.\n\nThe available data parameters are:\n\n\"user_perm\": User permutation to be used instead of running the reordering algorithms;\n\"comm\": Communicator for Multi-GPU multi-node mode.\n\n\n\n\n\n","category":"function"},{"location":"#CUDSS.cudss_get","page":"Home","title":"CUDSS.cudss_get","text":"value = cudss_get(solver::CudssSolver, parameter::String)\nvalue = cudss_get(config::CudssConfig, parameter::String)\nvalue = cudss_get(data::CudssData, parameter::String)\n\nThe available configuration parameters are:\n\n\"reordering_alg\": Algorithm for the reordering phase;\n\"factorization_alg\": Algorithm for the factorization phase;\n\"solve_alg\": Algorithm for the solving phase;\n\"matching_type\": Type of matching;\n\"solve_mode\": Potential modificator on the system matrix (transpose or adjoint);\n\"ir_n_steps\": Number of steps during the iterative refinement;\n\"ir_tol\": Iterative refinement tolerance;\n\"pivot_type\": Type of pivoting;\n\"pivot_threshold\": Pivoting threshold which is used to determine if digonal element is subject to pivoting;\n\"pivot_epsilon\": Pivoting epsilon, absolute value to replace singular diagonal elements;\n\"max_lu_nnz\": Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;\n\"hybrid_mode\": Memory mode – 0 (default = device-only) or 1 (hybrid = host/device);\n\"hybrid_device_memory_limit\": User-defined device memory limit (number of bytes) for the hybrid memory mode;\n\"use_cuda_register_memory\": A flag to enable (1) or disable (0) usage of cudaHostRegister() by the hybrid memory mode.\n\nThe available data parameters are:\n\n\"info\": Device-side error information;\n\"lu_nnz\": Number of non-zero entries in LU factors;\n\"npivots\": Number of pivots encountered during factorization;\n\"inertia\": Tuple of positive and negative indices of inertia for symmetric and hermitian non positive-definite matrix types;\n\"perm_reorder_row\": Reordering permutation for the rows;\n\"perm_reorder_col\": Reordering permutation for the columns;\n\"perm_row\": Final row permutation (which includes effects of both reordering and pivoting);\n\"perm_col\": Final column permutation (which includes effects of both reordering and pivoting);\n\"diag\": Diagonal of the factorized matrix;\n\"hybrid_device_memory_min\": Minimal amount of device memory (number of bytes) required in the hybrid memory mode.\n\nThe data parameters \"info\", \"lu_nnz\", \"perm_reorder_row\", \"perm_reorder_col\" and \"hybrid_device_memory_min\" require the phase \"analyse\" performed by cudss. The data parameters \"npivots\", \"inertia\" and \"diag\" require the phases \"analyse\" and \"factorization\" performed by cudss. The data parameters \"perm_row\" and \"perm_col\" are available but not yet functional.\n\n\n\n\n\n","category":"function"},{"location":"#CUDSS.cudss","page":"Home","title":"CUDSS.cudss","text":"cudss(phase::String, solver::CudssSolver{T}, x::CuVector{T}, b::CuVector{T})\ncudss(phase::String, solver::CudssSolver{T}, X::CuMatrix{T}, B::CuMatrix{T})\ncudss(phase::String, solver::CudssSolver{T}, X::CudssMatrix{T}, B::CudssMatrix{T})\n\nThe type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nThe available phases are \"analysis\", \"factorization\", \"refactorization\" and \"solve\". The phases \"solve_fwd\", \"solve_diag\" and \"solve_bwd\" are available but not yet functional.\n\n\n\n\n\n","category":"function"},{"location":"generic/#LLᵀ-and-LLᴴ","page":"Generic API","title":"LLᵀ and LLᴴ","text":"","category":"section"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"    LinearAlgebra.cholesky(A::CuSparseMatrixCSR{T,Cint}; view::Char='F') where T <: LinearAlgebra.BlasFloat\n    LinearAlgebra.cholesky!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: LinearAlgebra.BlasFloat","category":"page"},{"location":"generic/#LinearAlgebra.cholesky-Union{Tuple{CuSparseMatrixCSR{T, Int32}}, Tuple{T}} where T<:Union{Float32, Float64, ComplexF64, ComplexF32}","page":"Generic API","title":"LinearAlgebra.cholesky","text":"solver = cholesky(A::CuSparseMatrixCSR{T,Cint}; view::Char='F')\n\nCompute the LLᴴ factorization of a sparse matrix A on an NVIDIA GPU. The type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nInput argument\n\nA: a sparse Hermitian positive definite matrix stored in the CuSparseMatrixCSR format.\n\nKeyword argument\n\n*view: A character that specifies which triangle of the sparse matrix is provided. Possible options are L for the lower triangle, U for the upper triangle, and F for the full matrix.\n\nOutput argument\n\nsolver: Opaque structure CudssSolver that stores the factors of the LLᴴ decomposition.\n\n\n\n\n\n","category":"method"},{"location":"generic/#LinearAlgebra.cholesky!-Union{Tuple{T}, Tuple{CudssSolver{T}, CuSparseMatrixCSR{T, Int32}}} where T<:Union{Float32, Float64, ComplexF64, ComplexF32}","page":"Generic API","title":"LinearAlgebra.cholesky!","text":"solver = cholesky!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})\n\nCompute the LLᴴ factorization of a sparse matrix A on an NVIDIA GPU, reusing the symbolic factorization stored in solver. The type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\n\n\n\n\n","category":"method"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"using CUDA, CUDA.CUSPARSE\nusing CUDSS\nusing LinearAlgebra\nusing SparseArrays\n\nT = ComplexF64\nR = real(T)\nn = 100\np = 5\nA_cpu = sprand(T, n, n, 0.01)\nA_cpu = A_cpu * A_cpu' + I\nB_cpu = rand(T, n, p)\n\nA_gpu = CuSparseMatrixCSR(A_cpu |> triu)\nB_gpu = CuMatrix(B_cpu)\nX_gpu = similar(B_gpu)\n\nF = cholesky(A_gpu, view='U')\nX_gpu = F \\ B_gpu\n\nR_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu\nnorm(R_gpu)\n\n# In-place LLᴴ\nd_gpu = rand(R, n) |> CuVector\nA_gpu = A_gpu + Diagonal(d_gpu)\ncholesky!(F, A_gpu)\n\nC_cpu = rand(T, n, p)\nC_gpu = CuMatrix(C_cpu)\nldiv!(X_gpu, F, C_gpu)\n\nR_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu\nnorm(R_gpu)","category":"page"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"note: Note\nIf we only store one triangle of A_gpu, we can also use the wrappers Symmetric and Hermitian instead of using the keyword argument view in cholesky. For real matrices, both wrappers are allowed but only Hermitian can be used for complex matrices.","category":"page"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"H_gpu = Hermitian(A_gpu, :U)\nF = cholesky(H_gpu)","category":"page"},{"location":"generic/#LDLᵀ-and-LDLᴴ","page":"Generic API","title":"LDLᵀ and LDLᴴ","text":"","category":"section"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"    LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T,Cint}; view::Char='F') where T <: LinearAlgebra.BlasFloat\n    LinearAlgebra.ldlt!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: LinearAlgebra.BlasFloat","category":"page"},{"location":"generic/#LinearAlgebra.ldlt-Union{Tuple{CuSparseMatrixCSR{T, Int32}}, Tuple{T}} where T<:Union{Float32, Float64, ComplexF64, ComplexF32}","page":"Generic API","title":"LinearAlgebra.ldlt","text":"solver = ldlt(A::CuSparseMatrixCSR{T,Cint}; view::Char='F')\n\nCompute the LDLᴴ factorization of a sparse matrix A on an NVIDIA GPU. The type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nInput argument\n\nA: a sparse Hermitian matrix stored in the CuSparseMatrixCSR format.\n\nKeyword argument\n\n*view: A character that specifies which triangle of the sparse matrix is provided. Possible options are L for the lower triangle, U for the upper triangle, and F for the full matrix.\n\nOutput argument\n\nsolver: Opaque structure CudssSolver that stores the factors of the LDLᴴ decomposition.\n\n\n\n\n\n","category":"method"},{"location":"generic/#LinearAlgebra.ldlt!-Union{Tuple{T}, Tuple{CudssSolver{T}, CuSparseMatrixCSR{T, Int32}}} where T<:Union{Float32, Float64, ComplexF64, ComplexF32}","page":"Generic API","title":"LinearAlgebra.ldlt!","text":"solver = ldlt!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})\n\nCompute the LDLᴴ factorization of a sparse matrix A on an NVIDIA GPU, reusing the symbolic factorization stored in solver. The type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\n\n\n\n\n","category":"method"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"using CUDA, CUDA.CUSPARSE\nusing CUDSS\nusing LinearAlgebra\nusing SparseArrays\n\nT = Float64\nR = real(T)\nn = 100\np = 5\nA_cpu = sprand(T, n, n, 0.05) + I\nA_cpu = A_cpu + A_cpu'\nB_cpu = rand(T, n, p)\n\nA_gpu = CuSparseMatrixCSR(A_cpu |> tril)\nB_gpu = CuMatrix(B_cpu)\nX_gpu = similar(B_gpu)\n\nF = ldlt(A_gpu, view='L')\nX_gpu = F \\ B_gpu\n\nR_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu\nnorm(R_gpu)\n\n# In-place LDLᵀ\nd_gpu = rand(R, n) |> CuVector\nA_gpu = A_gpu + Diagonal(d_gpu)\nldlt!(F, A_gpu)\n\nC_cpu = rand(T, n, p)\nC_gpu = CuMatrix(C_cpu)\nldiv!(X_gpu, F, C_gpu)\n\nR_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu\nnorm(R_gpu)","category":"page"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"note: Note\nIf we only store one triangle of A_gpu, we can also use the wrappers Symmetric and Hermitian instead of using the keyword argument view in ldlt. For real matrices, both wrappers are allowed but only Hermitian can be used for complex matrices.","category":"page"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"S_gpu = Symmetric(A_gpu, :L)\nF = ldlt(S_gpu)","category":"page"},{"location":"generic/#LU","page":"Generic API","title":"LU","text":"","category":"section"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"    LinearAlgebra.lu(A::CuSparseMatrixCSR{T,Cint}) where T <: LinearAlgebra.BlasFloat\n    LinearAlgebra.lu!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: LinearAlgebra.BlasFloat","category":"page"},{"location":"generic/#LinearAlgebra.lu-Union{Tuple{CuSparseMatrixCSR{T, Int32}}, Tuple{T}} where T<:Union{Float32, Float64, ComplexF64, ComplexF32}","page":"Generic API","title":"LinearAlgebra.lu","text":"solver = lu(A::CuSparseMatrixCSR{T,Cint})\n\nCompute the LU factorization of a sparse matrix A on an NVIDIA GPU. The type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\nInput argument\n\nA: a sparse square matrix stored in the CuSparseMatrixCSR format.\n\nOutput argument\n\nsolver: an opaque structure CudssSolver that stores the factors of the LU decomposition.\n\n\n\n\n\n","category":"method"},{"location":"generic/#LinearAlgebra.lu!-Union{Tuple{T}, Tuple{CudssSolver{T}, CuSparseMatrixCSR{T, Int32}}} where T<:Union{Float32, Float64, ComplexF64, ComplexF32}","page":"Generic API","title":"LinearAlgebra.lu!","text":"solver = lu!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})\n\nCompute the LU factorization of a sparse matrix A on an NVIDIA GPU, reusing the symbolic factorization stored in solver. The type T can be Float32, Float64, ComplexF32 or ComplexF64.\n\n\n\n\n\n","category":"method"},{"location":"generic/","page":"Generic API","title":"Generic API","text":"using CUDA, CUDA.CUSPARSE\nusing CUDSS\nusing LinearAlgebra\nusing SparseArrays\n\nT = Float64\nn = 100\nA_cpu = sprand(T, n, n, 0.05) + I\nb_cpu = rand(T, n)\n\nA_gpu = CuSparseMatrixCSR(A_cpu)\nb_gpu = CuVector(b_cpu)\n\nF = lu(A_gpu)\nx_gpu = F \\ b_gpu\n\nr_gpu = b_gpu - A_gpu * x_gpu\nnorm(r_gpu)\n\n# In-place LU\nd_gpu = rand(T, n) |> CuVector\nA_gpu = A_gpu + Diagonal(d_gpu)\nlu!(F, A_gpu)\n\nc_cpu = rand(T, n)\nc_gpu = CuVector(c_cpu)\nldiv!(x_gpu, F, c_gpu)\n\nr_gpu = c_gpu - A_gpu * x_gpu\nnorm(r_gpu)","category":"page"}]
}

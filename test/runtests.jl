using Test
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays

function cudss_version()
  @test CUDSS.version() == v"0.1.0"
end

function cudss_dense()
  m = 20
  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu = rand(T, m, n)
    A_gpu = CuMatrix(A_cpu)
    matrix = CudssMatrix(A_gpu)
    format = Ref{CUDSS.cudssMatrixFormat_t}()
    CUDSS.cudssMatrixGetFormat(matrix, format)
    @test format[] == CUDSS.CUDSS_MFORMAT_DENSE

    A_cpu2 = rand(T, m, n)
    A_gpu2 = CuMatrix(A_cpu2)
    cudss_set(matrix, A_gpu2)
  end
end

function cudss_sparse()
  m = 20
  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu = sprand(T, m, n, 1.0)
    A_cpu = A_cpu + A_cpu'
    A_gpu = CuSparseMatrixCSR(A_cpu)
    @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
      @testset "view = $view" for view in ('L', 'U', 'F')
        matrix = CudssMatrix(A_gpu, structure, view) 
        format = Ref{CUDSS.cudssMatrixFormat_t}()
        CUDSS.cudssMatrixGetFormat(matrix, format)
        @test format[] == CUDSS.CUDSS_MFORMAT_CSR

        # cudssMatrixSetCsrPointers
        # A_cpu2 = sprand(T, m, n, 1.0)
        # A_cpu2 = A_cpu2 + A_cpu2'
        # A_gpu2 = CuSparseMatrixCSR(A_cpu2)
        # cudss_set(matrix, A_gpu2)
      end
    end
  end
end

function cudss_data()
  data = CudssData()
  @testset "parameter = $parameter" for parameter in ("info", "lu_nnz", "npivots", "inertia", "perm_reorder",
                                                      "perm_row", "perm_col", "diag", "user_perm")
    val = cudss_get(data, parameter)
  end
end

function cudss_config()
  config = CudssConfig()
  @testset "parameter = $parameter" for parameter in ("reordering_alg", "factorization_alg", "solve_alg",
                                                      "matching_type", "solve_mode", "ir_n_steps", "ir_tol",
                                                      "pivot_type", "pivot_threshold", "pivot_epsilon", "max_lu_nnz")
    val = cudss_get(config, parameter)
  end
end

function cudss_main()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu = rand(T, 10, 10)
    x_cpu = rand(T, 10)
    b_cpu = rand(T, 10)

    A_gpu = CuMatrix(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    matrix = CudssMatrix(A_gpu)
    solution = CudssMatrix(x_gpu)
    rhs = CudssMatrix(b_gpu)

    config = CudssConfig()
    data = CudssData()
    cudss("analysis", config, data, matrix, solution, rhs)
    cudss("factorization", config, data, matrix, solution, rhs)
    cudss("solve", phase, config, data, matrix, solution, rhs)
  end
end

@testset "CUDSS" begin
  @testset "version" begin
    cudss_version()
  end

  @testset "CudssMatrix" begin
    cudss_dense()
    cudss_sparse()
  end

  @testset "CudssData" begin
    cudss_data()
  end

  @testset "CudssConfig" begin
    cudss_config()
  end

  @testset "CudssExecution" begin
    cudss_main()
  end
end

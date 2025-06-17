function cudss_batched_dense()
  n = 20
  p = 4
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "CuVector" begin
      A_cpu = rand(T, n)
      A_gpu = [CuVector(A_cpu)]
      matrix = CudssBatchedMatrix(A_gpu)
      format = Ref{Cint}()
      CUDSS.cudssMatrixGetFormat(matrix, format)
      @test format[] == CUDSS.CUDSS_MFORMAT_BATCH

      A_cpu2 = rand(T, n)
      A_gpu2 = [CuVector(A_cpu2)]
      cudss_set(matrix, A_gpu2)
    end

    @testset "CuMatrix" begin
      A_cpu = rand(T, n, p)
      A_gpu = [CuMatrix(A_cpu)]
      matrix = CudssBatchedMatrix(A_gpu)
      format = Ref{CInt}()
      CUDSS.cudssMatrixGetFormat(matrix, format)
      @test format[] == CUDSS.CUDSS_MFORMAT_BATCH

      A_cpu2 = rand(T, n, p)
      A_gpu2 = [CuMatrix(A_cpu2)]
      cudss_set(matrix, A_gpu2)
    end
  end
end

function cudss_batched_sparse()
  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu = sprand(T, n, n, 1.0)
    A_cpu = A_cpu + A_cpu'
    A_gpu = [CuSparseMatrixCSR(A_cpu)]
    @testset "view = $view" for view in ('L', 'U', 'F')
      @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
        matrix = CudssBatchedMatrix(A_gpu, structure, view)
        format = Ref{Cint}()
        CUDSS.cudssMatrixGetFormat(matrix, format)
        @test format[] == CUDSS.CUDSS_MFORMAT_BATCH

        A_cpu2 = sprand(T, n, n, 1.0)
        A_cpu2 = A_cpu2 + A_cpu2'
        A_gpu2 = [CuSparseMatrixCSR(A_cpu2)]
        cudss_set(matrix, A_gpu2)
      end
    end
  end
end

function cudss_batched_solver()
  n = 20
  m = 30
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu1 = sprand(T, n, n, 1.0)
    A_cpu1 = A_cpu1 * A_cpu1' + I
    A_gpu1 = CuSparseMatrixCSR(A_cpu1)
    A_cpu2 = sprand(T, m, m, 1.0)
    A_cpu2 = A_cpu2 * A_cpu2' + I
    A_gpu2 = CuSparseMatrixCSR(A_cpu2)
    A_gpu = [A_gpu1, A_gpu2]
    @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
      @testset "view = $view" for view in ('L', 'U', 'F')
        solver = CudssBatchedSolver(A_gpu, structure, view)

        x_cpu1 = zeros(T, n)
        x_gpu1 = CuVector(x_cpu1)
        x_cpu2 = zeros(T, m)
        x_gpu2 = CuVector(x_cpu2)
        x_gpu = [x_gpu1, x_gpu2]
        b_cpu1 = rand(T, n)
        b_gpu1 = CuVector(b_cpu1)
        b_cpu2 = rand(T, m)
        b_gpu2 = CuVector(b_cpu2)
        b_gpu = [b_gpu1, b_gpu2]
        cudss("analysis", solver, x_gpu, b_gpu)
        cudss("factorization", solver, x_gpu, b_gpu)

        @testset "data parameter = $parameter" for parameter in CUDSS_DATA_PARAMETERS
          parameter ∈ ("perm_row", "perm_col", "perm_reorder_row", "perm_reorder_col", "diag", "comm") && continue
          @testset "cudss_get" begin
            (parameter == "user_perm") && continue
            (parameter == "inertia") && !(structure ∈ ("S", "H")) && continue
            val = cudss_get(solver, parameter)
          end
          @testset "cudss_set" begin
            if parameter == "user_perm"
              perm_cpu = Cint[i for i=n:-1:1]
              cudss_set(solver, parameter, perm_cpu)
              perm_gpu = CuVector{Cint}(perm_cpu)
              cudss_set(solver, parameter, perm_gpu)
            end
            (parameter == "info") && cudss_set(solver, parameter, 1)
          end
        end

        @testset "config parameter = $parameter" for parameter in CUDSS_CONFIG_PARAMETERS
          @testset "cudss_get" begin
            val = cudss_get(solver, parameter)
          end
          @testset "cudss_set" begin
            # (parameter == "matching_type") && cudss_set(solver, parameter, 0)
            (parameter == "solve_mode") && cudss_set(solver, parameter, 0)
            (parameter == "ir_n_steps") && cudss_set(solver, parameter, 1)
            (parameter == "ir_tol") && cudss_set(solver, parameter, 1e-8)
            (parameter == "pivot_threshold") && cudss_set(solver, parameter, 2.0)
            (parameter == "pivot_epsilon") && cudss_set(solver, parameter, 1e-12)
            (parameter == "max_lu_nnz") && cudss_set(solver, parameter, 10)
            (parameter == "hybrid_device_memory_limit") && cudss_set(solver, parameter, 2048)
            for algo in ("default", "algo1", "algo2", "algo3")
              (parameter == "reordering_alg") && cudss_set(solver, parameter, algo)
              (parameter == "factorization_alg") && cudss_set(solver, parameter, algo)
              (parameter == "solve_alg") && cudss_set(solver, parameter, algo)
            end
            for flag in (0, 1)
              (parameter == "hybrid_mode") && cudss_set(solver, parameter, flag)
              (parameter == "use_cuda_register_memory") && cudss_set(solver, parameter, flag)
            end
            for pivoting in ('C', 'R', 'N')
              (parameter == "pivot_type") && cudss_set(solver, parameter, pivoting)
            end
          end
        end
      end
    end
  end
end

function cudss_batched_execution()
  n = [40, 50, 80, 4, 12, 28, 51]
  p = [2, 3, 4, 2, 5, 5, 6]
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "Unsymmetric -- Non-Hermitian" begin
      @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
        A_cpu = [sprand(T, n[i], n[i], 0.02) + I for i = 1:7]
        x_cpu = [zeros(T, n[i]) for i = 1:7]
        b_cpu = [rand(T, n[i]) for i = 1:7]
        A_gpu = CuSparseMatrixCSR.(A_cpu)
        x_gpu = CuVector.(x_cpu)
        b_gpu = CuVector.(b_cpu)

        matrix = CudssBatchedMatrix(A_gpu, "G", 'F')
        config = CudssConfig()
        data = CudssData()
        solver = CudssBatchedSolver(matrix, config, data)
        cudss_set(solver, "pivot_type", pivot)

        cudss("analysis", solver, x_gpu, b_gpu)
        cudss("factorization", solver, x_gpu, b_gpu)
        cudss("solve", solver, x_gpu, b_gpu)

        r_gpu = b_gpu .- A_gpu .* x_gpu
        @test mapreduce(r -> norm(r) ≤ √eps(R), &, r_gpu)

        # In-place LU
        d_gpu = [rand(T, n[i]) |> CuVector for i = 1:7]
        A_gpu = [A_gpu[i] + Diagonal(d_gpu[i]) for i = 1:7]
        cudss_set(solver, A_gpu)

        c_cpu = [rand(T, n[i]) for i = 1:7]
        c_gpu = CuVector.(c_cpu)

        cudss("refactorization", solver, x_gpu, c_gpu)
        cudss("solve", solver, x_gpu, c_gpu)

        r_gpu = c_gpu .- A_gpu .* x_gpu
        @test mapreduce(r -> norm(r) ≤ √eps(R), &, r_gpu)
      end
    end

    @testset "Symmetric -- Hermitian" begin
      @testset "view = $view" for view in ('F', 'L', 'U')
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
          A_cpu = [sprand(T, n[i], n[i], 0.01) + I for i = 1:7]
          A_cpu = [A_cpu[i] + A_cpu[i]' for i = 1:7]
          X_cpu = [zeros(T, n[i], p[i]) for i = 1:7]
          B_cpu = [rand(T, n[i], p[i]) for i = 1:7]

          (view == 'L') && (A_gpu = CuSparseMatrixCSR.(A_cpu .|> tril))
          (view == 'U') && (A_gpu = CuSparseMatrixCSR.(A_cpu .|> triu))
          (view == 'F') && (A_gpu = CuSparseMatrixCSR.(A_cpu))
          X_gpu = CuMatrix.(X_cpu)
          B_gpu = CuMatrix.(B_cpu)

          structure = T <: Real ? "S" : "H"
          matrix = CudssBatchedMatrix(A_gpu, structure, view)
          config = CudssConfig()
          data = CudssData()
          solver = CudssBatchedSolver(matrix, config, data)
          cudss_set(solver, "pivot_type", pivot)

          cudss("analysis", solver, X_gpu, B_gpu)
          cudss("factorization", solver, X_gpu, B_gpu)
          cudss("solve", solver, X_gpu, B_gpu)

          R_gpu = B_gpu .- CuSparseMatrixCSR.(A_cpu) .* X_gpu
          @test mapreduce(r -> norm(r) ≤ √eps(R), &, R_gpu)

          # In-place LDLᵀ / LDLᴴ
          d_gpu = [rand(R, n[i]) |> CuVector for i = 1:7]
          A_gpu = [A_gpu[i] + Diagonal(d_gpu[i]) for i = 1:7]
          cudss_set(solver, A_gpu)

          C_cpu = [rand(T, n[i], p[i]) for i = 1:7]
          C_gpu = CuMatrix.(C_cpu)

          cudss("refactorization", solver, X_gpu, C_gpu)
          cudss("solve", solver, X_gpu, C_gpu)

          R_gpu = C_gpu .- ( CuSparseMatrixCSR.(A_cpu) .+ Diagonal.(d_gpu) ) .* X_gpu
          @test mapreduce(r -> norm(r) ≤ √eps(R), &, R_gpu)
        end
      end
    end

    @testset "SPD -- HPD" begin
      @testset "view = $view" for view in ('F', 'L', 'U')
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
          A_cpu = [sprand(T, n[i], n[i], 0.01) for i = 1:7]
          A_cpu = [A_cpu[i] * A_cpu[i]' + I for i = 1:7]
          X_cpu = [zeros(T, n[i], p[i]) for i = 1:7]
          B_cpu = [rand(T, n[i], p[i]) for i = 1:7]

          (view == 'L') && (A_gpu = CuSparseMatrixCSR.(A_cpu .|> tril))
          (view == 'U') && (A_gpu = CuSparseMatrixCSR.(A_cpu .|> triu))
          (view == 'F') && (A_gpu = CuSparseMatrixCSR.(A_cpu))
          X_gpu = CuMatrix.(X_cpu)
          B_gpu = CuMatrix.(B_cpu)

          structure = T <: Real ? "SPD" : "HPD"
          matrix = CudssBatchedMatrix(A_gpu, structure, view)
          config = CudssConfig()
          data = CudssData()
          solver = CudssBatchedSolver(matrix, config, data)
          cudss_set(solver, "pivot_type", pivot)

          cudss("analysis", solver, X_gpu, B_gpu)
          cudss("factorization", solver, X_gpu, B_gpu)
          cudss("solve", solver, X_gpu, B_gpu)

          R_gpu = B_gpu .- CuSparseMatrixCSR.(A_cpu) .* X_gpu
          @test mapreduce(r -> norm(r) ≤ √eps(R), &, R_gpu)

          # In-place LLᵀ / LLᴴ
          d_gpu = [rand(R, n[i]) |> CuVector for i = 1:7]
          A_gpu = [A_gpu[i] + Diagonal(d_gpu[i]) for i = 1:7]
          cudss_set(solver, A_gpu)

          C_cpu = [rand(T, n[i], p[i]) for i = 1:7]
          C_gpu = CuMatrix.(C_cpu)

          cudss("refactorization", solver, X_gpu, C_gpu)
          cudss("solve", solver, X_gpu, C_gpu)

          R_gpu = C_gpu .- ( CuSparseMatrixCSR.(A_cpu) .+ Diagonal.(d_gpu) ) .* X_gpu
          @test mapreduce(r -> norm(r) ≤ √eps(R), &, R_gpu)
        end
      end
    end
  end
end

function hybrid_batched_mode()
  function hybrid_batched_lu(T, A_cpu, x_cpu, b_cpu)
    A_gpu = CuSparseMatrixCSR.(A_cpu)
    x_gpu = CuVector.(x_cpu)
    b_gpu = CuVector.(b_cpu)

    solver = CudssBatchedSolver(A_gpu, "G", 'F')
    cudss_set(solver, "hybrid_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu .- A_gpu .* x_gpu
    return norm.(r_gpu)
  end

  function hybrid_batched_ldlt(T, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR.(A_cpu .|> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR.(A_cpu .|> triu)
    else
      A_gpu = CuSparseMatrixCSR.(A_cpu)
    end
    x_gpu = CuVector.(x_cpu)
    b_gpu = CuVector.(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssBatchedSolver(A_gpu, structure, uplo)
    cudss_set(solver, "hybrid_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu .- CuSparseMatrixCSR.(A_cpu) .* x_gpu
    return norm.(r_gpu)
  end

  function hybrid_batched_llt(T, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR.(A_cpu .|> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR.(A_cpu .|> triu)
    else
      A_gpu = CuSparseMatrixCSR.(A_cpu)
    end
    x_gpu = CuVector.(x_cpu)
    b_gpu = CuVector.(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssBatchedSolver(A_gpu, structure, uplo)
    cudss_set(solver, "hybrid_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu .- CuSparseMatrixCSR.(A_cpu) .* x_gpu
    return norm.(r_gpu)
  end

  n = [20, 25, 10, 5, 2]
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "LU" begin
      A_cpu = [sprand(T, n[i], n[i], 0.05) + I for i = 1:5]
      x_cpu = [zeros(T, n[i]) for i = 1:5]
      b_cpu = [rand(T, n[i]) for i = 1:5]
      res = hybrid_batched_lu(T, A_cpu, x_cpu, b_cpu)
      @test mapreduce(r -> r ≤ √eps(R), &, res)
    end
    @testset "LDLᵀ / LDLᴴ" begin
      A_cpu = [sprand(T, n[i], n[i], 0.05) + I for i = 1:5]
      A_cpu = [A_cpu[i] + A_cpu[i]' for i = 1:5]
      x_cpu = [zeros(T, n[i]) for i = 1:5]
      b_cpu = [rand(T, n[i]) for i = 1:5]
      @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
        res = hybrid_batched_ldlt(T, A_cpu, x_cpu, b_cpu, uplo)
        @test mapreduce(r -> r ≤ √eps(R), &, res)
      end
    end
    @testset "LLᵀ / LLᴴ" begin
      A_cpu = [sprand(T, n[i], n[i], 0.01) for i = 1:5]
      A_cpu = [A_cpu[i] * A_cpu[i]' + I for i = 1:5]
      x_cpu = [zeros(T, n[i]) for i = 1:5]
      b_cpu = [rand(T, n[i]) for i = 1:5]
      @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
        res = hybrid_batched_llt(T, A_cpu, x_cpu, b_cpu, uplo)
        @test mapreduce(r -> r ≤ √eps(R), &, res)
      end
    end
  end
end

function refactorization_batched_cholesky()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    n = 100
    p = 4
    nbatch = 8

    batch_A_cpu = SparseMatrixCSC{T}[]
    batch_A_gpu = CuSparseMatrixCSR{T,Cint}[]
    batch_X_gpu = CuMatrix{T}[]
    batch_B_gpu = CuMatrix{T}[]

    for i = 1:nbatch
      A_cpu = sprand(T, n, n, 0.05) + I
      A_cpu = A_cpu * A_cpu' -20 * I
      X_cpu = zeros(T, n, p)
      B_cpu = rand(T, n, p)

      push!(batch_A_cpu, A_cpu)
      push!(batch_A_gpu, A_cpu |> triu |> CuSparseMatrixCSR)
      push!(batch_X_gpu, X_cpu |> CuMatrix)
      push!(batch_B_gpu, B_cpu |> CuMatrix)
    end

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssBatchedSolver(batch_A_gpu, structure, 'U')

    cudss("analysis", solver, batch_X_gpu, batch_B_gpu)
    cudss("factorization", solver, batch_X_gpu, batch_B_gpu)
    cudss("solve", solver, batch_X_gpu, batch_B_gpu)

    info = cudss_get(solver, "info")
    @test info == 1

    for i = 1:nbatch
      batch_A_gpu[i] = batch_A_gpu[i] + 21 * I
      batch_A_cpu[i] = batch_A_cpu[i] + 21 * I
    end
    cudss_set(solver, batch_A_gpu)

    cudss("refactorization", solver, batch_X_gpu, batch_B_gpu)
    cudss("solve", solver, batch_X_gpu, batch_B_gpu)

    info = cudss_get(solver, "info")
    @test info == 0
  end
end

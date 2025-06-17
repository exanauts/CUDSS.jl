function cudss_version()
  @test CUDSS.version() >= v"0.6.0"
end

function cudss_dense()
  n = 20
  p = 4
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "CuVector" begin
      A_cpu = rand(T, n)
      A_gpu = CuVector(A_cpu)
      matrix = CudssMatrix(A_gpu)
      format = Ref{Cint}()
      CUDSS.cudssMatrixGetFormat(matrix, format)
      @test format[] == CUDSS.CUDSS_MFORMAT_DENSE

      A_cpu2 = rand(T, n)
      A_gpu2 = CuVector(A_cpu2)
      cudss_set(matrix, A_gpu2)
    end

    @testset "CuMatrix" begin
      A_cpu = rand(T, n, p)
      A_gpu = CuMatrix(A_cpu)
      matrix = CudssMatrix(A_gpu)
      format = Ref{Cint}()
      CUDSS.cudssMatrixGetFormat(matrix, format)
      @test format[] == CUDSS.CUDSS_MFORMAT_DENSE

      A_cpu2 = rand(T, n, p)
      A_gpu2 = CuMatrix(A_cpu2)
      cudss_set(matrix, A_gpu2)
    end
  end
end

function cudss_sparse()
  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu = sprand(T, n, n, 1.0)
    A_cpu = A_cpu + A_cpu'
    A_gpu = CuSparseMatrixCSR(A_cpu)
    @testset "view = $view" for view in ('L', 'U', 'F')
      @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
        matrix = CudssMatrix(A_gpu, structure, view) 
        format = Ref{Cint}()
        CUDSS.cudssMatrixGetFormat(matrix, format)
        @test format[] == CUDSS.CUDSS_MFORMAT_CSR

        A_cpu2 = sprand(T, n, n, 1.0)
        A_cpu2 = A_cpu2 + A_cpu2'
        A_gpu2 = CuSparseMatrixCSR(A_cpu2)
        cudss_set(matrix, A_gpu2)
      end
    end
  end
end

function cudss_solver()
  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A_cpu = sprand(T, n, n, 1.0)
    A_cpu = A_cpu * A_cpu' + I
    A_gpu = CuSparseMatrixCSR(A_cpu)
    @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
      @testset "view = $view" for view in ('L', 'U', 'F')
        solver = CudssSolver(A_gpu, structure, view)

        x_cpu = zeros(T, n)
        x_gpu = CuVector(x_cpu)
        b_cpu = rand(T, n)
        b_gpu = CuVector(b_cpu)
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
            if parameter != "host_nthreads"
              val = cudss_get(solver, parameter)
            end
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
            (parameter == "host_nthreads") && cudss_set(solver, parameter, 0)
            for algo in ("default", "algo1", "algo2", "algo3")
              (parameter == "reordering_alg") && cudss_set(solver, parameter, algo)
              (parameter == "factorization_alg") && cudss_set(solver, parameter, algo)
              (parameter == "solve_alg") && cudss_set(solver, parameter, algo)
              (parameter == "pivot_epsilon_alg") && cudss_set(solver, parameter, algo)
            end
            for flag in (0, 1)
              (parameter == "hybrid_mode") && cudss_set(solver, parameter, flag)
              (parameter == "hybrid_execute_mode") && cudss_set(solver, parameter, flag)
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

function cudss_execution()
  n = 100
  p = 5
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "Unsymmetric -- Non-Hermitian" begin
      @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
        A_cpu = sprand(T, n, n, 0.02) + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)

        A_gpu = CuSparseMatrixCSR(A_cpu)
        x_gpu = CuVector(x_cpu)
        b_gpu = CuVector(b_cpu)

        matrix = CudssMatrix(A_gpu, "G", 'F')
        config = CudssConfig()
        data = CudssData()
        solver = CudssSolver(matrix, config, data)
        cudss_set(solver, "pivot_type", pivot)

        cudss("analysis", solver, x_gpu, b_gpu)
        cudss("factorization", solver, x_gpu, b_gpu)
        cudss("solve", solver, x_gpu, b_gpu)

        r_gpu = b_gpu - A_gpu * x_gpu
        @test norm(r_gpu) ≤ √eps(R)

        # In-place LU
        d_gpu = rand(T, n) |> CuVector
        A_gpu = A_gpu + Diagonal(d_gpu)
        cudss_set(solver, A_gpu)

        c_cpu = rand(T, n)
        c_gpu = CuVector(c_cpu)

        cudss("refactorization", solver, x_gpu, c_gpu)
        cudss("solve", solver, x_gpu, c_gpu)

        r_gpu = c_gpu - A_gpu * x_gpu
        @test norm(r_gpu) ≤ √eps(R)
      end
    end

    @testset "Symmetric -- Hermitian" begin
      @testset "view = $view" for view in ('F', 'L', 'U')
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
          A_cpu = sprand(T, n, n, 0.01) + I
          A_cpu = A_cpu + A_cpu'
          X_cpu = zeros(T, n, p)
          B_cpu = rand(T, n, p)

          (view == 'L') && (A_gpu = CuSparseMatrixCSR(A_cpu |> tril))
          (view == 'U') && (A_gpu = CuSparseMatrixCSR(A_cpu |> triu))
          (view == 'F') && (A_gpu = CuSparseMatrixCSR(A_cpu))
          X_gpu = CuMatrix(X_cpu)
          B_gpu = CuMatrix(B_cpu)

          structure = T <: Real ? "S" : "H"
          matrix = CudssMatrix(A_gpu, structure, view)
          config = CudssConfig()
          data = CudssData()
          solver = CudssSolver(matrix, config, data)
          cudss_set(solver, "pivot_type", pivot)

          cudss("analysis", solver, X_gpu, B_gpu)
          cudss("factorization", solver, X_gpu, B_gpu)
          cudss("solve", solver, X_gpu, B_gpu)

          R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)

          # In-place LDLᵀ / LDLᴴ
          d_gpu = rand(R, n) |> CuVector
          A_gpu = A_gpu + Diagonal(d_gpu)
          cudss_set(solver, A_gpu)

          C_cpu = rand(T, n, p)
          C_gpu = CuMatrix(C_cpu)

          cudss("refactorization", solver, X_gpu, C_gpu)
          cudss("solve", solver, X_gpu, C_gpu)

          R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)
        end
      end
    end

    @testset "SPD -- HPD" begin
      @testset "view = $view" for view in ('F', 'L', 'U')
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
          A_cpu = sprand(T, n, n, 0.01)
          A_cpu = A_cpu * A_cpu' + I
          X_cpu = zeros(T, n, p)
          B_cpu = rand(T, n, p)

          (view == 'L') && (A_gpu = CuSparseMatrixCSR(A_cpu |> tril))
          (view == 'U') && (A_gpu = CuSparseMatrixCSR(A_cpu |> triu))
          (view == 'F') && (A_gpu = CuSparseMatrixCSR(A_cpu))
          X_gpu = CuMatrix(X_cpu)
          B_gpu = CuMatrix(B_cpu)

          structure = T <: Real ? "SPD" : "HPD"
          matrix = CudssMatrix(A_gpu, structure, view)
          config = CudssConfig()
          data = CudssData()
          solver = CudssSolver(matrix, config, data)
          cudss_set(solver, "pivot_type", pivot)

          cudss("analysis", solver, X_gpu, B_gpu)
          cudss("factorization", solver, X_gpu, B_gpu)
          cudss("solve", solver, X_gpu, B_gpu)

          R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)

          # In-place LLᵀ / LLᴴ
          d_gpu = rand(R, n) |> CuVector
          A_gpu = A_gpu + Diagonal(d_gpu)
          cudss_set(solver, A_gpu)

          C_cpu = rand(T, n, p)
          C_gpu = CuMatrix(C_cpu)

          cudss("refactorization", solver, X_gpu, C_gpu)
          cudss("solve", solver, X_gpu, C_gpu)

          R_gpu = C_gpu - ( CuSparseMatrixCSR(A_cpu) + Diagonal(d_gpu) ) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)
        end
      end
    end
  end
end

function cudss_generic()
  n = 100
  p = 5
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "Unsymmetric -- Non-Hermitian" begin
      A_cpu = sprand(T, n, n, 0.02) + I
      b_cpu = rand(T, n)

      A_gpu = CuSparseMatrixCSR(A_cpu)
      b_gpu = CuVector(b_cpu)

      @testset "ldiv!" begin
        x_cpu = zeros(T, n)
        x_gpu = CuVector(x_cpu)

        solver = lu(A_gpu)
        ldiv!(x_gpu, solver, b_gpu)
        r_gpu = b_gpu - A_gpu * x_gpu
        @test norm(r_gpu) ≤ √eps(R)

        A_gpu2 = rand(T) * A_gpu
        lu!(solver, A_gpu2)
        x_gpu .= b_gpu
        ldiv!(solver, x_gpu)
        r_gpu2 = b_gpu - A_gpu2 * x_gpu
        @test norm(r_gpu2) ≤ √eps(R)
      end

      @testset "\\" begin
        solver = lu(A_gpu)
        x_gpu = solver \ b_gpu
        r_gpu = b_gpu - A_gpu * x_gpu
        @test norm(r_gpu) ≤ √eps(R)

        A_gpu2 = rand(T) * A_gpu
        lu!(solver, A_gpu2)
        x_gpu = solver \ b_gpu
        r_gpu2 = b_gpu - A_gpu2 * x_gpu
        @test norm(r_gpu2) ≤ √eps(R)
      end
    end

    @testset "Symmetric -- Hermitian" begin
      @testset "view = $view" for view in ('F', 'L', 'U')
        A_cpu = sprand(T, n, n, 0.01) + I
        A_cpu = A_cpu + A_cpu'
        B_cpu = rand(T, n, p)

        (view == 'L') && (A_gpu = CuSparseMatrixCSR(A_cpu |> tril))
        (view == 'U') && (A_gpu = CuSparseMatrixCSR(A_cpu |> triu))
        (view == 'F') && (A_gpu = CuSparseMatrixCSR(A_cpu))
        B_gpu = CuMatrix(B_cpu)

        @testset "ldiv!" begin
          X_cpu = zeros(T, n, p)
          X_gpu = CuMatrix(X_cpu)

          solver = ldlt(A_gpu; view)
          ldiv!(X_gpu, solver, B_gpu)
          R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)

          c = rand(R)
          A_cpu2 = c * A_cpu
          A_gpu2 = c * A_gpu

          ldlt!(solver, A_gpu2)
          X_gpu .= B_gpu
          ldiv!(solver, X_gpu)
          R_gpu2 = B_gpu - CuSparseMatrixCSR(A_cpu2) * X_gpu
          @test norm(R_gpu2) ≤ √eps(R)
        end

        @testset "\\" begin
          solver = ldlt(A_gpu; view)
          X_gpu = solver \ B_gpu
          R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)

          c = rand(R)
          A_cpu2 = c * A_cpu
          A_gpu2 = c * A_gpu

          ldlt!(solver, A_gpu2)
          X_gpu = solver \ B_gpu
          R_gpu2 = B_gpu - CuSparseMatrixCSR(A_cpu2) * X_gpu
          @test norm(R_gpu2) ≤ √eps(R)
        end
      end
    end

    @testset "SPD -- HPD" begin
      @testset "view = $view" for view in ('F', 'L', 'U')
        A_cpu = sprand(T, n, n, 0.01)
        A_cpu = A_cpu * A_cpu' + I
        B_cpu = rand(T, n, p)

        (view == 'L') && (A_gpu = CuSparseMatrixCSR(A_cpu |> tril))
        (view == 'U') && (A_gpu = CuSparseMatrixCSR(A_cpu |> triu))
        (view == 'F') && (A_gpu = CuSparseMatrixCSR(A_cpu))
        B_gpu = CuMatrix(B_cpu)

        @testset "ldiv!" begin
          X_cpu = zeros(T, n, p)
          X_gpu = CuMatrix(X_cpu)

          solver = cholesky(A_gpu; view)
          ldiv!(X_gpu, solver, B_gpu)
          R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)

          c = rand(R)
          A_cpu2 = c * A_cpu
          A_gpu2 = c * A_gpu

          cholesky!(solver, A_gpu2)
          X_gpu .= B_gpu
          ldiv!(solver, X_gpu)
          R_gpu2 = B_gpu - CuSparseMatrixCSR(A_cpu2) * X_gpu
          @test norm(R_gpu2) ≤ √eps(R)
        end

        @testset "\\" begin
          solver = cholesky(A_gpu; view)
          X_gpu = solver \ B_gpu
          R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
          @test norm(R_gpu) ≤ √eps(R)

          c = rand(R)
          A_cpu2 = c * A_cpu
          A_gpu2 = c * A_gpu

          cholesky!(solver, A_gpu2)
          X_gpu = solver \ B_gpu
          R_gpu2 = B_gpu - CuSparseMatrixCSR(A_cpu2) * X_gpu
          @test norm(R_gpu2) ≤ √eps(R)
        end
      end
    end
  end
end

function user_permutation()
  function permutation_lu(T, A_cpu, x_cpu, b_cpu, permutation)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    solver = CudssSolver(A_gpu, "G", 'F')

    cudss_set(solver, "user_perm", permutation)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    nz = cudss_get(solver, "lu_nnz")
    return nz
  end

  function permutation_ldlt(T, A_cpu, x_cpu, b_cpu, permutation, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "user_perm", permutation)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    nz = cudss_get(solver, "lu_nnz")
    return nz
  end

  function permutation_llt(T, A_cpu, x_cpu, b_cpu, permutation, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "user_perm", permutation)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    nz = cudss_get(solver, "lu_nnz")
    return nz
  end

  n = 1000
  perm1_cpu = Vector{Cint}(undef, n)
  perm2_cpu = Vector{Cint}(undef, n)
  for i = 1:n
    perm1_cpu[i] = i
    perm2_cpu[i] = n-i+1
  end
  perm1_gpu = CuVector{Cint}(perm1_cpu)
  perm2_gpu = CuVector{Cint}(perm2_cpu)
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "LU" begin
      A_cpu = sprand(T, n, n, 0.05) + I
      x_cpu = zeros(T, n)
      b_cpu = rand(T, n)
      nz1_cpu = permutation_lu(T, A_cpu, x_cpu, b_cpu, perm1_cpu)
      nz2_cpu = permutation_lu(T, A_cpu, x_cpu, b_cpu, perm2_cpu)
      nz1_gpu = permutation_lu(T, A_cpu, x_cpu, b_cpu, perm1_gpu)
      nz2_gpu = permutation_lu(T, A_cpu, x_cpu, b_cpu, perm2_gpu)
      @test nz1_cpu == nz1_gpu
      @test nz2_cpu == nz2_gpu
      @test nz1_cpu != nz2_cpu
    end
    @testset "LDLᵀ / LDLᴴ" begin
      A_cpu = sprand(T, n, n, 0.05) + I
      A_cpu = A_cpu + A_cpu'
      x_cpu = zeros(T, n)
      b_cpu = rand(T, n)
      @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
        nz1_cpu = permutation_ldlt(T, A_cpu, x_cpu, b_cpu, perm1_cpu, uplo)
        nz2_cpu = permutation_ldlt(T, A_cpu, x_cpu, b_cpu, perm2_cpu, uplo)
        nz1_gpu = permutation_ldlt(T, A_cpu, x_cpu, b_cpu, perm1_gpu, uplo)
        nz2_gpu = permutation_ldlt(T, A_cpu, x_cpu, b_cpu, perm2_gpu, uplo)
        @test nz1_cpu == nz1_gpu
        @test nz2_cpu == nz2_gpu
        @test nz1_cpu != nz2_cpu
      end
    end
    @testset "LLᵀ / LLᴴ" begin
      A_cpu = sprand(T, n, n, 0.01)
      A_cpu = A_cpu * A_cpu' + I
      x_cpu = zeros(T, n)
      b_cpu = rand(T, n)
      @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
        nz1_cpu = permutation_llt(T, A_cpu, x_cpu, b_cpu, perm1_cpu, uplo)
        nz2_cpu = permutation_llt(T, A_cpu, x_cpu, b_cpu, perm2_cpu, uplo)
        nz1_gpu = permutation_llt(T, A_cpu, x_cpu, b_cpu, perm1_gpu, uplo)
        nz2_gpu = permutation_llt(T, A_cpu, x_cpu, b_cpu, perm2_gpu, uplo)
        @test nz1_cpu == nz1_gpu
        @test nz2_cpu == nz2_gpu
        @test nz1_cpu != nz2_cpu
      end
    end
  end
end

function iterative_refinement()
  function ir_lu(T, A_cpu, x_cpu, b_cpu, ir)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    solver = CudssSolver(A_gpu, "G", 'F')
    cudss_set(solver, "ir_n_steps", ir)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    return norm(r_gpu)
  end

  function ir_ldlt(T, A_cpu, x_cpu, b_cpu, ir, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "ir_n_steps", ir)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  function ir_llt(T, A_cpu, x_cpu, b_cpu, ir, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "ir_n_steps", ir)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  n = 100
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "number of iterative refinement: $ir" for ir in (1, 2)
      @testset "LU" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        res = ir_lu(T, A_cpu, x_cpu, b_cpu, ir)
        @test res ≤ √eps(R)
      end
      @testset "LDLᵀ / LDLᴴ" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        A_cpu = A_cpu + A_cpu'
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
          res = ir_ldlt(T, A_cpu, x_cpu, b_cpu, ir, uplo)
          @test res ≤ √eps(R)
        end
      end
      @testset "LLᵀ / LLᴴ" begin
        A_cpu = sprand(T, n, n, 0.01)
        A_cpu = A_cpu * A_cpu' + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
          res = ir_llt(T, A_cpu, x_cpu, b_cpu, ir, uplo)
          @test res ≤ √eps(R)
        end
      end
    end
  end
end

function small_matrices()
  function cudss_lu(T, A_cpu, x_cpu, b_cpu)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    solver = CudssSolver(A_gpu, "G", 'F')

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    return norm(r_gpu)
  end

  function cudss_ldlt(T, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  function cudss_llt(T, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "Size of the linear system: $n" for n in 1:16
      @testset "LU" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        res = cudss_lu(T, A_cpu, x_cpu, b_cpu)
        @test res ≤ √eps(R)
      end
      @testset "LDLᵀ / LDLᴴ" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        A_cpu = A_cpu + A_cpu'
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
          res = cudss_ldlt(T, A_cpu, x_cpu, b_cpu, uplo)
          @test res ≤ √eps(R)
        end
      end
      @testset "LLᵀ / LLᴴ" begin
        A_cpu = sprand(T, n, n, 0.01)
        A_cpu = A_cpu * A_cpu' + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
          res = cudss_llt(T, A_cpu, x_cpu, b_cpu, uplo)
          @test res ≤ √eps(R)
        end
      end
    end
  end
end

function hybrid_mode()
  function hybrid_lu(T, A_cpu, x_cpu, b_cpu)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    solver = CudssSolver(A_gpu, "G", 'F')
    cudss_set(solver, "hybrid_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    return norm(r_gpu)
  end

  function hybrid_ldlt(T, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "hybrid_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  function hybrid_llt(T, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "hybrid_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    @testset "LU" begin
      A_cpu = sprand(T, n, n, 0.05) + I
      x_cpu = zeros(T, n)
      b_cpu = rand(T, n)
      res = hybrid_lu(T, A_cpu, x_cpu, b_cpu)
      @test res ≤ √eps(R)
    end
    @testset "LDLᵀ / LDLᴴ" begin
      A_cpu = sprand(T, n, n, 0.05) + I
      A_cpu = A_cpu + A_cpu'
      x_cpu = zeros(T, n)
      b_cpu = rand(T, n)
      @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
        res = hybrid_ldlt(T, A_cpu, x_cpu, b_cpu, uplo)
        @test res ≤ √eps(R)
      end
    end
    @testset "LLᵀ / LLᴴ" begin
      A_cpu = sprand(T, n, n, 0.01)
      A_cpu = A_cpu * A_cpu' + I
      x_cpu = zeros(T, n)
      b_cpu = rand(T, n)
      @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
        res = hybrid_llt(T, A_cpu, x_cpu, b_cpu, uplo)
        @test res ≤ √eps(R)
      end
    end
  end
end

function refactorization_cholesky()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    n = 100
    p = 5
    A_cpu = sprand(T, n, n, 0.01)
    A_cpu = A_cpu * A_cpu' - 20 * I
    X_cpu = zeros(T, n, p)
    B_cpu = rand(T, n, p)

    A_gpu = CuSparseMatrixCSR(A_cpu |> triu)
    X_gpu = CuMatrix(X_cpu)
    B_gpu = CuMatrix(B_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, 'U')

    cudss("analysis", solver, X_gpu, B_gpu)
    cudss("factorization", solver, X_gpu, B_gpu)
    cudss("solve", solver, X_gpu, B_gpu)

    info = cudss_get(solver, "info")
    @test info == 1

    A_gpu = A_gpu + 21 * I
    cudss_set(solver, A_gpu)

    cudss("refactorization", solver, X_gpu, B_gpu)
    cudss("solve", solver, X_gpu, B_gpu)

    info = cudss_get(solver, "info")
    @test info == 0
  end
end

function cudss_version()
  @test CUDSS.version() >= v"0.7.0"
end

function cudss_dense()
  n = 20
  p = 4
  @testset "cudss_update -- precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "CuVector" begin
      A_cpu = rand(T, n)
      A_gpu = CuVector(A_cpu)
      matrix = CudssMatrix(A_gpu)
      format = Ref{Cint}()
      CUDSS.cudssMatrixGetFormat(matrix, format)
      @test format[] == CUDSS.CUDSS_MFORMAT_DENSE |> Cint

      A_cpu2 = rand(T, n)
      A_gpu2 = CuVector(A_cpu2)
      cudss_update(matrix, A_gpu2)
    end

    @testset "CuMatrix" begin
      A_cpu = rand(T, n, p)
      A_gpu = CuMatrix(A_cpu)
      matrix = CudssMatrix(A_gpu)
      format = Ref{Cint}()
      CUDSS.cudssMatrixGetFormat(matrix, format)
      @test format[] == CUDSS.CUDSS_MFORMAT_DENSE |> Cint

      A_cpu2 = rand(T, n, p)
      A_gpu2 = CuMatrix(A_cpu2)
      cudss_update(matrix, A_gpu2)
    end
  end
end

function cudss_sparse()
  n = 20
  @testset "cudss_update -- precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      A_cpu = sprand(T, n, n, 1.0)
      A_cpu = A_cpu + A_cpu'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
      @testset "view = $view" for view in ('L', 'U', 'F')
        @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
          matrix = CudssMatrix(A_gpu, structure, view)
          format = Ref{Cint}()
          CUDSS.cudssMatrixGetFormat(matrix, format)
          @test format[] == CUDSS.CUDSS_MFORMAT_CSR |> Cint

          A_cpu2 = sprand(T, n, n, 1.0)
          A_cpu2 = A_cpu2 + A_cpu2'
          A_gpu2 = CuSparseMatrixCSR{T,INT}(A_cpu2)
          cudss_update(matrix, A_gpu2)
        end
      end
    end
  end
end

function cudss_solver()
  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      A_cpu = sprand(T, n, n, 1.0)
      A_cpu = A_cpu * A_cpu' + I
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
      @testset "structure = $structure" for structure in ("G", "S", "H", "SPD", "HPD")
        @testset "view = $view" for view in ('L', 'U', 'F')
          solver = CudssSolver(A_gpu, structure, view)
          cudss_set(solver, "use_matching", 1)  # neeeded for "perm_matching" / "scale_row" / "scale_col"

          x_cpu = zeros(T, n)
          x_gpu = CuVector(x_cpu)
          b_cpu = rand(T, n)
          b_gpu = CuVector(b_cpu)
          cudss("analysis", solver, x_gpu, b_gpu)
          cudss("factorization", solver, x_gpu, b_gpu)

          memory_estimates = Vector{Int64}(undef, 16)
          buffer_int = Vector{INT}(undef, n)
          buffer_R = Vector{R}(undef, n)
          buffer_T = Vector{T}(undef, n)

          @testset "data parameter = $parameter" for parameter in CUDSS_DATA_PARAMETERS
            parameter ∈ ("comm", "user_schur_indices", "schur_shape", "schur_matrix", "user_elimination_tree", "elimination_tree", "user_host_interrupt") && continue
            @testset "cudss_set" begin
              (parameter == "nsuperpanels") && continue
              if parameter == "user_perm"
                perm_cpu = INT[i for i=n:-1:1]
                cudss_set(solver, parameter, perm_cpu)
                perm_gpu = CuVector{INT}(perm_cpu)
                cudss_set(solver, parameter, perm_gpu)
              end
              if parameter ∈ ("perm_row", "perm_col", "perm_reorder_row", "perm_reorder_col", "perm_matching")
                cudss_set(solver, parameter, buffer_int)
              end
              if parameter ∈ ("scale_row", "scale_col")
                cudss_set(solver, parameter, buffer_R)
              end
              (parameter == "diag") && cudss_set(solver, parameter, buffer_T)
              (parameter == "memory_estimates") && cudss_set(solver, parameter, memory_estimates)
              (parameter == "info") && cudss_set(solver, parameter, 1)
            end
            @testset "cudss_get" begin
              parameter ∈ ("comm", "user_perm", "perm_row", "perm_col") && continue
              (parameter == "inertia") && !(structure ∈ ("S", "H")) && continue
              val = cudss_get(solver, parameter)
            end
          end

          @testset "config parameter = $parameter" for parameter in CUDSS_CONFIG_PARAMETERS
            parameter ∈ ("device_indices", "nd_nlevels", "ubatch_size", "ubatch_index") && continue
            @testset "cudss_get" begin
              if parameter != "host_nthreads"
                val = cudss_get(solver, parameter)
              end
            end
            @testset "cudss_set" begin
              (parameter == "device_count") && cudss_set(solver, parameter, 1)
              (parameter == "solve_mode") && cudss_set(solver, parameter, 0)
              (parameter == "ir_n_steps") && cudss_set(solver, parameter, 1)
              (parameter == "ir_tol") && cudss_set(solver, parameter, 1e-8)
              (parameter == "pivot_threshold") && cudss_set(solver, parameter, 2.0)
              (parameter == "pivot_epsilon") && cudss_set(solver, parameter, 1e-12)
              (parameter == "max_lu_nnz") && cudss_set(solver, parameter, 10)
              (parameter == "hybrid_device_memory_limit") && cudss_set(solver, parameter, 2048)
              (parameter == "host_nthreads") && cudss_set(solver, parameter, 0)
              for algo in ("default", "algo1", "algo2", "algo3", "algo4", "algo5")
                (parameter == "matching_alg") && cudss_set(solver, parameter, algo)
                (parameter == "reordering_alg") && cudss_set(solver, parameter, algo)
                (parameter == "factorization_alg") && cudss_set(solver, parameter, algo)
                (parameter == "solve_alg") && cudss_set(solver, parameter, algo)
                (parameter == "pivot_epsilon_alg") && cudss_set(solver, parameter, algo)
              end
              for flag in (0, 1)
                (parameter == "schur_mode") && cudss_set(solver, parameter, flag)
                (parameter == "deterministic_mode") && cudss_set(solver, parameter, flag)
                (parameter == "hybrid_memory_mode") && cudss_set(solver, parameter, flag)
                (parameter == "hybrid_execute_mode") && cudss_set(solver, parameter, flag)
                (parameter == "use_superpanels") && cudss_set(solver, parameter, flag)
                (parameter == "use_matching") && cudss_set(solver, parameter, flag)
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
end

function cudss_execution()
  n = 100
  p = 5
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      @testset "Unsymmetric -- Non-Hermitian" begin
        @testset "Pivoting = $pivot" for pivot in ('C', 'R', 'N')
          A_cpu = sprand(T, n, n, 0.02) + I
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)

          A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
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
          cudss_update(solver, A_gpu)

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

            (view == 'L') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril))
            (view == 'U') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu))
            (view == 'F') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu))
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

            R_gpu = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)

            # In-place LDLᵀ / LDLᴴ
            d_gpu = rand(R, n) |> CuVector
            A_gpu = A_gpu + Diagonal(d_gpu)
            cudss_update(solver, A_gpu)

            C_cpu = rand(T, n, p)
            C_gpu = CuMatrix(C_cpu)

            cudss("refactorization", solver, X_gpu, C_gpu)
            cudss("solve", solver, X_gpu, C_gpu)

            R_gpu = C_gpu - ( CuSparseMatrixCSR{T,INT}(A_cpu) + Diagonal(d_gpu) ) * X_gpu
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

            (view == 'L') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril))
            (view == 'U') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu))
            (view == 'F') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu))
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

            R_gpu = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)

            # In-place LLᵀ / LLᴴ
            d_gpu = rand(R, n) |> CuVector
            A_gpu = A_gpu + Diagonal(d_gpu)
            cudss_update(solver, A_gpu)

            C_cpu = rand(T, n, p)
            C_gpu = CuMatrix(C_cpu)

            cudss("refactorization", solver, X_gpu, C_gpu)
            cudss("solve", solver, X_gpu, C_gpu)

            R_gpu = C_gpu - ( CuSparseMatrixCSR{T,INT}(A_cpu) + Diagonal(d_gpu) ) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)
          end
        end
      end
    end
  end
end

function cudss_generic()
  n = 100
  p = 5
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      @testset "Unsymmetric -- Non-Hermitian" begin
        A_cpu = sprand(T, n, n, 0.02) + I
        b_cpu = rand(T, n)
        x_cpu = zeros(T, n)

        A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
        b_gpu = CuVector(b_cpu)

        @testset "lu!" begin
          solver = CudssSolver(A_gpu, "G", 'F')
          x_gpu = CuVector(x_cpu)
          cudss("analysis", solver, x_gpu, b_gpu)
          solver = lu!(solver, A_gpu)
          @test !solver.fresh_factorization
        end

        @testset "ldiv!" begin
          solver = lu(A_gpu)
          x_gpu = CuVector(x_cpu)
          ldiv!(x_gpu, solver, b_gpu)
          r_gpu = b_gpu - A_gpu * x_gpu
          @test norm(r_gpu) ≤ √eps(R)

          A_gpu2 = rand(T) * A_gpu
          lu!(solver, A_gpu2)
          x_gpu .= b_gpu
          ldiv!(solver, x_gpu)
          r_gpu2 = b_gpu - A_gpu2 * x_gpu
          @test norm(r_gpu2) ≤ √eps(R)

          A_gpu3 = rand(T) * A_gpu
          lu!(solver, A_gpu3)
          x_gpu .= b_gpu
          cudss_x_gpu = CudssMatrix(x_gpu)
          ldiv!(solver, cudss_x_gpu)
          r_gpu3 = b_gpu - A_gpu3 * x_gpu
          @test norm(r_gpu3) ≤ √eps(R)

          A_gpu4 = rand(T) * A_gpu
          lu!(solver, A_gpu4)
          cudss_x_gpu = CudssMatrix(x_gpu)
          cudss_b_gpu = CudssMatrix(b_gpu)
          ldiv!(cudss_x_gpu, solver, cudss_b_gpu)
          r_gpu4 = b_gpu - A_gpu4 * x_gpu
          @test norm(r_gpu4) ≤ √eps(R)
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
          X_cpu = rand(T, n, p)

          (view == 'L') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril))
          (view == 'U') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu))
          (view == 'F') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu))
          B_gpu = CuMatrix(B_cpu)

          @testset "ldlt!" begin
            structure = T <: Real ? "S" : "H"
            solver = CudssSolver(A_gpu, structure, view)
            X_gpu = CuMatrix(X_cpu)
            cudss("analysis", solver, X_gpu, B_gpu)
            solver = ldlt!(solver, A_gpu)
            @test !solver.fresh_factorization
          end

          @testset "ldiv!" begin
            solver = ldlt(A_gpu; view)
            X_gpu = CuMatrix(X_cpu)
            ldiv!(X_gpu, solver, B_gpu)
            R_gpu = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)

            c = rand(R)
            A_cpu2 = c * A_cpu
            A_gpu2 = c * A_gpu
            ldlt!(solver, A_gpu2)
            X_gpu .= B_gpu
            ldiv!(solver, X_gpu)
            R_gpu2 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu2) * X_gpu
            @test norm(R_gpu2) ≤ √eps(R)

            c = rand(R)
            A_cpu3 = c * A_cpu
            A_gpu3 = c * A_gpu
            ldlt!(solver, A_gpu3)
            X_gpu .= B_gpu
            cudss_X_gpu = CudssMatrix(X_gpu)
            ldiv!(solver, cudss_X_gpu)
            R_gpu3 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu3) * X_gpu
            @test norm(R_gpu3) ≤ √eps(R)

            c = rand(R)
            A_cpu4 = c * A_cpu
            A_gpu4 = c * A_gpu
            ldlt!(solver, A_gpu4)
            cudss_X_gpu = CudssMatrix(X_gpu)
            cudss_B_gpu = CudssMatrix(B_gpu)
            ldiv!(cudss_X_gpu, solver, cudss_B_gpu)
            R_gpu4 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu4) * X_gpu
            @test norm(R_gpu4) ≤ √eps(R)
          end

          @testset "\\" begin
            solver = ldlt(A_gpu; view)
            X_gpu = solver \ B_gpu
            R_gpu = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)

            c = rand(R)
            A_cpu2 = c * A_cpu
            A_gpu2 = c * A_gpu

            ldlt!(solver, A_gpu2)
            X_gpu = solver \ B_gpu
            R_gpu2 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu2) * X_gpu
            @test norm(R_gpu2) ≤ √eps(R)
          end
        end
      end

      @testset "SPD -- HPD" begin
        @testset "view = $view" for view in ('F', 'L', 'U')
          A_cpu = sprand(T, n, n, 0.01)
          A_cpu = A_cpu * A_cpu' + I
          B_cpu = rand(T, n, p)
          X_cpu = zeros(T, n, p)

          (view == 'L') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril))
          (view == 'U') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu))
          (view == 'F') && (A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu))
          B_gpu = CuMatrix(B_cpu)

          @testset "cholesky!" begin
            structure = T <: Real ? "SPD" : "HPD"
            solver = CudssSolver(A_gpu, structure, view)
            X_gpu = CuMatrix(X_cpu)
            cudss("analysis", solver, X_gpu, B_gpu)
            solver = cholesky!(solver, A_gpu)
            @test !solver.fresh_factorization
          end

          @testset "ldiv!" begin
            solver = cholesky(A_gpu; view)
            X_gpu = CuMatrix(X_cpu)
            ldiv!(X_gpu, solver, B_gpu)
            R_gpu = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)

            c = rand(R)
            A_cpu2 = c * A_cpu
            A_gpu2 = c * A_gpu
            cholesky!(solver, A_gpu2)
            X_gpu .= B_gpu
            ldiv!(solver, X_gpu)
            R_gpu2 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu2) * X_gpu
            @test norm(R_gpu2) ≤ √eps(R)

            c = rand(R)
            A_cpu3 = c * A_cpu
            A_gpu3 = c * A_gpu
            cholesky!(solver, A_gpu3)
            X_gpu .= B_gpu
            cudss_X_gpu = CudssMatrix(X_gpu)
            ldiv!(solver, cudss_X_gpu)
            R_gpu3 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu3) * X_gpu
            @test norm(R_gpu3) ≤ √eps(R)

            c = rand(R)
            A_cpu4 = c * A_cpu
            A_gpu4 = c * A_gpu
            cholesky!(solver, A_gpu4)
            cudss_X_gpu = CudssMatrix(X_gpu)
            cudss_B_gpu = CudssMatrix(B_gpu)
            ldiv!(cudss_X_gpu, solver, cudss_B_gpu)
            R_gpu4 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu4) * X_gpu
            @test norm(R_gpu4) ≤ √eps(R)
          end

          @testset "\\" begin
            solver = cholesky(A_gpu; view)
            X_gpu = solver \ B_gpu
            R_gpu = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * X_gpu
            @test norm(R_gpu) ≤ √eps(R)

            c = rand(R)
            A_cpu2 = c * A_cpu
            A_gpu2 = c * A_gpu

            cholesky!(solver, A_gpu2)
            X_gpu = solver \ B_gpu
            R_gpu2 = B_gpu - CuSparseMatrixCSR{T,INT}(A_cpu2) * X_gpu
            @test norm(R_gpu2) ≤ √eps(R)
          end
        end
      end
    end
  end
end

function user_permutation()
  function permutation_lu(T, INT, A_cpu, x_cpu, b_cpu, permutation)
    A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
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

  function permutation_ldlt(T, INT, A_cpu, x_cpu, b_cpu, permutation, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
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

  function permutation_llt(T, INT, A_cpu, x_cpu, b_cpu, permutation, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
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

  @testset "integer = $INT" for INT in (Cint,) # Int64)
    n = 1000
    perm1_cpu = Vector{INT}(undef, n)
    perm2_cpu = Vector{INT}(undef, n)
    for i = 1:n
      perm1_cpu[i] = i
      perm2_cpu[i] = n-i+1
    end
    perm1_gpu = CuVector{INT}(perm1_cpu)
    perm2_gpu = CuVector{INT}(perm2_cpu)

    @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
      @testset "LU" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        nz1_cpu = permutation_lu(T, INT, A_cpu, x_cpu, b_cpu, perm1_cpu)
        nz2_cpu = permutation_lu(T, INT, A_cpu, x_cpu, b_cpu, perm2_cpu)
        nz1_gpu = permutation_lu(T, INT, A_cpu, x_cpu, b_cpu, perm1_gpu)
        nz2_gpu = permutation_lu(T, INT, A_cpu, x_cpu, b_cpu, perm2_gpu)
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
          nz1_cpu = permutation_ldlt(T, INT, A_cpu, x_cpu, b_cpu, perm1_cpu, uplo)
          nz2_cpu = permutation_ldlt(T, INT, A_cpu, x_cpu, b_cpu, perm2_cpu, uplo)
          nz1_gpu = permutation_ldlt(T, INT, A_cpu, x_cpu, b_cpu, perm1_gpu, uplo)
          nz2_gpu = permutation_ldlt(T, INT, A_cpu, x_cpu, b_cpu, perm2_gpu, uplo)
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
          nz1_cpu = permutation_llt(T, INT, A_cpu, x_cpu, b_cpu, perm1_cpu, uplo)
          nz2_cpu = permutation_llt(T, INT, A_cpu, x_cpu, b_cpu, perm2_cpu, uplo)
          nz1_gpu = permutation_llt(T, INT, A_cpu, x_cpu, b_cpu, perm1_gpu, uplo)
          nz2_gpu = permutation_llt(T, INT, A_cpu, x_cpu, b_cpu, perm2_gpu, uplo)
          @test nz1_cpu == nz1_gpu
          @test nz2_cpu == nz2_gpu
          @test nz1_cpu != nz2_cpu
        end
      end
    end
  end
end

function iterative_refinement()
  function ir_lu(T, INT, A_cpu, x_cpu, b_cpu, ir)
    A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
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

  function ir_ldlt(T, INT, A_cpu, x_cpu, b_cpu, ir, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "ir_n_steps", ir)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  function ir_llt(T, INT, A_cpu, x_cpu, b_cpu, ir, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "ir_n_steps", ir)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  n = 100
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      @testset "number of iterative refinement: $ir" for ir in (1, 2)
        @testset "LU" begin
          A_cpu = sprand(T, n, n, 0.05) + I
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)
          res = ir_lu(T, INT, A_cpu, x_cpu, b_cpu, ir)
          @test res ≤ √eps(R)
        end
        @testset "LDLᵀ / LDLᴴ" begin
          A_cpu = sprand(T, n, n, 0.05) + I
          A_cpu = A_cpu + A_cpu'
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)
          @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
            res = ir_ldlt(T, INT, A_cpu, x_cpu, b_cpu, ir, uplo)
            @test res ≤ √eps(R)
          end
        end
        @testset "LLᵀ / LLᴴ" begin
          A_cpu = sprand(T, n, n, 0.01)
          A_cpu = A_cpu * A_cpu' + I
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)
          @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
            res = ir_llt(T, INT, A_cpu, x_cpu, b_cpu, ir, uplo)
            @test res ≤ √eps(R)
          end
        end
      end
    end
  end
end

function small_matrices()
  function cudss_lu(T, INT, A_cpu, x_cpu, b_cpu)
    A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    solver = CudssSolver(A_gpu, "G", 'F')

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    return norm(r_gpu)
  end

  function cudss_ldlt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  function cudss_llt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      @testset "Size of the linear system: $n" for n in 1:16
        @testset "LU" begin
          A_cpu = sprand(T, n, n, 0.05) + I
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)
          res = cudss_lu(T, INT, A_cpu, x_cpu, b_cpu)
          @test res ≤ √eps(R)
        end
        @testset "LDLᵀ / LDLᴴ" begin
          A_cpu = sprand(T, n, n, 0.05) + I
          A_cpu = A_cpu + A_cpu'
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)
          @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
            res = cudss_ldlt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
            @test res ≤ √eps(R)
          end
        end
        @testset "LLᵀ / LLᴴ" begin
          A_cpu = sprand(T, n, n, 0.01)
          A_cpu = A_cpu * A_cpu' + I
          x_cpu = zeros(T, n)
          b_cpu = rand(T, n)
          @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
            res = cudss_llt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
            @test res ≤ √eps(R)
          end
        end
      end
    end
  end
end

function hybrid_memory_mode()
  function hybrid_lu(T, INT, A_cpu, x_cpu, b_cpu)
    A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    solver = CudssSolver(A_gpu, "G", 'F')
    cudss_set(solver, "hybrid_memory_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    return norm(r_gpu)
  end

  function hybrid_ldlt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "S" : "H"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "hybrid_memory_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  function hybrid_llt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
    if uplo == 'L'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> tril)
    elseif uplo == 'U'
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
    else
      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
    end
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    structure = T <: Real ? "SPD" : "HPD"
    solver = CudssSolver(A_gpu, structure, uplo)
    cudss_set(solver, "hybrid_memory_mode", 1)

    cudss("analysis", solver, x_gpu, b_gpu)
    nbytes_gpu = cudss_get(solver, "hybrid_device_memory_min")
    cudss_set(solver, "hybrid_device_memory_limit", nbytes_gpu)

    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - CuSparseMatrixCSR{T,INT}(A_cpu) * x_gpu
    return norm(r_gpu)
  end

  n = 20
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      @testset "LU" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        res = hybrid_lu(T, INT, A_cpu, x_cpu, b_cpu)
        @test res ≤ √eps(R)
      end
      @testset "LDLᵀ / LDLᴴ" begin
        A_cpu = sprand(T, n, n, 0.05) + I
        A_cpu = A_cpu + A_cpu'
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
          res = hybrid_ldlt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
          if T == ComplexF64
            @test_broken res ≤ √eps(R)
          else
            @test res ≤ √eps(R)
          end
        end
      end
      @testset "LLᵀ / LLᴴ" begin
        A_cpu = sprand(T, n, n, 0.01)
        A_cpu = A_cpu * A_cpu' + I
        x_cpu = zeros(T, n)
        b_cpu = rand(T, n)
        @testset "uplo = $uplo" for uplo in ('L', 'U', 'F')
          res = hybrid_llt(T, INT, A_cpu, x_cpu, b_cpu, uplo)
          @test res ≤ √eps(R)
        end
      end
    end
  end
end

function refactorization_cholesky()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint,) # Int64)
      R = real(T)
      n = 100
      p = 5
      A_cpu = sprand(T, n, n, 0.01)
      A_cpu = A_cpu * A_cpu' - 20 * I
      X_cpu = zeros(T, n, p)
      B_cpu = rand(T, n, p)

      A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> triu)
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
      @assert A_gpu isa CuSparseMatrixCSR{T,INT}
      cudss_update(solver, A_gpu)

      cudss("refactorization", solver, X_gpu, B_gpu)
      cudss("solve", solver, X_gpu, B_gpu)

      info = cudss_get(solver, "info")
      @test info == 0
    end
  end
end

function cudss_task_concurrency()
  # Test that CUDSS.jl correctly handles task-based concurrency.
  # Each Julia task should get its own CUDSS handle and use its own CUDA stream.
  # This test verifies that multiple concurrent solvers produce correct results.
  #
  # Note: We use @async (not Threads.@spawn) because CUDA.jl creates one stream
  # per Task, not per Thread. This test works even with a single-threaded Julia.

  n = 100
  num_tasks = 4

  @testset "precision = $T" for T in (Float32, Float64)
    R = real(T)

    # Create independent linear systems for each task
    systems = map(1:num_tasks) do i
      A_cpu = sprand(T, n, n, 0.05)
      A_cpu = A_cpu * A_cpu' + I  # Make SPD
      b_cpu = rand(T, n)
      x_expected = A_cpu \ b_cpu
      (A_cpu, b_cpu, x_expected)
    end

    # Solve all systems concurrently using tasks
    results = Vector{Any}(undef, num_tasks)

    @sync begin
      for i in 1:num_tasks
        @async begin
          A_cpu, b_cpu, _ = systems[i]

          # Each task creates its own GPU arrays and solver
          A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
          x_gpu = CUDA.zeros(T, n)
          b_gpu = CuVector(b_cpu)

          structure = T <: Real ? "SPD" : "HPD"
          solver = CudssSolver(A_gpu, structure, 'L')

          cudss("analysis", solver, x_gpu, b_gpu)
          cudss("factorization", solver, x_gpu, b_gpu)
          cudss("solve", solver, x_gpu, b_gpu)

          # Store result
          results[i] = Array(x_gpu)
        end
      end
    end

    # Verify all results are correct
    for i in 1:num_tasks
      _, _, x_expected = systems[i]
      @test norm(results[i] - x_expected) / norm(x_expected) ≤ √eps(R)
    end
  end

  @testset "stream isolation" begin
    # Test that each task uses a different stream
    T = Float64

    A_cpu = sprand(T, n, n, 0.05)
    A_cpu = A_cpu * A_cpu' + I
    b_cpu = rand(T, n)

    streams_used = Vector{Any}(undef, 2)

    @sync begin
      for i in 1:2
        @async begin
          # Record the stream this task is using
          streams_used[i] = CUDA.stream()

          A_gpu = CuSparseMatrixCSR(A_cpu |> tril)
          x_gpu = CUDA.zeros(T, n)
          b_gpu = CuVector(b_cpu)

          solver = CudssSolver(A_gpu, "SPD", 'L')
          cudss("analysis", solver, x_gpu, b_gpu)
          cudss("factorization", solver, x_gpu, b_gpu)
          cudss("solve", solver, x_gpu, b_gpu)
        end
      end
    end

    # Verify that different tasks used different streams
    @test streams_used[1] !== streams_used[2]
  end
end

function cudss_version()
  @test CUDSS.version() == v"0.1.0"
end

function cudss_dense()
  n = 20
  p = 4
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "CuVector" begin
      A_cpu = rand(T, n)
      A_gpu = CuVector(A_cpu)
      matrix = CudssMatrix(A_gpu)
      format = Ref{CUDSS.cudssMatrixFormat_t}()
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
      format = Ref{CUDSS.cudssMatrixFormat_t}()
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
        format = Ref{CUDSS.cudssMatrixFormat_t}()
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
    A_cpu = A_cpu + A_cpu'
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

        @testset "config parameter = $parameter" for parameter in CUDSS_CONFIG_PARAMETERS
          val = cudss_get(solver, parameter)
          for val in (CUDSS_ALG_DEFAULT, CUDSS_ALG_1, CUDSS_ALG_2, CUDSS_ALG_3)
            (parameter == "reordering_alg") && cudss_set(solver, parameter, val)
            (parameter == "factorization_alg") && cudss_set(solver, parameter, val)
            (parameter == "solve_alg") && cudss_set(solver, parameter, val)
          end
          (parameter == "matching_type") && cudss_set(solver, parameter, 0)
          (parameter == "solve_mode") && cudss_set(solver, parameter, 0)
          (parameter == "ir_n_steps") && cudss_set(solver, parameter, 1)
          (parameter == "ir_tol") && cudss_set(solver, parameter, 1e-8)
          for val in ('C', 'R', 'N')
            (parameter == "pivot_type") && cudss_set(solver, parameter, val)
          end
          (parameter == "pivot_threshold") && cudss_set(solver, parameter, 2.0)
          (parameter == "pivot_epsilon") && cudss_set(solver, parameter, 1e-12)
          (parameter == "max_lu_nnz") && cudss_set(solver, parameter, 10)
        end

        @testset "data parameter = $parameter" for parameter in CUDSS_DATA_PARAMETERS
          parameter ∈ ("perm_row", "perm_col", "perm_reorder", "diag") && continue
          if parameter ≠ "user_perm"
            (parameter == "inertia") && !(structure ∈ ("S", "H")) && continue
            val = cudss_get(solver, parameter)
          else
            perm = Cint[i for i=n:-1:1]
            cudss_set(solver, parameter, perm)
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

      cudss("analysis", solver, x_gpu, b_gpu)
      cudss("factorization", solver, x_gpu, b_gpu)
      cudss("solve", solver, x_gpu, b_gpu)

      r_gpu = b_gpu - A_gpu * x_gpu
      @test norm(r_gpu) ≤ √eps(R)
    end

    @testset "view = $view" for view in ('L', 'U', 'F')
      @testset "Symmetric -- Hermitian" begin
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
        (structure == "H") && cudss_set(solver, "pivot_type", 'N')

        cudss("analysis", solver, X_gpu, B_gpu)
        cudss("factorization", solver, X_gpu, B_gpu)
        cudss("solve", solver, X_gpu, B_gpu)

        R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
        @test norm(R_gpu) ≤ √eps(R)
      end

      @testset "SPD -- HPD" begin
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

        cudss("analysis", solver, X_gpu, B_gpu)
        cudss("factorization", solver, X_gpu, B_gpu)
        cudss("solve", solver, X_gpu, B_gpu)

        R_gpu = B_gpu - CuSparseMatrixCSR(A_cpu) * X_gpu
        @test norm(R_gpu) ≤ √eps(R)
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

    @testset "view = $view" for view in ('F',)
      @testset "Symmetric -- Hermitian" begin
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
          R_gpu = B_gpu - A_gpu * X_gpu
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
          R_gpu = B_gpu - A_gpu * X_gpu
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

      @testset "SPD -- HPD" begin
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
          R_gpu = B_gpu - A_gpu * X_gpu
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
          R_gpu = B_gpu - A_gpu * X_gpu
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

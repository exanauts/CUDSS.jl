function cudss_schur_lu()
  @testset "precision = $T" for T in (Float32, Float64,)  # ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint, Int64)
      @testset "indexing = $index" for index in ('Z',) #, 'O')
        @testset "Dense Schur complement = $dense_schur" for dense_schur in (false, true)
          # A = [A₁₁ A₁₂] where A₁₁ = [4 0], A₁₂ = [1 0 2]
          #     [A₂₁ A₂₂]             [0 5]        [0 3 0]
          #
          # A₂₁ = [0 6] and A₂₂ = [8 0 0 ]
          #       [7 0]           [0 9 1 ]
          #       [0 0]           [0 1 10]
          #
          # The matrix A and the Schur complement S = A₂₂ − A₂₁(A₁₁)⁻¹A₁₂ are:
          #
          #     [4 0 1 0 2 ]
          #     [0 5 0 3 0 ]          [ 8    -3.6  0  ]
          # A = [0 6 8 0 0 ]  and S = [-1.75  9   -2.5]
          #     [7 0 0 9 1 ]          [ 0     1    10 ]
          #     [0 0 0 1 10]
          offset = (index == 'Z') ? one(INT) : zero(INT)
          rows = INT[1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5]
          cols = INT[1, 3, 5, 2, 4, 2, 3, 1, 4, 5, 4, 5]
          vals = T[4.0, 1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0, 9.0, 1.0, 1.0, 10.0]
          A_cpu = sparse(rows, cols, vals, 5, 5)
          A11 = Matrix{T}(A_cpu[1:2,1:2])
          A12 = Matrix{T}(A_cpu[1:2,3:5])
          A21 = Matrix{T}(A_cpu[3:5,1:2])
          A22 = Matrix{T}(A_cpu[3:5,3:5])
          S_cpu = A22 - A21 * (A11 \ A12)

          # Right-hand side such the solution is a vector of ones
          b_cpu = T[7.0, 8.0, 14.0, 17.0, 11.0]

          A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
          if index == 'Z'
            A_gpu.rowPtr .-= offset
            A_gpu.colVal .-= offset
          end
          x_gpu = CuVector{T}(undef, 5)
          b_gpu = CuVector(b_cpu)
          solver = CudssSolver(A_gpu, "G", 'F'; index)

          # Enable the Schur complement computation
          cudss_set(solver, "schur_mode", 1)

          # Rows and columns for the Schur complement of the block A₂₂
          schur_indices = INT[0, 0, 1, 1, 1]
          cudss_set(solver, "user_schur_indices", schur_indices)

          # Compute the Schur complement with a partial factorization
          # [A₁₁ A₁₂] = [L₁₁  0] [U₁₁ U₁₂]
          # [A₂₁ A₂₂]   [L₂₁  I] [ 0   S ]
          cudss("analysis", solver, x_gpu, b_gpu)
          cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)

          # Dimension of the Schur complement nₛ and the number of nonzeros
          (nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

          if dense_schur
            # Dense storage for the Schur complement
            S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
            S_cudss_dense = CudssMatrix(S_gpu)

            # Update the dense matrix S_gpu
            cudss_set(solver, "schur_matrix", S_cudss_dense.matrix)
            cudss_get(solver, "schur_matrix")
            @test Matrix(S_gpu) ≈ S_cpu
          else
            # Sparse storage for the Schur complement
            S_rowPtr = CuVector{INT}(undef, nrows_S+1)
            S_colVal = CuVector{INT}(undef, nnz_S)
            S_nzVal = CuVector{T}(undef, nnz_S)
            dim_S = (nrows_S, ncols_S)
            S_gpu = CuSparseMatrixCSR{T,INT}(S_rowPtr, S_colVal, S_nzVal, dim_S)
            S_cudss_sparse = CudssMatrix(S_gpu, "G", 'F'; index)

            # Update the sparse matrix S_gpu
            cudss_set(solver, "schur_matrix", S_cudss_sparse.matrix)
            cudss_get(solver, "schur_matrix")
            if index == 'Z'
              S_gpu.rowPtr .+= offset
              S_gpu.colVal .+= offset
            end
            @test Matrix(S_gpu) ≈ S_cpu
          end

          # [A₁₁ A₁₂] [x₁] = [b₁] ⟺ A₁₁x₁ = b₁ - A₁₂x₂
          # [A₂₁ A₂₂] [x₂]   [b₂]   Sx₂ = b₂ - A₂₁(A₁₁)⁻¹b₁ = bₛ
          #
          # Compute bₛ with a partial forward solve
          # bₛ is stored in the last nₛ components of b_gpu
          # nₛ = 3 is the size of the Schur complement
          cudss("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous=false)
          bs_gpu = x_gpu[3:5]
          bs_cpu = b_cpu[3:5] - A21 * (A11 \ b_cpu[1:2])
          @test Vector(bs_gpu) ≈ bs_cpu

          if dense_schur
            # Compute x₂ with the dense LU of cuSOLVER
            F, ipiv, _ = CUSOLVER.getrf!(S_gpu)
            x2_gpu = copy(bs_gpu)
            CUSOLVER.getrs!('N', F, ipiv, x2_gpu)
          else
            # Compute x₂ with the sparse LU of cuDSS
            x2_gpu = lu(S_gpu) \ bs_gpu
          end
          x2_cpu = lu(S_cpu) \ bs_cpu
          @test Vector(x2_gpu) ≈ x2_cpu

          # Compute x₁ with a partial backward solve
          # x₂ must be stored in the last nₛ components of x_gpu
          x_gpu[3:5] .= x2_gpu
          cudss("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous=false)
          x1_gpu = b_gpu[1:2]
          x1_cpu = A11 \ (b_cpu[1:2] - A12 * x2_cpu)
          @test Vector(x1_gpu) ≈ x1_cpu
        end
      end
    end
  end
end

function cudss_schur_ldlt()
  @testset "precision = $T" for T in (Float32, Float64,)  # ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint, Int64)
      @testset "indexing = $index" for index in ('Z',) #, 'O')
        @testset "Dense Schur complement = $dense_schur" for dense_schur in (false, true)
          @testset "Triangle of the matrix: $uplo" for (uplo, op) in (('L', tril), ('U', triu), ('F', identity))
            (!dense_schur && uplo == 'F') && continue
            # A = [A₁₁ A₁₂] where A₁₁ = [4 0], A₁₂ = [1 0 2]
            #     [A₂₁ A₂₂]             [0 2]        [0 3 0]
            #
            # A₂₁ = [1 0] and A₂₂ = [3 0 0]
            #       [0 3]           [0 2 1]
            #       [2 0]           [0 1 2]
            #
            # The matrix A and the Schur complement S = A₁₁ − A₁₂(A₂₂)⁻¹A₂₁ are:
            #
            #     [4 0 1 0 2]
            #     [0 2 0 3 0]
            # A = [1 0 3 0 0]  and S = [1   2]
            #     [0 3 0 2 1]          [2  -4]
            #     [2 0 0 1 2]
            offset = (index == 'Z') ? one(INT) : zero(INT)
            rows = INT[1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
            cols = INT[1, 3, 5, 2, 4, 1, 3, 2, 4, 5, 1, 4, 5]
            vals = T[4.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            A_cpu = sparse(rows, cols, vals, 5, 5)
            A11 = Matrix{T}(A_cpu[1:2,1:2])
            A12 = Matrix{T}(A_cpu[1:2,3:5])
            A21 = Matrix{T}(A_cpu[3:5,1:2])
            A22 = Matrix{T}(A_cpu[3:5,3:5])
            S_cpu = A11 - A12 * (A22 \ A21)

            # Right-hand side such the solution is a vector of ones
            b_cpu = T[7.0, 5.0, 4.0, 6.0, 5.0]

            A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> op)
            if index == 'Z'
              A_gpu.rowPtr .-= offset
              A_gpu.colVal .-= offset
            end
            x_gpu = CuVector{T}(undef, 5)
            b_gpu = CuVector{T}(b_cpu)
            structure = T <: Real ? "S" : "H"
            solver = CudssSolver(A_gpu, structure, uplo; index)

            # Enable the Schur complement computation
            cudss_set(solver, "schur_mode", 1)

            # Rows and columns for the Schur complement of the block A₁₁
            schur_indices = INT[1, 1, 0, 0, 0]
            cudss_set(solver, "user_schur_indices", schur_indices)

            # Compute the Schur complement
            cudss("analysis", solver, x_gpu, b_gpu)
            cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)

            # Dimension of the Schur complement nₛ and the number of nonzeros
            (nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

            if dense_schur
              # Dense storage for the Schur complement
              S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
              S_cudss_dense = CudssMatrix(S_gpu)

              # Update the dense matrix S_gpu
              cudss_set(solver, "schur_matrix", S_cudss_dense.matrix)
              cudss_get(solver, "schur_matrix")
              @test Matrix(S_gpu) ≈ S_cpu
            else
              # Maximum number of nonzeros in one triangle of the Schur complement
              nnz_S = min(nnz_S, nrows_S * (nrows_S + 1) ÷ 2)

              # Sparse storage for the Schur complement
              S_rowPtr = CuVector{INT}(undef, nrows_S+1)
              S_colVal = CuVector{INT}(undef, nnz_S)
              S_nzVal = CuVector{T}(undef, nnz_S)
              dim_S = (nrows_S, ncols_S)
              S_gpu = CuSparseMatrixCSR{T,INT}(S_rowPtr, S_colVal, S_nzVal, dim_S)
              S_cudss_sparse = CudssMatrix(S_gpu, structure, uplo; index)

              # Update the sparse matrix S_gpu
              cudss_set(solver, "schur_matrix", S_cudss_sparse.matrix)
              cudss_get(solver, "schur_matrix")
              if index == 'Z'
                S_gpu.rowPtr .+= offset
                S_gpu.colVal .+= offset
              end
              @test Matrix(S_gpu) ≈ op(S_cpu)
            end

            # to be fixed...
            if false
              # [A₁₁ A₁₂] [x₁] = [b₁] ⟺ Sx₁ = b₁ - A₁₂(A₂₂)⁻¹b₂ = bₛ
              # [A₂₁ A₂₂] [x₂]   [b₂]   A₂₂x₂ = b₂ - A₂₁x₁
              #
              # Compute bₛ with a partial forward solve
              # bₛ is stored in the first nₛ = 2 components of x_gpu
              # nₛ = 2 is the size of the Schur complement
              cudss("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous=false)
              bs_gpu = x_gpu[1:2]
              bs_cpu = b_cpu[1:2] - A12 * (A22 \ b_cpu[3:5])
              @test Vector(bs_gpu) ≈ bs_cpu

              if dense_schur
                # Compute x₂ with the dense LDLᵀ / LDLᴴ of cuSOLVER
                uplo_cusolver = (uplo == 'F') ? 'L' : uplo
                F, ipiv, _ = CUSOLVER.sytrf!(uplo_cusolver, S_gpu)
                x1_gpu = copy(bs_gpu)
                CUSOLVER.sytrs!(uplo_cusolver, F, CuVector{Int64}(ipiv), x1_gpu)
              else
                # Compute x₂ with the sparse LDLᵀ / LDLᴴ of cuDSS
                x1_gpu = ldlt(S_gpu) \ bs_gpu
              end
              x1_cpu = bunchkaufman(S_cpu) \ bs_cpu
              @test Vector(x1_gpu) ≈ x1_cpu

              # Compute x₂ with a partial backward solve
              # x₁ must be stored the first nₛ components of x_gpu
              x_gpu[1:2] .= x1_gpu
              cudss("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous=false)
              x2_gpu = b_gpu[3:5]
              x2_cpu = A22 \ (b_cpu[3:5] - A21 * x1_cpu)
              @test Vector(x2_gpu) ≈ x2_cpu
            end
          end
        end
      end
    end
  end
end

function cudss_schur_cholesky()
  @testset "precision = $T" for T in (Float32, Float64,)  # ComplexF32, ComplexF64)
    @testset "integer = $INT" for INT in (Cint, Int64)
      @testset "indexing = $index" for index in ('Z',) # 'O')
        @testset "Dense Schur complement = $dense_schur" for dense_schur in (false, true)
          @testset "Triangle of the matrix: $uplo" for (uplo, op) in (('L', tril), ('U', triu), ('F', identity))
            (!dense_schur && uplo == 'F') && continue
            # A = [A₁₁ A₁₂] where A₁₁ = [2.5  1 ], A₁₂ = [1 0 0]
            #     [A₂₁ A₂₂]             [ 1  2.5]        [0 1 0]
            #
            # A₂₁ = [1 0] and A₂₂ = [2 0 0]
            #       [0 1]           [0 2 0]
            #       [0 0]           [0 0 2]
            #
            # The matrix A and the Schur complement S = A₁₁ − A₁₂(A₂₂)⁻¹A₂₁ are:
            #
            #     [2.5  1   1  0  0]
            #     [ 1  2.5  0  1  0]
            # A = [ 1   0   2  0  0]  and S = [2 1]
            #     [ 0   1   0  2  0]          [1 2]
            #     [ 0   0   0  0  2]
            offset = (index == 'Z') ? one(INT) : zero(INT)
            rows = INT[1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
            cols = INT[1, 2, 3, 1, 2, 4, 1, 3, 2, 4, 5]
            vals = T[2.5, 1, 1, 1, 2.5, 1, 1, 2, 1, 2, 2]
            A_cpu = sparse(rows, cols, vals, 5, 5)
            A11 = Matrix{T}(A_cpu[1:2,1:2])
            A12 = Matrix{T}(A_cpu[1:2,3:5])
            A21 = Matrix{T}(A_cpu[3:5,1:2])
            A22 = Matrix{T}(A_cpu[3:5,3:5])
            S_cpu = A11 - A12 * (A22 \ A21)

            # Right-hand side such the solution is a vector of ones
            b_cpu = T[4.5, 4.5, 3.0, 3.0, 2.0]

            A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu |> op)
            if index == 'Z'
              A_gpu.rowPtr .-= offset
              A_gpu.colVal .-= offset
            end
            x_gpu = CuVector{T}(undef, 5)
            b_gpu = CuVector{T}(b_cpu)
            structure = T <: Real ? "SPD" : "HPD"
            solver = CudssSolver(A_gpu, structure, uplo; index)

            # Enable the Schur complement computation
            cudss_set(solver, "schur_mode", 1)

            # Rows and columns for the Schur complement of the block A₁₁
            schur_indices = INT[1, 1, 0, 0, 0]
            cudss_set(solver, "user_schur_indices", schur_indices)

            # Compute the Schur complement
            cudss("analysis", solver, x_gpu, b_gpu)
            cudss("factorization", solver, x_gpu, b_gpu; asynchronous=false)

            # Dimension of the Schur complement nₛ and the number of nonzeros
            (nrows_S, ncols_S, nnz_S) = cudss_get(solver, "schur_shape")

            if dense_schur
              # Dense storage for the Schur complement
              S_gpu = CuMatrix{T}(undef, nrows_S, ncols_S)
              S_cudss_dense = CudssMatrix(S_gpu)

              # Update the dense matrix S_gpu
              cudss_set(solver, "schur_matrix", S_cudss_dense.matrix)
              cudss_get(solver, "schur_matrix")
              @test Matrix(S_gpu) ≈ S_cpu
            else
              # Maximum number of nonzeros in one triangle of the Schur complement
              nnz_S = min(nnz_S, nrows_S * (nrows_S + 1) ÷ 2)

              # Sparse storage for the Schur complement
              S_rowPtr = CuVector{INT}(undef, nrows_S+1)
              S_colVal = CuVector{INT}(undef, nnz_S)
              S_nzVal = CuVector{T}(undef, nnz_S)
              dim_S = (nrows_S, ncols_S)
              S_gpu = CuSparseMatrixCSR{T,INT}(S_rowPtr, S_colVal, S_nzVal, dim_S)
              S_cudss_sparse = CudssMatrix(S_gpu, structure, uplo; index)

              # Update the sparse matrix S_gpu
              cudss_set(solver, "schur_matrix", S_cudss_sparse.matrix)
              cudss_get(solver, "schur_matrix")
              if index == 'Z'
                S_gpu.rowPtr .+= offset
                S_gpu.colVal .+= offset
              end
              @test Matrix(S_gpu) ≈ op(S_cpu)
            end

            # to be fixed...
            if false
              # [A₁₁ A₁₂] [x₁] = [b₁] ⟺ Sx₁ = b₁ - A₁₂(A₂₂)⁻¹b₂ = bₛ
              # [A₂₁ A₂₂] [x₂]   [b₂]   A₂₂x₂ = b₂ - A₂₁x₁
              #
              # Compute bₛ with a partial forward solve
              # bₛ is stored in the first nₛ = 2 components of x_gpu
              # nₛ = 2 is the size of the Schur complement
              cudss("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous=false)
              bs_gpu = x_gpu[1:2]
              bs_cpu = b_cpu[1:2] - A12 * (A22 \ b_cpu[3:5])
              @test Vector(bs_gpu) ≈ bs_cpu

              if dense_schur
                # Compute x₂ with the dense LLᵀ / LLᴴ of cuSOLVER
                uplo_cusolver = (uplo == 'F') ? 'L' : uplo
                F, _ = CUSOLVER.potrf!(uplo_cusolver, S_gpu)
                x1_gpu = copy(bs_gpu)
                CUSOLVER.potrs!(uplo_cusolver, F, x1_gpu)
              else
                # Compute x₂ with the sparse LLᵀ / LLᴴ of cuDSS
                x1_gpu = cholesky(S_gpu) \ bs_gpu
              end
              x1_cpu = cholesky(S_cpu) \ bs_cpu
              @test Vector(x1_gpu) ≈ x1_cpu

              # Compute x₂ with a partial backward solve
              # x₁ must be stored the first nₛ components of x_gpu
              x_gpu[1:2] .= x1_gpu
              cudss("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous=false)
              x2_gpu = b_gpu[3:5]
              x2_cpu = A22 \ (b_cpu[3:5] - A21 * x1_cpu)
              @test Vector(x2_gpu) ≈ x2_cpu
            end
          end
        end
      end
    end
  end
end

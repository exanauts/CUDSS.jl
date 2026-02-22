function uniform_batch_lu()
  @testset "$name_api API" for (name_api, generic) in (("cuDSS", false), ("Generic", true))
    @testset "strided storage = $strided" for strided in (false, true)
      @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "integer = $INT" for INT in (Cint,) # Int64)
          # Collection of unsymmetric linear systems
          #        [1+λ  0   3  ]
          # A(λ) = [ 4  5+λ  0  ]
          #        [ 2   6  2+λ ]
          R = real(T)
          n = 3
          nbatch = 3
          nnzA = 7
          rowPtr = CuVector{INT}([1, 3, 5, 8])
          colVal = CuVector{INT}([1, 3, 1, 2, 1, 2, 3])

          # List of values for λ
          Λ = [1.0, 10.0, -20.0]
          if strided
            nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                                 1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                                 1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
          else
            nzVal = CuMatrix{T}([1+Λ[1] 1+Λ[2] 1+Λ[3];
                                 3      3      3     ;
                                 4      4      4     ;
                                 5+Λ[1] 5+Λ[2] 5+Λ[3];
                                 2      2      2     ;
                                 6      6      6     ;
                                 2+Λ[1] 2+Λ[2] 2+Λ[3]])
          end

          if generic
            Aλ_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nzVal, (n,n))
            solver = lu(Aλ_gpu)
          else
            # Constructor for uniform batch of systems
            solver = CudssSolver(rowPtr, colVal, nzVal, "G", 'F')

            # Specify that it is a uniform batch of size "nbatch"
            cudss_set(solver, "ubatch_size", nbatch)
          end

          if strided
            xλ_gpu = CuVector{T}(undef, n * nbatch)
            bλ_gpu = CuVector{T}([1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0,
                                  7.0, 8.0, 9.0])
          else
            xλ_gpu = CuMatrix{T}(undef, n, nbatch)
            bλ_gpu = CuMatrix{T}([1.0 4.0 7.0;
                                  2.0 5.0 8.0;
                                  3.0 6.0 9.0])
          end

          if generic
            ldiv!(xλ_gpu, solver, bλ_gpu)
          else
            cudss_bλ_gpu = CudssMatrix(T, n; nbatch)
            cudss_update(cudss_bλ_gpu, bλ_gpu)

            cudss_xλ_gpu = CudssMatrix(T, n; nbatch)
            cudss_update(cudss_xλ_gpu, xλ_gpu)

            cudss("analysis", solver, cudss_xλ_gpu, cudss_bλ_gpu)
            cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
            cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
          end

          rλ_gpu = rand(R, nbatch)
          for i = 1:nbatch
            if strided
              nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
              x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
            else
              nz = nzVal[:,i]
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              b_gpu = bλ_gpu[:,i]
              x_gpu = xλ_gpu[:,i]
            end
            r_gpu = b_gpu - A_gpu * x_gpu
            rλ_gpu[i] = norm(r_gpu)
          end
          @test norm(rλ_gpu) ≤ √eps(R)

          # Refactorize all matrices of the uniform batch
          Λ = [-2.0, -10.0, 30.0]
          if strided
            new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                                     1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                                     1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
          else
            new_nzVal = CuMatrix{T}([1+Λ[1] 1+Λ[2] 1+Λ[3];
                                     3      3      3     ;
                                     4      4      4     ;
                                     5+Λ[1] 5+Λ[2] 5+Λ[3];
                                     2      2      2     ;
                                     6      6      6     ;
                                     2+Λ[1] 2+Λ[2] 2+Λ[3]])
          end

          if generic
            Aλ_gpu.nzVal = vec(new_nzVal)
            lu!(solver, Aλ_gpu)

            xλ_gpu .= bλ_gpu
            ldiv!(solver, xλ_gpu)
          else
            cudss_update(solver, rowPtr, colVal, new_nzVal)
            cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
            cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu; asynchronous=false)
          end

          for i = 1:nbatch
            if strided
              nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
              x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
            else
              nz = new_nzVal[:,i]
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              b_gpu = bλ_gpu[:,i]
              x_gpu = xλ_gpu[:,i]
            end
            r_gpu = b_gpu - A_gpu * x_gpu
            rλ_gpu[i] = norm(r_gpu)
          end
          @test norm(rλ_gpu) ≤ √eps(R)
        end
      end
    end
  end
end

function uniform_batch_ldlt()
  @testset "$name_api API" for (name_api, generic) in (("cuDSS", false), ("Generic", true))
    @testset "strided storage = $strided" for strided in (false, true)
      @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "integer = $INT" for INT in (Cint,) # Int64)
          @testset "Triangle of the sparse matrix = $uplo" for uplo in ('L', 'U', 'F')
            R = real(T)
            n = 5
            nbatch = 2
            nrhs = 2
            if uplo == 'L'
              nnzA = 8
              rowPtr = CuVector{INT}([1, 2, 3, 6, 7, 9])
              colVal = CuVector{INT}([1, 2, 1, 2, 3, 4, 3, 5])
              if T <: AbstractFloat
                if strided
                  nzVal = CuVector{T}([4, 3, 1, 2, 5, 1, 1, 2,
                                       2, 3, 1, 1, 6, 4, 2, 8])
                else
                  nzVal = CuMatrix{T}([4 2;
                                       3 3;
                                       1 1;
                                       2 1;
                                       5 6;
                                       1 4;
                                       1 2;
                                       2 8])
                end
              else
                if strided
                  nzVal = CuVector{T}([4, 3, 1+im, 2-im, 5, 1, 1+im, 2,
                                       2, 3, 1-im, 1+im, 6, 4, 2-im, 8])
                else
                  nzVal = CuMatrix{T}([4    2   ;
                                       3    3   ;
                                       1+im 1-im;
                                       2-im 1+im;
                                       5    6   ;
                                       1    4   ;
                                       1+im 2-im;
                                       2    8   ])
                end
              end
            end
            if uplo == 'U'
              nnzA = 8
              rowPtr = CuVector{INT}([1, 3, 5, 7, 8, 9])
              colVal = CuVector{INT}([1, 3, 2, 3, 3, 5, 4, 5])
              if T <: AbstractFloat
                if strided
                  nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                                       2, 1, 3, 1, 6, 2, 4, 8])
                else
                  nzVal = CuMatrix{T}([4 2;
                                       1 1;
                                       3 3;
                                       2 1;
                                       5 6;
                                       1 2;
                                       1 4;
                                       2 8])
                end
              else
                if strided
                  nzVal = CuVector{T}([4, 1-im, 3, 2+im, 5, 1-im, 1, 2,
                                       2, 1+im, 3, 1-im, 6, 2+im, 4, 8])
                else
                  nzVal = CuMatrix{T}([4    2   ;
                                       1-im 1+im;
                                       3    3   ;
                                       2+im 1-im;
                                       5    6   ;
                                       1-im 2+im;
                                       1    4   ;
                                       2    8   ])
                end
              end
            end
            if uplo == 'F'
              nnzA = 11
              rowPtr = CuVector{INT}([1, 3, 5, 9, 10, 12])
              colVal = CuVector{INT}([1, 3, 2, 3, 1, 2, 3, 5, 4, 3, 5])
              if T <: AbstractFloat
                if strided
                  nzVal = CuVector{T}([4, 1, 3, 2, 1, 2, 5, 1, 1, 1, 2,
                                       2, 1, 3, 1, 1, 1, 6, 2, 4, 2, 8])
                else
                  nzVal = CuMatrix{T}([4 2;
                                       1 1;
                                       3 3;
                                       2 1;
                                       1 1;
                                       2 1;
                                       5 6;
                                       1 2;
                                       1 4;
                                       1 2;
                                       2 8])
                end
              else
                if strided
                  nzVal = CuVector{T}([4, 1-im, 3, 2+im, 1+im, 2-im, 5, 1-im, 1, 1+im, 2,
                                       2, 1+im, 3, 1-im, 1-im, 1+im, 6, 2+im, 4, 2-im, 8])
                else
                  nzVal = CuMatrix{T}([4    2   ;
                                       1-im 1+im;
                                       3    3   ;
                                       2+im 1-im;
                                       1+im 1-im;
                                       2-im 1+im;
                                       5    6   ;
                                       1-im 2+im;
                                       1    4   ;
                                       1+im 2-im;
                                       2    8   ])
                end
              end
            end

            if T <: AbstractFloat
              if strided
                Xs_gpu = CuVector{T}(undef, n * nrhs * nbatch)
                Bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,  -7, -12, -25, -4, -13,
                                      13, 15, 29, 8, 14, -13, -15, -29, -8, -14])
              else
                Xs_gpu = CuArray{T}(undef, n, nrhs, nbatch)
                Bs_gpu = CuArray{T}([7 -7 ;
                                    12 -12;
                                    25 -25;
                                    4  -4 ;
                                    13 -13;;;
                                    13 -13;
                                    15 -15;
                                    29 -29;
                                    8  -8 ;
                                    14 -14])
              end
            else
              if strided
                Xs_gpu = CuVector{T}(undef, n * nrhs * nbatch)
                Bs_gpu = CuVector{T}([ 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im,
                                      13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im])
              else
                Xs_gpu = CuArray{T}(undef, n, nrhs, nbatch)
                Bs_gpu = CuArray{T}([ 7+im  -7+im;
                                     12+im -12+im;
                                     25+im -25+im;
                                      4+im  -4+im;
                                     13+im -13+im;;;
                                     13-im -13-im;
                                     15-im -15-im;
                                     29-im -29-im;
                                      8-im  -8-im;
                                     14-im -14-im])
              end
            end

            if generic
              As_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nzVal, (n,n))
              solver = ldlt(As_gpu; view=uplo)
            else
              # Constructor for uniform batch of systems
              solver = CudssSolver(rowPtr, colVal, nzVal, "H", uplo)

              # Specify that it is a uniform batch of size "nbatch"
              cudss_set(solver, "ubatch_size", nbatch)
            end

            if generic
              ldiv!(Xs_gpu, solver, Bs_gpu)
            else
              cudss_Bs_gpu = CudssMatrix(T, n, nrhs; nbatch)
              cudss_update(cudss_Bs_gpu, Bs_gpu)

              cudss_Xs_gpu = CudssMatrix(T, n, nrhs; nbatch)
              cudss_update(cudss_Xs_gpu, Xs_gpu)

              cudss("analysis", solver, cudss_Xs_gpu, cudss_Bs_gpu)
              cudss("factorization", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)
              cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)
            end

            Rs_gpu = rand(R, nbatch)
            for i = 1:nbatch
              if strided
                nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
              else
                nz = nzVal[:,i]
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              A_cpu = SparseMatrixCSC(A_gpu)
              if (uplo == 'L' || uplo == 'U')
                A_cpu = A_cpu + A_cpu' - Diagonal(A_cpu)
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
              if strided
                B_gpu = reshape(Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
                X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
              else
                B_gpu = Bs_gpu[:,:,i]
                X_gpu = Xs_gpu[:,:,i]
              end
              R_gpu = B_gpu - A_gpu * X_gpu
              Rs_gpu[i] = norm(R_gpu)
            end
            @test norm(Rs_gpu) ≤ √eps(R)

            if uplo == 'L'
              if T <: AbstractFloat
                if strided
                  new_nzVal = CuVector{T}([-4, -3, -1, -2, -5, -1, -1, -2,
                                           -2, -3, -1, -1, -6, -4, -2, -8])
                else
                  new_nzVal = CuMatrix{T}([-4 -2;
                                           -3 -3;
                                           -1 -1;
                                           -2 -1;
                                           -5 -6;
                                           -1 -4;
                                           -1 -2;
                                           -2 -8])
                end
              else
                if strided
                  new_nzVal = CuVector{T}([-4, -3, -1-im, -2+im, -5, -1, -1-im, -2,
                                           -2, -3, -1+im, -1-im, -6, -4, -2+im, -8])
                else
                  new_nzVal = CuMatrix{T}([-4    -2   ;
                                           -3    -3   ;
                                           -1-im -1+im;
                                           -2+im -1-im;
                                           -5    -6   ;
                                           -1    -4   ;
                                           -1-im -2+im;
                                           -2    -8   ])
                end
              end
            end
            if uplo == 'U'
              if T <: AbstractFloat
                if strided
                  new_nzVal = CuVector{T}([-4, -1, -3, -2, -5, -1, -1, -2,
                                           -2, -1, -3, -1, -6, -2, -4, -8])
                else
                  new_nzVal = CuMatrix{T}([-4 -2;
                                           -1 -1;
                                           -3 -3;
                                           -2 -1;
                                           -5 -6;
                                           -1 -2;
                                           -1 -4;
                                           -2 -8])
                end
              else
                if strided
                  new_nzVal = CuVector{T}([-4, -1+im, -3, -2-im, -5, -1+im, -1, -2,
                                           -2, -1-im, -3, -1+im, -6, -2-im, -4, -8])
                else
                  new_nzVal = CuMatrix{T}([-4    -2   ;
                                           -1+im -1-im;
                                           -3    -3   ;
                                           -2-im -1+im;
                                           -5    -6   ;
                                           -1+im -2-im;
                                           -1    -4   ;
                                           -2    -8   ])
                end
              end
            end
            if uplo == 'F'
              if T <: AbstractFloat
                if strided
                  new_nzVal = CuVector{T}([-4, -1, -3, -2, -1, -2, -5, -1, -1, -1, -2,
                                           -2, -1, -3, -1, -1, -1, -6, -2, -4, -2, -8])
                else
                  new_nzVal = CuMatrix{T}([-4 -2;
                                           -1 -1;
                                           -3 -3;
                                           -2 -1;
                                           -1 -1;
                                           -2 -1;
                                           -5 -6;
                                           -1 -2;
                                           -1 -4;
                                           -1 -2;
                                           -2 -8])
                end
              else
                if strided
                  new_nzVal = CuVector{T}([-4, -1+im, -3, -2-im, -1-im, -2+im, -5, -1+im, -1, -1-im, -2,
                                           -2, -1-im, -3, -1+im, -1+im, -1-im, -6, -2-im, -4, -2+im, -8])
                else
                  new_nzVal = CuMatrix{T}([-4    -2   ;
                                           -1+im -1-im;
                                           -3    -3   ;
                                           -2-im -1+im;
                                           -1-im -1+im;
                                           -2+im -1-im;
                                           -5    -6   ;
                                           -1+im -2-im;
                                           -1    -4   ;
                                           -1-im -2+im;
                                           -2    -8   ])
                end
              end
            end

            if generic
              As_gpu.nzVal = new_nzVal
              ldlt!(solver, As_gpu)
            else
              cudss_update(solver, rowPtr, colVal, new_nzVal)
              cudss("refactorization", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)
            end

            if T <: AbstractFloat
              if strided
                new_Bs_gpu = CuVector{T}([13, 15, 29, 8, 14, -13, -15, -29, -8, -14,
                                           7, 12, 25, 4, 13,  -7, -12, -25, -4, -13])
              else
                new_Bs_gpu = CuArray{T}([13 -13;
                                         15 -15;
                                         29 -29;
                                         8  -8 ;
                                         14 -14;;;
                                         7  -7 ;
                                         12 -12;
                                         25 -25;
                                         4  -4 ;
                                         13 -13])
              end
            else
              if strided
                new_Bs_gpu = CuVector{T}([13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im,
                                           7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im])
              else
                new_Bs_gpu = CuArray{T}([13-im -13-im;
                                         15-im -15-im;
                                         29-im -29-im;
                                          8-im  -8-im;
                                         14-im -14-im;;;
                                          7+im  -7+im;
                                         12+im -12+im;
                                         25+im -25+im;
                                          4+im  -4+im;
                                         13+im -13+im])
              end
            end

            if generic
              Xs_gpu .= new_Bs_gpu
              ldiv!(solver, Xs_gpu)
            else
              cudss_update(cudss_Bs_gpu, new_Bs_gpu)
              cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu; asynchronous=false)
            end

            for i = 1:nbatch
              if strided
                nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
              else
                nz = new_nzVal[:,i]
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              A_cpu = SparseMatrixCSC(A_gpu)
              if (uplo == 'L' || uplo == 'U')
                A_cpu = A_cpu + A_cpu' - Diagonal(A_cpu)
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
              if strided
                B_gpu = reshape(new_Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
                X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
              else
                B_gpu = new_Bs_gpu[:,:,i]
                X_gpu = Xs_gpu[:,:,i]
              end
              R_gpu = B_gpu - A_gpu * X_gpu
              Rs_gpu[i] = norm(R_gpu)
            end
            @test norm(Rs_gpu) ≤ √eps(R)
          end
        end
      end
    end
  end
end

function uniform_batch_cholesky()
  @testset "$name_api API" for (name_api, generic) in (("cuDSS", false), ("Generic", true))
    @testset "strided storage = $strided" for strided in (false, true)
      @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "integer = $INT" for INT in (Cint,) # Int64)
          @testset "Triangle of the sparse matrix = $uplo" for uplo in ('L', 'U', 'F')
            R = real(T)
            n = 5
            nbatch = 2
            if uplo == 'L'
              nnzA = 8
              rowPtr = CuVector{INT}([1, 2, 3, 6, 7, 9])
              colVal = CuVector{INT}([1, 2, 1, 2, 3, 4, 3, 5])
              if strided
                nzVal = CuVector{T}([4, 3, 1, 2, 5, 1, 1, 2,
                                     2, 3, 1, 1, 6, 2, 4, 8])
              else
                nzVal = CuMatrix{T}([4 2;
                                     3 3;
                                     1 1;
                                     2 1;
                                     5 6;
                                     1 2;
                                     1 4;
                                     2 8])
              end
            end
            if uplo == 'U'
              nnzA = 8
              rowPtr = CuVector{INT}([1, 3, 5, 7, 8, 9])
              colVal = CuVector{INT}([1, 3, 2, 3, 3, 5, 4, 5])
              if strided
                nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                                     2, 1, 3, 1, 6, 2, 4, 8])
              else
                nzVal = CuMatrix{T}([4 2;
                                     1 1;
                                     3 3;
                                     2 1;
                                     5 6;
                                     1 2;
                                     1 4;
                                     2 8])
              end
            end
            if uplo == 'F'
              nnzA = 11
              rowPtr = CuVector{INT}([1, 3, 5, 9, 10, 12])
              colVal = CuVector{INT}([1, 3, 2, 3, 1, 2, 3, 5, 4, 3, 5])
              if strided
                nzVal = CuVector{T}([4, 1, 3, 2, 1, 2, 5, 1, 1, 1, 2,
                                     2, 1, 3, 1, 1, 1, 6, 2, 4, 2, 8])
              else
                nzVal = CuMatrix{T}([4 2;
                                     1 1;
                                     3 3;
                                     2 1;
                                     1 1;
                                     2 1;
                                     5 6;
                                     1 2;
                                     1 4;
                                     1 2;
                                     2 8])
              end
            end

            if generic
              As_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nzVal, (n,n))
              solver = cholesky(As_gpu; view=uplo)
            else
              # Constructor for uniform batch of systems
              solver = CudssSolver(rowPtr, colVal, nzVal, "HPD", uplo)

              # Specify that it is a uniform batch of size "nbatch"
              cudss_set(solver, "ubatch_size", nbatch)
            end

            if strided
              xs_gpu = CuVector{T}(undef, n * nbatch)
              bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,
                                    13, 15, 29, 8, 14])
            else
              xs_gpu = CuMatrix{T}(undef, n, nbatch)
              bs_gpu = CuMatrix{T}([7  13;
                                    12 15;
                                    25 29;
                                    4  8 ;
                                    13 14])
            end

            if generic
              ldiv!(xs_gpu, solver, bs_gpu)
            else
              cudss_bs_gpu = CudssMatrix(T, n; nbatch)
              cudss_update(cudss_bs_gpu, bs_gpu)

              cudss_xs_gpu = CudssMatrix(T, n; nbatch)
              cudss_update(cudss_xs_gpu, xs_gpu)

              cudss("analysis", solver, cudss_xs_gpu, cudss_bs_gpu)
              cudss("factorization", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)
              cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)
            end

            rs_gpu = rand(R, nbatch)
            for i = 1:nbatch
              if strided
                nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
              else
                nz = nzVal[:,i]
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              A_cpu = SparseMatrixCSC(A_gpu)
              if (uplo == 'L' || uplo == 'U')
                A_cpu = A_cpu + A_cpu' - Diagonal(A_cpu)
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
              if strided
                b_gpu = bs_gpu[1 + (i-1) * n : i * n]
                x_gpu = xs_gpu[1 + (i-1) * n : i * n]
              else
                b_gpu = bs_gpu[:,i]
                x_gpu = xs_gpu[:,i]
              end
              r_gpu = b_gpu - A_gpu * x_gpu
              rs_gpu[i] = norm(r_gpu)
            end
            @test norm(rs_gpu) ≤ √eps(R)

            if uplo == 'L'
              if strided
                new_nzVal = CuVector{T}([8, 6, 2, 4, 10,  2, 2,  4,
                                         6, 9, 3, 3, 18, 12, 6, 24])
              else
                new_nzVal = CuMatrix{T}([8  6 ;
                                         6  9 ;
                                         2  3 ;
                                         4  3 ;
                                         10 18;
                                         2  12;
                                         2  6 ;
                                         4  24])
              end
            end
            if uplo == 'U'
              if strided
                new_nzVal = CuVector{T}([8, 2, 6, 4, 10, 2,  2,  4,
                                         6, 3, 9, 3, 18, 6, 12, 24])
              else
                new_nzVal = CuMatrix{T}([8  6 ;
                                         2  3 ;
                                         6  9 ;
                                         4  3 ;
                                         10 18;
                                         2  6 ;
                                         2  12;
                                         4  24])
              end
            end
            if uplo == 'F'
              if strided
                new_nzVal = CuVector{T}([8, 2, 6, 4, 2, 4, 10, 2,  2, 2,  4,
                                         6, 3, 9, 3, 3, 3, 18, 6, 12, 6, 24])
              else
                new_nzVal = CuMatrix{T}([8  6 ;
                                         2  3 ;
                                         6  9 ;
                                         4  3 ;
                                         2  3 ;
                                         4  3 ;
                                         10 18;
                                         2  6 ;
                                         2  12;
                                         2  6 ;
                                         4  24])
              end
            end

            if generic
              As_gpu.nzVal = vec(new_nzVal)
              cholesky!(solver, As_gpu)

              xs_gpu .= bs_gpu
              ldiv!(solver, xs_gpu)
            else
              cudss_update(solver, rowPtr, colVal, new_nzVal)
              cudss("refactorization", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)
              cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu; asynchronous=false)
            end

            for i = 1:nbatch
              if strided
                nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
              else
                nz = new_nzVal[:,i]
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(rowPtr, colVal, nz, (n,n))
              A_cpu = SparseMatrixCSC(A_gpu)
              if (uplo == 'L' || uplo == 'U')
                A_cpu = A_cpu + A_cpu' - Diagonal(A_cpu)
              end
              A_gpu = CuSparseMatrixCSR{T,INT}(A_cpu)
              if strided
                b_gpu = bs_gpu[1 + (i-1) * n : i * n]
                x_gpu = xs_gpu[1 + (i-1) * n : i * n]
              else
                b_gpu = bs_gpu[:,i]
                x_gpu = xs_gpu[:,i]
              end
              r_gpu = b_gpu - A_gpu * x_gpu
              rs_gpu[i] = norm(r_gpu)
            end
            @test norm(rs_gpu) ≤ √eps(R)
          end
        end
      end
    end
  end
end

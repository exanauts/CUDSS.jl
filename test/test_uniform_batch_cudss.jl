function cudss_uniform_batch_lu()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    # Collection of unsymmetric linear systems
    #        [1+λ  0   3  ]
    # A(λ) = [ 4  5+λ  0  ]
    #        [ 2   6  2+λ ]
    R = real(T)
    n = 3
    nbatch = 3
    nnzA = 7
    rowPtr = CuVector{Cint}([1, 3, 5, 8])
    colVal = CuVector{Cint}([1, 3, 1, 2, 1, 2, 3])

    # List of values for λ
    Λ = [1.0, 10.0, -20.0]
    nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                         1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                         1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])

    cudss_bλ_gpu = CudssMatrix(T, n; nbatch)
    bλ_gpu = CuVector{T}([1.0, 2.0, 3.0,
                          4.0, 5.0, 6.0,
                          7.0, 8.0, 9.0])
    cudss_update(cudss_bλ_gpu, bλ_gpu)

    cudss_xλ_gpu = CudssMatrix(T, n; nbatch)
    xλ_gpu = CuVector{T}(undef, n * nbatch)
    cudss_update(cudss_xλ_gpu, xλ_gpu)

    # Constructor for uniform batch of systems
    solver = CudssSolver(rowPtr, colVal, nzVal, "G", 'F')

    # Specify that it is a uniform batch of size "nbatch"
    cudss_set(solver, "ubatch_size", nbatch)

    cudss("analysis", solver, cudss_xλ_gpu, cudss_bλ_gpu)
    cudss("factorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
    cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

    rλ_gpu = rand(R, nbatch)
    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
        x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rλ_gpu[i] = norm(r_gpu)
    end
    @test norm(rλ_gpu) ≤ √eps(R)

    # Refactorize all matrices of the uniform batch
    Λ = [-2.0, -10.0, 30.0]
    new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                             1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                             1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])

    cudss_update(solver, rowPtr, colVal, new_nzVal)
    cudss("refactorization", solver, cudss_xλ_gpu, cudss_bλ_gpu)
    cudss("solve", solver, cudss_xλ_gpu, cudss_bλ_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
        x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rλ_gpu[i] = norm(r_gpu)
    end
    @test norm(rλ_gpu) ≤ √eps(R)
  end
end

function generic_uniform_batch_lu()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    # Collection of unsymmetric linear systems
    #        [1+λ  0   3  ]
    # A(λ) = [ 4  5+λ  0  ]
    #        [ 2   6  2+λ ]
    R = real(T)
    n = 3
    nbatch = 3
    nnzA = 7
    rowPtr = CuVector{Cint}([1, 3, 5, 8])
    colVal = CuVector{Cint}([1, 3, 1, 2, 1, 2, 3])

    # List of values for λ
    Λ = [1.0, 10.0, -20.0]
    nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                         1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                         1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])

    Aλ_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, (n,n))
    solver = lu(Aλ_gpu)

    bλ_gpu = CuVector{T}([1.0, 2.0, 3.0,
                          4.0, 5.0, 6.0,
                          7.0, 8.0, 9.0])
    xλ_gpu = CuVector{T}(undef, n * nbatch)
    ldiv!(xλ_gpu, solver, bλ_gpu)

    rλ_gpu = rand(R, nbatch)
    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
        x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rλ_gpu[i] = norm(r_gpu)
    end
    @test norm(rλ_gpu) ≤ √eps(R)

    bλ2_gpu = CuMatrix{T}([1.0 4.0 7.0;
                           2.0 5.0 8.0;
                           3.0 6.0 9.0])
    xλ2_gpu = CuMatrix{T}(undef, n, nbatch)
    ldiv!(xλ2_gpu, solver, bλ2_gpu)

    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        b_gpu = bλ2_gpu[:, i]
        x_gpu = xλ2_gpu[:, i]
        r_gpu = b_gpu - A_gpu * x_gpu
        rλ_gpu[i] = norm(r_gpu)
    end
    @test norm(rλ_gpu) ≤ √eps(R)

    # Refactorize all matrices of the uniform batch
    Λ = [-2.0, -10.0, 30.0]
    new_nzVal = CuVector{T}([1+Λ[1], 3, 4, 5+Λ[1], 2, 6, 2+Λ[1],
                             1+Λ[2], 3, 4, 5+Λ[2], 2, 6, 2+Λ[2],
                             1+Λ[3], 3, 4, 5+Λ[3], 2, 6, 2+Λ[3]])
    Aλ_gpu.nzVal = new_nzVal
    lu!(solver, Aλ_gpu)

    xλ_gpu .= bλ_gpu
    ldiv!(solver, xλ_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        b_gpu = bλ_gpu[1 + (i-1) * n : i * n]
        x_gpu = xλ_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rλ_gpu[i] = norm(r_gpu)
    end
    @test norm(rλ_gpu) ≤ √eps(R)

    xλ2_gpu .= bλ2_gpu
    ldiv!(solver, xλ2_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        b_gpu = bλ2_gpu[:, i]
        x_gpu = xλ2_gpu[:, i]
        r_gpu = b_gpu - A_gpu * x_gpu
        rλ_gpu[i] = norm(r_gpu)
    end
    @test norm(rλ_gpu) ≤ √eps(R)
  end
end

function cudss_uniform_batch_ldlt()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    n = 5
    nbatch = 2
    nrhs = 2
    nnzA = 8
    rowPtr = CuVector{Cint}([1, 2, 3, 6, 7, 9])
    colVal = CuVector{Cint}([1, 2, 1, 2, 3, 4, 3, 5])
    if T <: AbstractFloat
      nzVal = CuVector{T}([4, 3, 1, 2, 5, 1, 1, 2,
                           2, 3, 1, 1, 6, 4, 2, 8])
    else
      nzVal = CuVector{T}([4, 3, 1+im, 2-im, 5, 1, 1+im, 2,
                           2, 3, 1-im, 1+im, 6, 4, 2-im, 8])
    end

    cudss_Bs_gpu = CudssMatrix(T, n, nrhs; nbatch)
    if T <: AbstractFloat
      Bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,  -7, -12, -25, -4, -13,
                            13, 15, 29, 8, 14, -13, -15, -29, -8, -14])
    else
      Bs_gpu = CuVector{T}([ 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im,
                            13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im])
    end
    cudss_update(cudss_Bs_gpu, Bs_gpu)

    cudss_Xs_gpu = CudssMatrix(T, n, nrhs; nbatch)
    Xs_gpu = CuVector{T}(undef, n * nrhs * nbatch)
    cudss_update(cudss_Xs_gpu, Xs_gpu)

    # Constructor for uniform batch of systems
    solver = CudssSolver(rowPtr, colVal, nzVal, "H", 'L')

    # Specify that it is a uniform batch of size "nbatch"
    cudss_set(solver, "ubatch_size", nbatch)

    cudss("analysis", solver, cudss_Xs_gpu, cudss_Bs_gpu)
    cudss("factorization", solver, cudss_Xs_gpu, cudss_Bs_gpu)
    cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu)

    Rs_gpu = rand(R, nbatch)
    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        B_gpu = reshape(Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        R_gpu = B_gpu - A_gpu * X_gpu
        Rs_gpu[i] = norm(R_gpu)
    end
    @test norm(Rs_gpu) ≤ √eps(R)

    if T <: AbstractFloat
      new_nzVal = CuVector{T}([-4, -3,  1, -2, -5, -1, -1, -2,
                               -2, -3, -1, -1, -6, -4, -2, -8])
    else
      new_nzVal = CuVector{T}([-4, -3,  1-im, -2+im, -5, -1, -1-im, -2,
                               -2, -3, -1+im, -1-im, -6, -4, -2+im, -8])
    end

    cudss_update(solver, rowPtr, colVal, new_nzVal)
    cudss("refactorization", solver, cudss_Xs_gpu, cudss_Bs_gpu)

    if T <: AbstractFloat
      new_Bs_gpu = CuVector{T}([13, 15, 29, 8, 14, -13, -15, -29, -8, -14,
                                 7, 12, 25, 4, 13,  -7, -12, -25, -4, -13])
    else
      new_Bs_gpu = CuVector{T}([13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im,
                                 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im])
    end
    cudss_update(cudss_Bs_gpu, new_Bs_gpu)
    cudss("solve", solver, cudss_Xs_gpu, cudss_Bs_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        B_gpu = reshape(new_Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        R_gpu = B_gpu - A_gpu * X_gpu
        Rs_gpu[i] = norm(R_gpu)
    end
    @test norm(Rs_gpu) ≤ √eps(R)
  end
end

function generic_uniform_batch_ldlt()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    n = 5
    nbatch = 2
    nrhs = 2
    nnzA = 8
    rowPtr = CuVector{Cint}([1, 2, 3, 6, 7, 9])
    colVal = CuVector{Cint}([1, 2, 1, 2, 3, 4, 3, 5])
    if T <: AbstractFloat
      nzVal = CuVector{T}([4, 3, 1, 2, 5, 1, 1, 2,
                           2, 3, 1, 1, 6, 4, 2, 8])
    else
      nzVal = CuVector{T}([4, 3, 1+im, 2-im, 5, 1, 1+im, 2,
                           2, 3, 1-im, 1+im, 6, 4, 2-im, 8])
    end
    As_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, (n,n))
    solver = ldlt(As_gpu)

    if T <: AbstractFloat
      Bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,  -7, -12, -25, -4, -13,
                            13, 15, 29, 8, 14, -13, -15, -29, -8, -14])
    else
      Bs_gpu = CuVector{T}([ 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im,
                            13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im])
    end
    Xs_gpu = CuVector{T}(undef, n * nrhs * nbatch)
    ldiv!(Xs_gpu, solver, Bs_gpu)

    Rs_gpu = rand(R, nbatch)
    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        B_gpu = reshape(Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        R_gpu = B_gpu - A_gpu * X_gpu
        Rs_gpu[i] = norm(R_gpu)
    end
    @test norm(Rs_gpu) ≤ √eps(R)

    Bs2_gpu = reshape(copy(Bs_gpu), n, nrhs, nbatch)
    Xs2_gpu = reshape(copy(Xs_gpu), n, nrhs, nbatch)
    ldiv!(Xs2_gpu, solver, Bs2_gpu)

    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        B_gpu = Bs2_gpu[:, :, i]
        X_gpu = Xs2_gpu[:, :, i]
        R_gpu = B_gpu - A_gpu * X_gpu
        Rs_gpu[i] = norm(R_gpu)
    end
    @test norm(Rs_gpu) ≤ √eps(R)

    if T <: AbstractFloat
      new_nzVal = CuVector{T}([-4, -3,  1, -2, -5, -1, -1, -2,
                               -2, -3, -1, -1, -6, -4, -2, -8])
    else
      new_nzVal = CuVector{T}([-4, -3,  1-im, -2+im, -5, -1, -1-im, -2,
                               -2, -3, -1+im, -1-im, -6, -4, -2+im, -8])
    end
    As_gpu.nzVal = new_nzVal
    ldlt!(solver, As_gpu)

    if T <: AbstractFloat
      new_Bs_gpu = CuVector{T}([13, 15, 29, 8, 14, -13, -15, -29, -8, -14,
                                 7, 12, 25, 4, 13,  -7, -12, -25, -4, -13])
    else
      new_Bs_gpu = CuVector{T}([13-im, 15-im, 29-im, 8-im, 14-im, -13-im, -15-im, -29-im, -8-im, -14-im,
                                 7+im, 12+im, 25+im, 4+im, 13+im,  -7+im, -12+im, -25+im, -4+im, -13+im])
    end

    Xs_gpu .= Bs_gpu
    ldiv!(solver, Xs_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        B_gpu = reshape(new_Bs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        X_gpu = reshape(Xs_gpu[1 + (i-1) * n * nrhs : i * n * nrhs], n, nrhs)
        R_gpu = B_gpu - A_gpu * X_gpu
        Rs_gpu[i] = norm(R_gpu)
    end
    @test norm(Rs_gpu) ≤ √eps(R)

    new_Bs2_gpu = reshape(new_Bs_gpu, n, nrhs, nbatch)
    new_Xs2_gpu = copy(new_Bs2_gpu)
    ldiv!(solver, Xs2_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        B_gpu = new_Bs2_gpu[:, :, i]
        X_gpu = new_Xs2_gpu[:, :, i]
        R_gpu = B_gpu - A_gpu * X_gpu
        Rs_gpu[i] = norm(R_gpu)
    end
    @test norm(Rs_gpu) ≤ √eps(R)
  end
end

function cudss_uniform_batch_cholesky()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    n = 5
    nbatch = 2
    nnzA = 8
    rowPtr = CuVector{Cint}([1, 3, 5, 7, 8, 9])
    colVal = CuVector{Cint}([1, 3, 2, 3, 3, 5, 4, 5])
    nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                         2, 1, 3, 1, 6, 2, 4, 8])

    cudss_bs_gpu = CudssMatrix(T, n; nbatch)
    bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,
                          13, 15, 29, 8, 14])
    cudss_update(cudss_bs_gpu, bs_gpu)

    cudss_xs_gpu = CudssMatrix(T, n; nbatch)
    xs_gpu = CuVector{T}(undef, n * nbatch)
    cudss_update(cudss_xs_gpu, xs_gpu)

    # Constructor for uniform batch of systems
    solver = CudssSolver(rowPtr, colVal, nzVal, "SPD", 'U')

    # Specify that it is a uniform batch of size "nbatch"
    cudss_set(solver, "ubatch_size", nbatch)

    cudss("analysis", solver, cudss_xs_gpu, cudss_bs_gpu)
    cudss("factorization", solver, cudss_xs_gpu, cudss_bs_gpu)
    cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu)

    rs_gpu = rand(R, nbatch)
    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        b_gpu = bs_gpu[1 + (i-1) * n : i * n]
        x_gpu = xs_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rs_gpu[i] = norm(r_gpu)
    end
    @test norm(rs_gpu) ≤ √eps(R)

    new_nzVal = CuVector{T}([8, 2, 6, 4, 10, 2,  2,  4,
                             6, 3, 9, 3, 18, 6, 12, 24])
    cudss_update(solver, rowPtr, colVal, new_nzVal)
    cudss("refactorization", solver, cudss_xs_gpu, cudss_bs_gpu)
    cudss("solve", solver, cudss_xs_gpu, cudss_bs_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        b_gpu = bs_gpu[1 + (i-1) * n : i * n]
        x_gpu = xs_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rs_gpu[i] = norm(r_gpu)
    end
    @test norm(rs_gpu) ≤ √eps(R)
  end
end

function generic_uniform_batch_cholesky()
  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    R = real(T)
    n = 5
    nbatch = 2
    nnzA = 8
    rowPtr = CuVector{Cint}([1, 3, 5, 7, 8, 9])
    colVal = CuVector{Cint}([1, 3, 2, 3, 3, 5, 4, 5])
    nzVal = CuVector{T}([4, 1, 3, 2, 5, 1, 1, 2,
                         2, 1, 3, 1, 6, 2, 4, 8])

    As_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nzVal, (n,n))
    solver = cholesky(As_gpu)

    bs_gpu = CuVector{T}([ 7, 12, 25, 4, 13,
                          13, 15, 29, 8, 14])
    xs_gpu = CuVector{T}(undef, n * nbatch)
    ldiv!(xs_gpu, solver, bs_gpu)

    rs_gpu = rand(R, nbatch)
    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        b_gpu = bs_gpu[1 + (i-1) * n : i * n]
        x_gpu = xs_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rs_gpu[i] = norm(r_gpu)
    end
    @test norm(rs_gpu) ≤ √eps(R)

    bs2_gpu = CuMatrix{T}([ 7 13;
                           12 15;
                           25 29;
                            4  8;
                           13 14])
    xs2_gpu = CuMatrix{T}(undef, n, nbatch)
    ldiv!(xs2_gpu, solver, bs2_gpu)

    for i = 1:nbatch
        nz = nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        b_gpu = bs2_gpu[:, i]
        x_gpu = xs2_gpu[:, i]
        r_gpu = b_gpu - A_gpu * x_gpu
        rs_gpu[i] = norm(r_gpu)
    end
    @test norm(rs_gpu) ≤ √eps(R)

    new_nzVal = CuVector{T}([8, 2, 6, 4, 10, 2,  2,  4,
                             6, 3, 9, 3, 18, 6, 12, 24])
    As_gpu.nzVal = new_nzVal
    cholesky!(solver, As_gpu)

    xs_gpu .= bs_gpu
    ldiv!(solver, xs_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        b_gpu = bs_gpu[1 + (i-1) * n : i * n]
        x_gpu = xs_gpu[1 + (i-1) * n : i * n]
        r_gpu = b_gpu - A_gpu * x_gpu
        rs_gpu[i] = norm(r_gpu)
    end
    @test norm(rs_gpu) ≤ √eps(R)

    xs2_gpu .= bs2_gpu
    ldiv!(solver, xs2_gpu)

    for i = 1:nbatch
        nz = new_nzVal[1 + (i-1) * nnzA : i * nnzA]
        A_gpu = CuSparseMatrixCSR{T,Cint}(rowPtr, colVal, nz, (n,n))
        A_cpu = SparseMatrixCSC(A_gpu)
        A_gpu = CuSparseMatrixCSR(A_cpu + A_cpu' - Diagonal(A_cpu))
        b_gpu = bs2_gpu[:, i]
        x_gpu = xs2_gpu[:, i]
        r_gpu = b_gpu - A_gpu * x_gpu
        rs_gpu[i] = norm(r_gpu)
    end
    @test norm(rs_gpu) ≤ √eps(R)
  end
end

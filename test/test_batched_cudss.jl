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

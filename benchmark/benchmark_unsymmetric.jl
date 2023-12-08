using LinearAlgebra
using SparseArrays

using CUDA, CUDA.CUSPARSE
using KLU
using CUDSS
using CUSOLVERRF

using MatrixMarket

names = ["case9", "case57", "case118", "case300", "case1354pegase",
         "case_ACTIVSg2000", "case2869pegase", "case9241pegase",
         "case13659pegase", "case_ACTIVSg10k", "case_ACTIVSg25k", "case_ACTIVSg70k"]

const DATA_UNSYMMETRIC = joinpath(@__DIR__, "data", "unsymmetric")

for name in names
    println("Problem : ", name)

    A_cpu = mmread(joinpath(DATA_UNSYMMETRIC, "Gx_$(name).mtx"))
    B_cpu = mmread(joinpath(DATA_UNSYMMETRIC, "Gu_$(name).mtx"))
    B_cpu = Matrix(B_cpu)

    m,n = size(A_cpu)
    p = size(B_cpu, 2)
    X_cpu = zeros(n, p)

    # CPU
    println("UMFPACK")
    F_umfpack = lu(A_cpu)
    timer_factorization = @elapsed lu!(F_umfpack, A_cpu)
    timer_solve = @elapsed ldiv!(X_cpu, F_umfpack, B_cpu)
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println()

    println("KLU")
    F_klu = klu(A_cpu)
    timer_factorization = @elapsed klu!(F_klu, A_cpu)
    timer_solve = @elapsed KLU.ldiv!(X_cpu, F_klu, B_cpu)
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println()

    A_gpu = CuSparseMatrixCSR(A_cpu)
    B_gpu = CuMatrix(B_cpu)
    X_gpu = CuMatrix(X_cpu)

    # GPU
    if name ≠ "case_ACTIVSg70k"
        println("CUSOLVERRF")
        F_cusolverrf = CUSOLVERRF.RFLU(A_gpu; nrhs=p, symbolic=:KLU)
        timer_factorization = CUDA.@elapsed CUDA.@sync lu!(F_cusolverrf, A_gpu)
        timer_solve = CUDA.@elapsed CUDA.@sync ldiv!(X_gpu, F_cusolverrf, B_gpu)
        println("timer_factorization : ", timer_factorization, " seconds")
        println("timer_solve : ", timer_solve, " seconds")
        println()
    end

    println("CUDSS")
    solver = CudssSolver(A_gpu, 'G', 'F')
    timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, X_gpu, B_gpu)
    timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, X_gpu, B_gpu)
    timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, X_gpu, B_gpu)
    println("timer_analyis : ", timer_analyis, " seconds")
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println()
end

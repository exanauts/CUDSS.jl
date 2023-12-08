using CUDA, CUDA.CUSPARSE
using CUDSS, CUSOLVERRF
using SparseArrays
using LinearAlgebra

using DelimitedFiles
using MatrixMarket

names = ["case9", "case57", "case118", "case300", "case1354pegase",
         "case_ACTIVSg2000", "case2869pegase", "case9241pegase",
         "case13659pegase", "case_ACTIVSg10k", "case_ACTIVSg25k", "case_ACTIVSg70k"]

const DATA_UNSYMMETRIC = joinpath(@__DIR__, "data", "unsymmetric")

global first_run = true

function cusolverrf(A_gpu, b_gpu)
    n, p = size(b_gpu)
    rf = CUSOLVERRF.RFLU(A_gpu; nrhs=p, symbolic=:RF)
    ldiv!(rf, b_gpu)
end

for name in names
    A_cpu = mmread(joinpath(DATA_UNSYMMETRIC, "Gx_$(name).mtx"))
    B_cpu = mmread(joinpath(DATA_UNSYMMETRIC, "Gu_$(name).mtx"))
    B_cpu = Matrix(B_cpu)

    A_gpu = CuSparseMatrixCSR(A_cpu)
    B_gpu = CuMatrix(B_cpu)

    m,n = size(A_gpu)
    p = size(B_gpu, 2)
    X_cpu = zeros(n, p)
    X_gpu = CuMatrix(X_cpu)

    solver = CudssSolver(A_gpu, 'G', 'F')

    if first_run
        timer_analyis = cudss("analysis", solver, X_gpu, B_gpu)
        timer_factorization = cudss("factorization", solver, X_gpu, B_gpu)
        timer_solve = cudss("solve", solver, X_gpu, B_gpu)
        global first_run = false
    end

    timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, X_gpu, B_gpu)
    timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, X_gpu, B_gpu)
    timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, X_gpu, B_gpu)

    R_gpu = B_gpu - A_gpu * X_gpu
    RNorm = norm(R_gpu)
    println("Problem : ", name)
    println("timer_analyis : ", timer_analyis, " seconds")
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println("‖B - AX‖ : ", RNorm)
    println()
    println("timer CUDSS : ", timer_analyis + timer_factorization + timer_solve, " seconds")
    (problem != "case_ACTIVSg70k") && (timer_cusolverrf = CUDA.@elapsed CUDA.@sync cusolverrf(A_gpu, B_gpu))
    println("timer CUSOLVERRF : ", timer_cusolverrf, " seconds")
    println()
end

const DATA_SYMMETRIC = joinpath(@__DIR__, "data", "symmetric")
stats = readdlm(joinpath(DATA_SYMMETRIC, "statistics.txt"), '\t', Int)

all_cases = ["case118", "case1354pegase", "case2869pegase"]

for case in all_cases
    idx = findfirst(isequal(case), all_cases)
    ipm_it = 5

    # Number of variables, constraints, states, controls and inequalities.
    nvar, ncon, nx, nu, nineq = stats[idx, :]

    Kaug = mmread(joinpath(DATA_SYMMETRIC, "$(case)_$(ipm_it).mtx"))
    rhs = readdlm(joinpath(DATA_SYMMETRIC, "$(case)_$(ipm_it)_rhs.txt"))[:]
    baug = rhs[1:nvar+nineq+ncon]

    A_gpu = CuSparseMatrixCSR(Kaug)
    b_gpu = CuVector(baug)

    m,n = size(A_gpu)
    x_cpu = zeros(n)
    x_gpu = CuVector(x_cpu)

    solver = CudssSolver(A_gpu, 'S', 'F')

    timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, x_gpu, b_gpu)
    timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, x_gpu, b_gpu)
    timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    rNorm = norm(r_gpu)
    println("Problem : ", name, " -- Augmented system")
    println("timer_analyis : ", timer_analyis, " seconds")
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println("‖b - Ax‖ : ", rNorm)
    println()

    r1 = rhs[1:nvar]
    r2 = rhs[nvar+1:nvar+nineq]
    r3 = rhs[nvar+nineq+1:nvar+nineq+nx]
    r4 = rhs[nvar+nineq+nx+1:nvar+nineq+nx+nineq]

    # Hessian of Lagrangian.
    W = Kaug[1:nvar, 1:nvar]
    # Slack regularization.
    Σ = Kaug[nvar+1:nvar+nineq, nvar+1:nvar+nineq]
    # Jacobian of equality constraints.
    G = Kaug[nvar+nineq+1:nvar+nineq+nx, 1:nvar]
    # Jacobian of inequality constraints.
    H = Kaug[nvar+nineq+nx+1:nvar+nineq+nx+nineq, 1:nvar]

    # Condensed matrix
    K = Symmetric(W, :L) + H' * Σ * H

    # Condensed KKT system.
    Kcond = [
        K G'
        G spzeros(nx, nx)
    ]
    bcond = vcat(r1 + H' * (Σ * r4 - r2), r3)

    A_gpu = CuSparseMatrixCSR(Kcond)
    b_gpu = CuVector(bcond)

    m,n = size(A_gpu)
    x_cpu = zeros(n)
    x_gpu = CuVector(x_cpu)

    solver = CudssSolver(A_gpu, 'S', 'F')

    timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, x_gpu, b_gpu)
    timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, x_gpu, b_gpu)
    timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    rNorm = norm(r_gpu)
    println("Problem : ", name, " -- Condensed system")
    println("timer_analyis : ", timer_analyis, " seconds")
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println("‖b - Ax‖ : ", rNorm)
    println()

    # Solve the Schur complement based on the Kcond
    K2 = Matrix(K)
    inv_K = inv(K2)
    Kschur = G * inv_K * G'
    bschur = G * inv_K * bcond[1:nvar] - bcond[nvar+1:nvar+nx]

    A_gpu = CuSparseMatrixCSR(Kcond + 10^8*I)
    b_gpu = CuVector(bcond)

    m,n = size(A_gpu)
    x_cpu = zeros(n)
    x_gpu = CuVector(x_cpu)

    solver = CudssSolver(A_gpu, "SPD", 'F')

    timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, x_gpu, b_gpu)
    timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, x_gpu, b_gpu)
    timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    rNorm = norm(r_gpu)
    println("Problem : ", name, " -- Schur complement system")
    println("timer_analyis : ", timer_analyis, " seconds")
    println("timer_factorization : ", timer_factorization, " seconds")
    println("timer_solve : ", timer_solve, " seconds")
    println("‖b - Ax‖ : ", rNorm)
    println()
end

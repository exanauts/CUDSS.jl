using CUDA, CUDA.CUSPARSE
using CUDSS, CUSOLVERRF
using SparseArrays
using LinearAlgebra

using DelimitedFiles
using MatrixMarket

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

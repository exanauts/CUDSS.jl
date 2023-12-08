using CUDA, CUDA.CUSPARSE
using CUDSS, CUSOLVERRF
using SparseArrays
using LinearAlgebra

using MatrixMarket
using DelimitedFiles

const DATA_SYMMETRIC = joinpath(@__DIR__, "data", "symmetric")
stats = readdlm(joinpath(DATA_SYMMETRIC, "statistics.txt"), '\t', Int)

all_cases = ["case118", "case1354pegase", "case2869pegase"]
nipm_it = Dict("case118" => 15, "case1354pegase" => 50, "case2869pegase" => 60)

# path_aug = joinpath(@__DIR__, "data", "symmetric", "augmented_systems")
path_cond = joinpath(@__DIR__, "data", "symmetric", "condensed_systems")

save_data = true
if save_data
    isdir(path_aug) || mkdir(path_aug)
    isdir(path_cond) || mkdir(path_cond)
end

for case in all_cases
    for ipm_it = 5:5:nipm_it[case]
        # Number of variables, constraints, states, controls and inequalities.
        idx = findfirst(isequal(case), all_cases)
        nvar, ncon, nx, nu, nineq = stats[idx, :]

        Kaug = mmread(joinpath(DATA_SYMMETRIC, "$(case)_$(ipm_it).mtx"))
        rhs = readdlm(joinpath(DATA_SYMMETRIC, "$(case)_$(ipm_it)_rhs.txt"))[:]
        baug = rhs[1:nvar+nineq+ncon]
        # xstar = Kaug \ baug

        # A_gpu = CuSparseMatrixCSR(Kaug)
        # b_gpu = CuVector(baug)

        # m,n = size(A_gpu)
        # x_cpu = zeros(n)
        # x_gpu = CuVector(x_cpu)

        # solver = CudssSolver(A_gpu, 'S', 'F')

        # timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, x_gpu, b_gpu)
        # timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, x_gpu, b_gpu)
        # timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, x_gpu, b_gpu)

        # if save_data
        #     folder = joinpath(path_aug, "$(case)_$(ipm_it)")
        #     isdir(folder) || mkdir(folder)
        #     writedlm(joinpath(folder, "A_rowptr.txt"), Vector(A_gpu.rowPtr))
        #     writedlm(joinpath(folder, "A_colval.txt"), Vector(A_gpu.colVal))
        #     writedlm(joinpath(folder, "A_nzval.txt"), Vector(A_gpu.nzVal))
        #     writedlm(joinpath(folder, "b.txt"), Vector(b_gpu))
        #     writedlm(joinpath(folder, "x.txt"), Vector(x_gpu))
        #     writedlm(joinpath(folder, "xstar.txt"), xstar)
        #     raug = baug - Kaug * xstar
        #     println(norm(raug))
        # end

        # r_gpu = b_gpu - A_gpu * x_gpu
        # rNorm = norm(r_gpu)
        # println("Problem : ", name, " -- Augmented system -- ipm_it : ", ipm_it)
        # println("timer_analyis : ", timer_analyis, " seconds")
        # println("timer_factorization : ", timer_factorization, " seconds")
        # println("timer_solve : ", timer_solve, " seconds")
        # println("‖b - Ax‖ : ", rNorm)
        # println()

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
        xstar = Kcond \ bcond

        A_gpu = CuSparseMatrixCSR(Kcond)
        b_gpu = CuVector(bcond)

        m,n = size(A_gpu)
        x_cpu = zeros(n)
        x_gpu = CuVector(x_cpu)

        solver = CudssSolver(A_gpu, 'S', 'F')

        timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, x_gpu, b_gpu)
        timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, x_gpu, b_gpu)
        timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, x_gpu, b_gpu)

        if save_data
            folder = joinpath(path_cond, "$(case)_$(ipm_it)")
            isdir(folder) || mkdir(folder)
            writedlm(joinpath(folder, "A_rowptr.txt"), Vector(A_gpu.rowPtr))
            writedlm(joinpath(folder, "A_colval.txt"), Vector(A_gpu.colVal))
            writedlm(joinpath(folder, "A_nzval.txt"), Vector(A_gpu.nzVal))
            writedlm(joinpath(folder, "b.txt"), Vector(b_gpu))
            writedlm(joinpath(folder, "x.txt"), Vector(x_gpu))
            writedlm(joinpath(folder, "xstar.txt"), xstar)
            rcond = bcond - Kcond * xstar
            println(norm(rcond))
        end

        r_gpu = b_gpu - A_gpu * x_gpu
        rNorm = norm(r_gpu)
        println("Problem : ", name, " -- Condensed system -- ipm_it : ", ipm_it)
        println("timer_analyis : ", timer_analyis, " seconds")
        println("timer_factorization : ", timer_factorization, " seconds")
        println("timer_solve : ", timer_solve, " seconds")
        println("‖b - Ax‖ : ", rNorm)
        println()

        # Solve the Schur complement based on the Kcond
        # K2 = Matrix(K)
        # inv_K = inv(K2)
        # Kschur = G * inv_K * G'
        # bschur = G * inv_K * bcond[1:nvar] - bcond[nvar+1:nvar+nx]

        # A_gpu = CuSparseMatrixCSR(Kcond + 10^8*I)
        # b_gpu = CuVector(bcond)

        # m,n = size(A_gpu)
        # x_cpu = zeros(n)
        # x_gpu = CuVector(x_cpu)

        # solver = CudssSolver(A_gpu, "SPD", 'F')

        # timer_analyis = CUDA.@elapsed CUDA.@sync cudss("analysis", solver, x_gpu, b_gpu)
        # timer_factorization = CUDA.@elapsed CUDA.@sync cudss("factorization", solver, x_gpu, b_gpu)
        # timer_solve = CUDA.@elapsed CUDA.@sync cudss("solve", solver, x_gpu, b_gpu)

        # r_gpu = b_gpu - A_gpu * x_gpu
        # rNorm = norm(r_gpu)
        # println("Problem : ", name, " -- Schur complement system")
        # println("timer_analyis : ", timer_analyis, " seconds")
        # println("timer_factorization : ", timer_factorization, " seconds")
        # println("timer_solve : ", timer_solve, " seconds")
        # println("‖b - Ax‖ : ", rNorm)
        # println()
    end
end

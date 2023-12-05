using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays
using LinearAlgebra
using MatrixMarket

names = ["case9", "case57", "case118", "case300", "case1354pegase",
         "case_ACTIVSg2000", "case2869pegase", "case9241pegase",
         "case13659pegase", "case_ACTIVSg10k", "case_ACTIVSg25k", "case_ACTIVSg70k"]

const DATA_DIR = joinpath(@__DIR__, "data", "unsymmetric")

for name in names
	A_cpu = mmread(joinpath(DATA_DIR, "Gx_$(name).mtx"))
	B_cpu = mmread(joinpath(DATA_DIR, "Gu_$(name).mtx"))
	B_cpu = Matrix(B_cpu)

	A_gpu = CuSparseMatrixCSR(A_cpu)
	B_gpu = CuMatrix(B_cpu)

	m,n = size(A_gpu)
	p = size(B_gpu, 2)
	X_cpu = zeros(n, p)
	X_gpu = CuMatrix(X_cpu)

	solver = CudssSolver(A_gpu, 'G', 'F')

	cudss("analysis", solver, X_gpu, B_gpu)
	cudss("factorization", solver, X_gpu, B_gpu)
	cudss("solve", solver, X_gpu, B_gpu)

	R_gpu = B_gpu - A_gpu * X_gpu
	RNorm = norm(R_gpu)
	println("Problem : ", name)
	println("‖B - AX‖ : ", RNorm)
	println()
end

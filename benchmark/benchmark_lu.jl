

using LinearAlgebra
using SparseArrays

using MatrixMarket
using CUDA
using CUDA.CUSPARSE

using KLU
using CUDSS
using CUSOLVERRF


const DATA_UNSYMMETRIC = joinpath(@__DIR__, "data", "unsymmetric")

case = "Gx_case_ACTIVSg25k.mtx"
datafile = joinpath(DATA_UNSYMMETRIC, case)

# Instantiate data
Gx = mmread(datafile)
n = size(Gx, 1)
# Create a new matrix for refactorization
Gx2 = deepcopy(Gx)
nonzeros(Gx2) .*= 2.0
# Create left and right-hand-side
x_cpu = zeros(n)
b_cpu = rand(n)

@info "Benchmark case $(case)"

#=
    CPU
=#
@info "UMFPACK"
umfpack_fac = lu(Gx)
print("Factorization: ")
@time lu!(umfpack_fac, Gx2)
print("Backsolve:     ")
@time ldiv!(x_cpu, umfpack_fac, b_cpu)

@info "KLU"
klu_fac = klu(Gx)
print("Factorization: ")
@time klu!(klu_fac, Gx2)
print("Backsolve:     ")
@time ldiv!(x_cpu, umfpack_fac, b_cpu)

#=
    GPU
=#

# Move data to device.
dGx = CuSparseMatrixCSR(Gx)
dGx2 = CuSparseMatrixCSR(Gx2)
x_gpu = CuVector(x_cpu)
b_gpu = CuVector(b_cpu)

@info "CUSOLVERRF"
cusolverrf_fac = CUSOLVERRF.RFLU(dGx; symbolic=:KLU)
print("Factorization: ")
CUDA.@time lu!(cusolverrf_fac, dGx2)
print("Backsolve:     ")
adum = CUDA.zeros(1)
# WARNING: the time can be inexact for cusolverrf's backsolve,
# even after explicitly calling CUDA.synchronize()
CUDA.@time begin
    ldiv!(x_gpu, cusolverrf_fac, b_gpu)
end


@info "CUDSS"
mat = CudssMatrix(dGx, 'G', 'F')
config = CudssConfig()
data = CudssData()
solver = CudssSolver(mat, config, data)

# Initial solve
cudss("analysis", solver, x_gpu, b_gpu)

cudss_set(solver.matrix, nonzeros(dGx2))
print("Factorization: ")
CUDA.@time cudss("factorization", solver, x_gpu, b_gpu)
print("Backsolve:     ")
CUDA.@time cudss("solve", solver, x_gpu, b_gpu)


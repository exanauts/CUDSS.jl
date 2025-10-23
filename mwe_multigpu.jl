using CUDA, CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays
using Adapt
using BenchmarkTools
using KernelAbstractions

# Utility functions
cpu(x) = adapt(CPU(), x)
gpu(x) = adapt(CUDABackend(), x)

function create_matrix(T, n, density)
    A_cpu = sprand(T, n, n, density)
    A_cpu = A_cpu + A_cpu' + I
    A_gpu = CuSparseMatrixCSR(gpu(A_cpu.colptr), gpu(A_cpu.rowval), gpu(A_cpu.nzval), size(A_cpu))
    GB = nnz(A_cpu) * (sizeof(T) + sizeof(Int32)) / (1024^3)
    println("Creating matrix of size $(n)×$(n) with density $(density) and $(nnz(A_cpu)) non-zeros.")
    println("Approximate size of the matrix in GPU memory: $(round(GB, digits=2)) GB")
    return A_gpu
end

function create_multigpu_solver(A::T, device_indices::Vector{Cint}) where {T <: AbstractSparseMatrix}
    # Configure devices for this task
    CUDSS.devices!(device_indices)

    # Create multi-GPU objects
    data = CudssData(device_indices)
    config = CudssConfig(device_indices)
    matrix = CudssMatrix(A, "S", 'F')
    solver = CudssSolver(matrix, config, data)
    return solver
end

function analysis!(solver, x, b)
    cudss("analysis", solver, x, b)
    # Note: Synchronization is now handled automatically for multi-GPU operations
end

function factorization!(solver, x, b)
    cudss("factorization", solver, x, b)
    # Note: Synchronization is now handled automatically for multi-GPU operations
end

function solve!(solver, x, b)
    cudss("solve", solver, x, b)
    # Note: Synchronization is now handled automatically for multi-GPU operations
end

# Main execution
println("\n" * "="^80)
println("Multi-GPU Linear System Solver Example")
println("="^80)

# Detect available GPUs
device_count = CUDA.ndevices()
println("\nDetected $device_count CUDA device$(device_count > 1 ? "s" : ""):")
for i in 0:device_count-1
    CUDA.device!(i)
    dev = CUDA.device()
    props = CUDA.name(dev)
    mem_gb = CUDA.totalmem(dev) / (1024^3)
    println("  Device $i: $props ($(round(mem_gb, digits=2)) GB)")
end

# Configure device indices
# Note: cuDSS allows using the same device multiple times for testing
# Here we demonstrate using all available devices
device_indices = collect(Cint(0):Cint(device_count-1))
println("\n" * "="^80)
println("Configuration: Using $(length(device_indices)) GPU device(s)")
println("Device indices: $device_indices")
println("="^80)

# Create test problem
T = Float64
n = 100  # Matrix size
density = 0.01  # Sparsity pattern

println("\n" * "-"^80)
println("Problem Setup")
println("-"^80)
A = create_matrix(T, n, density)

# Create multi-GPU solver
println("\n" * "-"^80)
println("Creating Multi-GPU Solver")
println("-"^80)
@time solver = create_multigpu_solver(A, device_indices)
println("Solver created successfully using $(CUDSS.ndevices()) device(s)")

# Create right-hand side and solution vectors
b = gpu(rand(T, n))
x = similar(b)

# Analysis phase
println("\n" * "-"^80)
println("Analysis Phase")
println("-"^80)
@time analysis!(solver, x, b)

# Factorization phase
println("\n" * "-"^80)
println("Factorization Phase")
println("-"^80)
@time factorization!(solver, x, b)

# Solve phase
println("\n" * "-"^80)
println("Solve Phase")
println("-"^80)
@time solve!(solver, x, b)

# Verify solution
println("\n" * "-"^80)
println("Solution Verification")
println("-"^80)

# Check on CPU
r_cpu = cpu(b) - cpu(A) * cpu(x)
residual_norm_cpu = norm(r_cpu)
println("Residual norm ||b - A*x|| (CPU): $(residual_norm_cpu)")

# Check on GPU
r_gpu = b - A * x
residual_norm_gpu = norm(r_gpu)
println("Residual norm ||b - A*x|| (GPU): $(residual_norm_gpu)")

relative_error = residual_norm_gpu / norm(b)
println("Relative error: $(relative_error)")

if relative_error < 1e-10
    println("✓ Solution verified successfully!")
else
    println("⚠ Warning: Residual is larger than expected")
end

println("\n" * "="^80)
println("Multi-GPU Example Completed Successfully")
println("="^80)

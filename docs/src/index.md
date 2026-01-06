# [CUDSS.jl documentation](@id Home)

## Overview

[CUDSS.jl](https://github.com/exanauts/CUDSS.jl) is a Julia interface to the NVIDIA [cuDSS](https://developer.nvidia.com/cudss) library.
NVIDIA cuDSS provides three factorizations (LDU, LDLᵀ, LLᵀ) for solving sparse linear systems on GPUs.
For more details on using cuDSS, refer to the official [cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html).

## Installation

```julia
julia> ]
pkg> add CUDSS
pkg> test CUDSS
```

## CUDA.jl Integration

CUDSS.jl is designed to integrate seamlessly with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)'s task-based concurrency model.

### Task-Based Stream Management

CUDA.jl automatically creates **one CUDA stream per Julia task per device**. CUDSS.jl follows this model:

- Each Julia `Task` that uses CUDSS gets its own CUDSS handle
- CUDSS operations automatically execute on the current task's stream
- No manual stream management is required for concurrent CUDSS usage

This means you can safely run multiple independent CUDSS solvers concurrently using Julia's task-based parallelism:

```julia
using CUDA, CUDA.CUSPARSE
using CUDSS

# Create sparse matrices for two independent systems
A1_gpu = CuSparseMatrixCSR(A1_cpu)
A2_gpu = CuSparseMatrixCSR(A2_cpu)

# Solve concurrently - each task uses its own stream and CUDSS handle
@sync begin
    Threads.@spawn begin
        solver1 = CudssSolver(A1_gpu, "SPD", 'L')
        cudss("analysis", solver1, x1_gpu, b1_gpu)
        cudss("factorization", solver1, x1_gpu, b1_gpu)
        cudss("solve", solver1, x1_gpu, b1_gpu)
    end

    Threads.@spawn begin
        solver2 = CudssSolver(A2_gpu, "SPD", 'L')
        cudss("analysis", solver2, x2_gpu, b2_gpu)
        cudss("factorization", solver2, x2_gpu, b2_gpu)
        cudss("solve", solver2, x2_gpu, b2_gpu)
    end
end
```

### Stream and Context Compatibility

CUDSS.jl respects CUDA.jl's context and stream management functions:

- **`CUDA.stream!()`**: CUDSS operations will automatically use the new stream
- **`CUDA.device!()`**: CUDSS will use the appropriate device-specific handle
- **`CUDA.context!()`**: CUDSS maintains separate handles per context

This integration follows the same pattern used by CUDA.jl's built-in libraries (CUBLAS, CUSPARSE, CUSOLVER), ensuring consistent behavior across all GPU operations in your application.

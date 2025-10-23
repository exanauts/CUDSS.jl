using Test, Random
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays
using LinearAlgebra

import CUDSS: CUDSS_DATA_PARAMETERS, CUDSS_CONFIG_PARAMETERS

@info("CUDSS_INSTALLATION : $(CUDSS.CUDSS_INSTALLATION)")

Random.seed!(666)  # Random tests are diabolical

include("test_cudss.jl")
include("test_batched_cudss.jl")
include("test_cudss_mg.jl")

@testset "CUDSS" begin
  @testset "version" begin
    cudss_version()
  end

  @testset "CudssMatrix" begin
    cudss_dense()
    cudss_sparse()
  end

  @testset "CudssData" begin
    # Issue #1
    data = CudssData()
  end

 @testset "CudssSolver" begin
    cudss_solver()
  end

  @testset "CudssExecution" begin
    cudss_execution()
  end

  @testset "Generic API" begin
    cudss_generic()
  end

  @testset "User permutation" begin
    user_permutation()
  end

  @testset "Iterative refinement" begin
    iterative_refinement()
  end

  @testset "Small matrices" begin
    small_matrices()
  end

  @testset "Hybrid memory mode" begin
    hybrid_memory_mode()
  end

  @testset "Refactorization Cholesky" begin
    refactorization_cholesky()
  end
end

@testset "Batched CUDSS" begin
  @testset "CudssBatchedMatrix" begin
    cudss_batched_dense()
    cudss_batched_sparse()
  end

 @testset "CudssBatchedSolver" begin
    cudss_batched_solver()
  end

  @testset "CudssExecution -- batched" begin
    cudss_batched_execution()
  end

  @testset "Hybrid memory mode -- batched" begin
    batched_hybrid_memory_mode()
  end

  @testset "Refactorization Cholesky -- batched" begin
    refactorization_batched_cholesky()
  end
end

@testset "Multi-GPU CUDSS" begin
  @testset "Device Management" begin
    cudss_mg_device_management()
  end

  @testset "Data Creation" begin
    cudss_mg_data_creation()
  end

  @testset "Solver" begin
    cudss_mg_solver()
  end

  @testset "Multiple RHS" begin
    cudss_mg_multiple_rhs()
  end

  @testset "Refactorization" begin
    cudss_mg_refactorization()
  end

  @testset "Views" begin
    cudss_mg_views()
  end

  @testset "Task Isolation" begin
    cudss_mg_task_isolation()
  end
end

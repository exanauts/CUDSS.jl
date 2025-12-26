using Test, Random
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using CUDSS
using SparseArrays
using LinearAlgebra

import CUDSS: CUDSS_DATA_PARAMETERS, CUDSS_CONFIG_PARAMETERS

@info("CUDSS_INSTALLATION : $(CUDSS.CUDSS_INSTALLATION)")

Random.seed!(666)  # Random tests are diabolical

# Needed for the tests with Int64
function CuSparseMatrixCSR{T,INT}(A::CuSparseMatrixCSR) where {T,INT}
  CuSparseMatrixCSR{T,INT}(CuVector{INT}(A.rowPtr), CuVector{INT}(A.colVal), CuVector{T}(A.nzVal), A.dims)
end

function CuSparseMatrixCSR{T,INT}(A::SparseMatrixCSC) where {T,INT}
  B = CuSparseMatrixCSC(A)
  C = CuSparseMatrixCSR(B)
  CuSparseMatrixCSR{T,INT}(CuVector{INT}(C.rowPtr), CuVector{INT}(C.colVal), CuVector{T}(C.nzVal), C.dims)
end

include("test_cudss.jl")
include("test_schur_cudss.jl")
include("test_uniform_batch_cudss.jl")
include("test_nonuniform_batch_cudss.jl")

@testset "Schur complement CUDSS" begin
  @testset "Schur complement LU" begin
    cudss_schur_lu()
  end

  @testset "Schur complement LDL" begin
    cudss_schur_ldlt()
  end

  @testset "Schur complement Cholesky" begin
    cudss_schur_cholesky()
  end
end

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

@testset "Uniform batch CUDSS" begin
  @testset "cuDSS API" begin
    @testset "Uniform batch LU" begin
      cudss_uniform_batch_lu()
    end

    @testset "Uniform batch LDL" begin
      cudss_uniform_batch_ldlt()
    end

    @testset "Uniform batch Cholesky" begin
      cudss_uniform_batch_cholesky()
    end
  end

  @testset "Generic API" begin
    @testset "Uniform batch LU" begin
      generic_uniform_batch_lu()
    end

    @testset "Uniform batch LDL" begin
      generic_uniform_batch_ldlt()
    end

    @testset "Uniform batch Cholesky" begin
      generic_uniform_batch_cholesky()
    end
  end
end

@testset "Non-uniform batch CUDSS" begin
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

using Test, Random
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays
using LinearAlgebra

import CUDSS: CUDSS_DATA_PARAMETERS, CUDSS_CONFIG_PARAMETERS

@info("CUDSS_INSTALLATION : $(CUDSS.CUDSS_INSTALLATION)")

Random.seed!(666)  # Random tests are diabolical

include("test_cudss.jl")

@testset "CUDSS" begin
  @testset "version" begin
    cudss_version()
  end

  @testset "CudssMatrix" begin
    cudss_dense()
    cudss_sparse()
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
end

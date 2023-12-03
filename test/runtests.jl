using Test
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays
using LinearAlgebra

import CUDSS: CUDSS_DATA_PARAMETERS, CUDSS_CONFIG_PARAMETERS
import CUDSS: CUDSS_ALG_DEFAULT, CUDSS_ALG_1, CUDSS_ALG_2, CUDSS_ALG_3

include("test_cudss.jl")

@testset "CUDSS" begin
  @testset "version" begin
    cudss_version()
  end

  @testset "CudssMatrix" begin
    cudss_dense()
    cudss_sparse()
  end

  # @testset "CudssData" begin
  #   cudss_data()
  # end

  @testset "CudssConfig" begin
    cudss_config()
  end

 @testset "CudssSolver" begin
    cudss_solver()
  end

  @testset "CudssExecution" begin
    cudss_execution()
  end
end

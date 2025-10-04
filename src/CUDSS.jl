module CUDSS

using CUDA, CUDA.CUSPARSE, CUDA.CUBLAS
using GPUToolbox
using CUDSS_jll
using LinearAlgebra
using SparseArrays

if haskey(ENV, "JULIA_CUDSS_LIBRARY_PATH") && Sys.islinux()
  const libcudss = joinpath(ENV["JULIA_CUDSS_LIBRARY_PATH"], "libcudss.so")
  const CUDSS_INSTALLATION = "CUSTOM"
else
  using CUDSS_jll
  const CUDSS_INSTALLATION = "YGGDRASIL"
end

import CUDA: libraryPropertyType, cudaDataType, initialize_context, retry_reclaim, CUstream, unsafe_free!
import CUDA.APIUtils: HandleCache
import LinearAlgebra: lu, lu!, ldlt, ldlt!, cholesky, cholesky!, ldiv!, BlasFloat, BlasReal, checksquare, Factorization
import Base: \

abstract type AbstractCudssMatrix{T,INT} end
abstract type AbstractCudssSolver{T,INT} <: Factorization{T} end
const CudssInt = Union{Cint, Int64}

include("libcudss.jl")
include("error.jl")
include("types.jl")
include("helpers.jl")
include("management.jl")
include("interfaces.jl")
include("generic.jl")

end # module CUDSS

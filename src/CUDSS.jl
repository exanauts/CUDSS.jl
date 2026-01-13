module CUDSS

using CUDA, CUDA.CUSPARSE, CUDA.CUBLAS
using GPUToolbox
using CUDSS_jll
using LinearAlgebra
using SparseArrays

if CUDA.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDSS_jll
end

function __init__()
  if CUDA.functional()
    global libcudss
    global CUDSS_INSTALLATION
    if haskey(ENV, "JULIA_CUDSS_LIBRARY_PATH") && Sys.islinux()
      libcudss = joinpath(ENV["JULIA_CUDSS_LIBRARY_PATH"], "libcudss.so")
      CUDSS_INSTALLATION = "CUSTOM"
    elseif CUDA.local_toolkit
      dirs = CUDA_Runtime_Discovery.find_toolkit()
      path = CUDA_Runtime_Discovery.get_library(dirs, "cudss"; optional=true)
      (path === nothing) && error("cuDSS is not available on your system (looked in $(join(dirs, ", "))).")
      libcudss = path
      cudss_version = version()
      if v"0.7.0" <= cudss_version < v"0.8"
        CUDSS_INSTALLATION = "LOCAL"
      else
        libcudss = nothing # Reset the library to nothing because we are failing out (just in case)
        error("cuDSS found: $(path) is version $(cudss_version) but CUDSS.jl expects version 0.7.*")
      end
    else
      !CUDSS_jll.is_available() && error("cuDSS is not available for your platform.")
      libcudss = CUDSS_jll.libcudss
      CUDSS_INSTALLATION = "YGGDRASIL"
    end
  end
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

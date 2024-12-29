module CUDSS

using CUDA, CUDA.APIUtils, CUDA.CUSPARSE, CUDA.CUBLAS
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

import CUDA: @checked, libraryPropertyType, cudaDataType, initialize_context, retry_reclaim, CUstream, @gcsafe_ccall
import LinearAlgebra: lu, lu!, ldlt, ldlt!, cholesky, cholesky!, ldiv!, BlasFloat, BlasReal, checksquare
import Base: \

include("libcudss.jl")
include("error.jl")
include("types.jl")
include("helpers.jl")
include("management.jl")
include("interfaces.jl")
include("generic.jl")

end # module CUDSS

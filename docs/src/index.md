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

## Types

```@docs
CudssMatrix
CudssBatchedMatrix
CudssConfig
CudssData
CudssSolver
CudssBatchedSolver
```

## Functions

```@docs
cudss_set
cudss_get
cudss
```

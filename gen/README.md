# Wrapping headers

This directory contains a script `wrapper.jl` that can be used to automatically
generate wrappers from C headers of NVIDIA cuDSS. This is done using Clang.jl.

In CUDSS.jl, the wrappers need to know whether pointers passed into the
library point to CPU or GPU memory (i.e. `Ptr` or `CuPtr`). This information is
not available from the headers, and instead should be provided by the developer.
The specific information is embedded in the TOML file `cudss.toml`.

# Usage

Either run `julia wrapper.jl` directly, or include it and call the `main()` function.
Be sure to activate the project environment in this folder (`julia --project`), which will install `Clang.jl` and `JuliaFormatter.jl`.

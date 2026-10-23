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

By default the headers are taken from the `CUDSS_jll` artifact. To wrap a different
cuDSS release (for example a local install that is newer than the latest `CUDSS_jll`),
point `CUDSS_INCLUDE_DIR` at its `include` directory:

```bash
cd gen && CUDSS_INCLUDE_DIR=$HOME/.local/opt/cudss/current/include julia --project=. wrapper.jl
```

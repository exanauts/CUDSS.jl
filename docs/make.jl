using Documenter, CUDSS
using LinearAlgebra
using CUDA, CUDA.CUSPARSE

makedocs(
  modules = [CUDSS],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "CUDSS.jl",
  pages = ["Home"        => "index.md",
           "Types"       => "types.md",
           "Functions"   => "functions.md",
           "Generic API" => "generic.md",
           "Batch API"   => "batch.md",
           "Options"     => "options.md"]
)

deploydocs(
  repo = "github.com/exanauts/CUDSS.jl.git",
  push_preview = true,
  devbranch = "main",
)

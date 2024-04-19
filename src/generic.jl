"""
    solver = lu(A::CuSparseMatrixCSR{T,Cint})

Compute the LU factorization of a sparse matrix `A` on an NVIDIA GPU.
The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

#### Input argument

* `A`: a sparse square matrix stored in the `CuSparseMatrixCSR` format.

#### Output argument

* `solver`: an opaque structure [`CudssSolver`](@ref) that stores the factors of the LU decomposition.
"""
function LinearAlgebra.lu(A::CuSparseMatrixCSR{T,Cint}; check = false) where T <: BlasFloat
  n = checksquare(A)
  solver = CudssSolver(A, "G", 'F')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

"""
    solver = lu!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})

Compute the LU factorization of a sparse matrix `A` on an NVIDIA GPU, reusing the symbolic factorization stored in `solver`.
The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.
"""
function LinearAlgebra.lu!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}; check = false) where T <: BlasFloat
  n = checksquare(A)
  cudss_set(solver, A)
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("factorization", solver, x, b)
  return solver
end

"""
    solver = ldlt(A::CuSparseMatrixCSR{T,Cint}; view::Char='F')

Compute the LDLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU.
The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

#### Input argument

* `A`: a sparse Hermitian matrix stored in the `CuSparseMatrixCSR` format.

#### Keyword argument

*`view`: A character that specifies which triangle of the sparse matrix is provided. Possible options are `L` for the lower triangle, `U` for the upper triangle, and `F` for the full matrix.

#### Output argument

* `solver`: Opaque structure [`CudssSolver`](@ref) that stores the factors of the LDLᴴ decomposition.
"""
function LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T,Cint}; view::Char='F', check = false) where T <: BlasFloat
  n = checksquare(A)
  structure = T <: Real ? "S" : "H"
  solver = CudssSolver(A, structure, view)
  (T <: Complex) && cudss_set(solver, "pivot_type", 'N')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

LinearAlgebra.ldlt(A::Symmetric{T,<:CuSparseMatrixCSR{T,Cint}}) where T <: BlasReal = LinearAlgebra.ldlt(A.data, view=A.uplo)
LinearAlgebra.ldlt(A::Hermitian{T,<:CuSparseMatrixCSR{T,Cint}}) where T <: BlasFloat = LinearAlgebra.ldlt(A.data, view=A.uplo)

"""
    solver = ldlt!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})

Compute the LDLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU, reusing the symbolic factorization stored in `solver`.
The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.
"""
function LinearAlgebra.ldlt!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}; check = false) where T <: BlasFloat
  n = checksquare(A)
  cudss_set(solver, A)
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("factorization", solver, x, b)
  return solver
end

"""
    solver = cholesky(A::CuSparseMatrixCSR{T,Cint}; view::Char='F')

Compute the LLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU.
The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.

#### Input argument

* `A`: a sparse Hermitian positive definite matrix stored in the `CuSparseMatrixCSR` format.

#### Keyword argument

*`view`: A character that specifies which triangle of the sparse matrix is provided. Possible options are `L` for the lower triangle, `U` for the upper triangle, and `F` for the full matrix.

#### Output argument

* `solver`: Opaque structure [`CudssSolver`](@ref) that stores the factors of the LLᴴ decomposition.
"""
function LinearAlgebra.cholesky(A::CuSparseMatrixCSR{T,Cint}; view::Char='F', check = false) where T <: BlasFloat
  n = checksquare(A)
  structure = T <: Real ? "SPD" : "HPD"
  solver = CudssSolver(A, structure, view)
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

LinearAlgebra.cholesky(A::Symmetric{T,<:CuSparseMatrixCSR{T,Cint}}; check = false) where T <: BlasReal = LinearAlgebra.cholesky(A.data, view=A.uplo)
LinearAlgebra.cholesky(A::Hermitian{T,<:CuSparseMatrixCSR{T,Cint}}; check = false) where T <: BlasFloat = LinearAlgebra.cholesky(A.data, view=A.uplo)

"""
    solver = cholesky!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint})

Compute the LLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU, reusing the symbolic factorization stored in `solver`.
The type `T` can be `Float32`, `Float64`, `ComplexF32` or `ComplexF64`.
"""
function LinearAlgebra.cholesky!(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T,Cint}; check = false) where T <: BlasFloat
  n = checksquare(A)
  cudss_set(solver, A)
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("factorization", solver, x, b)
  return solver
end

for type in (:CuVector, :CuMatrix)
  @eval begin
    function LinearAlgebra.ldiv!(solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, b, b)
      return b
    end

    function LinearAlgebra.ldiv!(x::$type{T}, solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, x, b)
      return x
    end

    function Base.:\(solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      x = similar(b)
      ldiv!(x, solver, b)
      return x
    end
  end
end

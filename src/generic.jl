"""
    solver = lu(A::CuSparseMatrixCSR{T,INT})

Compute the LU factorization of a sparse matrix `A` on an NVIDIA GPU.
The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

#### Input argument

* `A`: a sparse square matrix stored in the `CuSparseMatrixCSR` format.

#### Output argument

* `solver`: an opaque structure [`CudssSolver`](@ref) that stores the factors of the LU decomposition.
"""
function LinearAlgebra.lu(A::CuSparseMatrixCSR{T,INT}; check = false) where {T <: BlasFloat, INT <: CudssInt}
  n = checksquare(A)
  nbatch = length(A.nzVal) ÷ length(A.rowVal)
  solver = CudssSolver(A, "G", 'F')
  (nbatch > 1) && cudss_set(solver, "ubatch_size", nbatch)
  x = CudssMatrix(T, n; nbatch)
  b = CudssMatrix(T, n; nbatch)
  cudss("analysis", solver, x, b; asynchronous=true)
  cudss("factorization", solver, x, b; asynchronous=false)
  return solver
end

"""
    solver = lu!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT})

Compute the LU factorization of a sparse matrix `A` on an NVIDIA GPU, reusing the symbolic factorization stored in `solver`.
The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.
"""
function LinearAlgebra.lu!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}; check = false) where {T <: BlasFloat, INT <: CudssInt}
  n = checksquare(A)
  nbatch = length(A.nzVal) ÷ length(A.rowVal)
  cudss_update(solver, A)
  x = CudssMatrix(T, n; nbatch)
  b = CudssMatrix(T, n; nbatch)
  phase = solver.fresh_factorization ? "factorization" : "refactorization"
  cudss(phase, solver, x, b; asynchronous=false)
  return solver
end

"""
    solver = ldlt(A::CuSparseMatrixCSR{T,INT}; view::Char='F')

Compute the LDLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU.
The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

#### Input argument

* `A`: a sparse Hermitian matrix stored in the `CuSparseMatrixCSR` format.

#### Keyword argument

*`view`: A character that specifies which triangle of the sparse matrix is provided. Possible options are `L` for the lower triangle, `U` for the upper triangle, and `F` for the full matrix.

#### Output argument

* `solver`: Opaque structure [`CudssSolver`](@ref) that stores the factors of the LDLᴴ decomposition.
"""
function LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T,INT}; view::Char='F', check = false) where {T <: BlasFloat, INT <: CudssInt}
  n = checksquare(A)
  structure = T <: Real ? "S" : "H"
  nbatch = length(A.nzVal) ÷ length(A.rowVal)
  solver = CudssSolver(A, structure, view)
  (nbatch > 1) && cudss_set(solver, "ubatch_size", nbatch)
  x = CudssMatrix(T, n; nbatch)
  b = CudssMatrix(T, n; nbatch)
  cudss("analysis", solver, x, b; asynchronous=true)
  cudss("factorization", solver, x, b; asynchronous=false)
  return solver
end

LinearAlgebra.ldlt(A::Symmetric{T,<:CuSparseMatrixCSR{T,INT}}; check = false) where {T <: BlasReal, INT <: CudssInt} = LinearAlgebra.ldlt(A.data, view=A.uplo)
LinearAlgebra.ldlt(A::Hermitian{T,<:CuSparseMatrixCSR{T,INT}}; check = false) where {T <: BlasFloat, INT <: CudssInt} = LinearAlgebra.ldlt(A.data, view=A.uplo)

"""
    solver = ldlt!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT})

Compute the LDLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU, reusing the symbolic factorization stored in `solver`.
The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.
"""
function LinearAlgebra.ldlt!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}; check = false) where {T <: BlasFloat, INT <: CudssInt}
  n = checksquare(A)
  nbatch = length(A.nzVal) ÷ length(A.rowVal)
  cudss_update(solver, A)
  x = CudssMatrix(T, n; nbatch)
  b = CudssMatrix(T, n; nbatch)
  phase = solver.fresh_factorization ? "factorization" : "refactorization"
  cudss(phase, solver, x, b; asynchronous=false)
  return solver
end

"""
    solver = cholesky(A::CuSparseMatrixCSR{T,INT}; view::Char='F')

Compute the LLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU.
The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

#### Input argument

* `A`: a sparse Hermitian positive definite matrix stored in the `CuSparseMatrixCSR` format.

#### Keyword argument

*`view`: A character that specifies which triangle of the sparse matrix is provided. Possible options are `L` for the lower triangle, `U` for the upper triangle, and `F` for the full matrix.

#### Output argument

* `solver`: Opaque structure [`CudssSolver`](@ref) that stores the factors of the LLᴴ decomposition.
"""
function LinearAlgebra.cholesky(A::CuSparseMatrixCSR{T,INT}; view::Char='F', check = false) where {T <: BlasFloat, INT <: CudssInt}
  n = checksquare(A)
  nbatch = length(A.nzVal) ÷ length(A.rowVal)
  structure = T <: Real ? "SPD" : "HPD"
  solver = CudssSolver(A, structure, view)
  (nbatch > 1) && cudss_set(solver, "ubatch_size", nbatch)
  x = CudssMatrix(T, n; nbatch)
  b = CudssMatrix(T, n; nbatch)
  cudss("analysis", solver, x, b; asynchronous=true)
  cudss("factorization", solver, x, b; asynchronous=false)
  return solver
end

LinearAlgebra.cholesky(A::Symmetric{T,<:CuSparseMatrixCSR{T,INT}}; check = false) where {T <: BlasReal, INT <: CudssInt} = LinearAlgebra.cholesky(A.data, view=A.uplo)
LinearAlgebra.cholesky(A::Hermitian{T,<:CuSparseMatrixCSR{T,INT}}; check = false) where {T <: BlasFloat, INT <: CudssInt} = LinearAlgebra.cholesky(A.data, view=A.uplo)

"""
    solver = cholesky!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT})

Compute the LLᴴ factorization of a sparse matrix `A` on an NVIDIA GPU, reusing the symbolic factorization stored in `solver`.
The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.
"""
function LinearAlgebra.cholesky!(solver::CudssSolver{T,INT}, A::CuSparseMatrixCSR{T,INT}; check = false) where {T <: BlasFloat, INT <: CudssInt}
  n = checksquare(A)
  nbatch = length(A.nzVal) ÷ length(A.rowVal)
  cudss_update(solver, A)
  x = CudssMatrix(T, n; nbatch)
  b = CudssMatrix(T, n; nbatch)
  phase = solver.fresh_factorization ? "factorization" : "refactorization"
  cudss(phase, solver, x, b; asynchronous=false)
  return solver
end

for type in (:CuVector, :CuMatrix)
  @eval begin
    function LinearAlgebra.ldiv!(solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, b, b; asynchronous=false)
      return b
    end

    function LinearAlgebra.ldiv!(x::$type{T}, solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, x, b; asynchronous=false)
      return x
    end

    function Base.:\(solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      x = similar(b)
      ldiv!(x, solver, b)
      return x
    end
  end
end
